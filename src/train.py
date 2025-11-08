# ===== src/train.py =====
import os, time, gc
from pathlib import Path
import numpy as np

import torch
from torch import optim
from torch.cuda import amp

# import relativi perché questo file vive dentro il package 'src'
from .dataset import make_loader
from .models import build_models
from .losses import d_hinge_loss, g_hinge_loss
from .utils import set_seed, save_checkpoint, evaluate_metrics

print("[train.py] imported ok", flush=True)

# path di progetto (root = cartella che contiene 'src')
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CKPT_DIR   = PROJECT_ROOT / "outputs" / "checkpoints"
SAMPLE_DIR = PROJECT_ROOT / "outputs" / "samples"
LOG_DIR    = PROJECT_ROOT / "outputs" / "logs"
for d in (CKPT_DIR, SAMPLE_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

def train_gan(
    X_train, S_train,
    proj_dir: str | os.PathLike = PROJECT_ROOT,
    batch_size=128, epochs=5, lr=1e-4,
    latent_dim=128, d_model=128, depth=3, nhead=None, ff_dim=None, dropout=0.1,
    workers=0, use_amp=False, r1_every=8, r1_gamma=1.0, seed=42
):
    """
    Training loop per la cGAN su time-series di traffico.
    Assunzioni: X_train in [-1,1], shape (N, T, C); S_train shape (N, F).
    """
    print("[train_gan] starting...", flush=True)

    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train_gan] device={device}", flush=True)

    # Evita backend SDPA 'efficient'/'flash' che può rompere il backward su alcune build
    if device == "cuda":
        try:
            torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)
            print("[train_gan] SDPA -> flash=False, math=True, mem_efficient=False", flush=True)
        except Exception as e:
            print("[train_gan] SDPA config skipped:", e, flush=True)

    scaler = amp.GradScaler(enabled=(device == "cuda" and use_amp))

    # ===== DataLoader =====
    loader = make_loader(
        X_train, S_train,
        batch_size=batch_size,
        num_workers=workers,          # su Windows/SSH lascia 0/1
        pin_memory=(device == "cuda"),
        shuffle=True
    )

    # ===== Modelli =====
    seq_len, channels = X_train.shape[1], X_train.shape[2]
    cond_dim = S_train.shape[1]
    G, D = build_models(seq_len, channels, cond_dim,
                        latent_dim, d_model, depth, nhead, ff_dim, dropout)
    G.to(device); D.to(device)

    g_opt = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    d_opt = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    proj_dir = Path(proj_dir)
    ckpt_dir   = proj_dir / "outputs" / "checkpoints"
    sample_dir = proj_dir / "outputs" / "samples"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    log_every = 50

    for epoch in range(1, epochs + 1):
        G.train(); D.train()
        for xb, sb in loader:
            xb = xb.to(device, dtype=torch.float32, non_blocking=True)
            sb = sb.to(device, dtype=torch.float32, non_blocking=True)
            B  = xb.size(0)

            # ====== Train D ======
            z = torch.randn(B, latent_dim, device=device)
            with amp.autocast(enabled=(device == "cuda" and use_amp)):
                x_fake = G(z, sb).detach()
                real_scores = D(xb, sb)
                fake_scores = D(x_fake, sb)
                d_loss = d_hinge_loss(real_scores, fake_scores)

            d_opt.zero_grad(set_to_none=True)
            scaler.scale(d_loss).backward()
            scaler.step(d_opt)

            # ====== Train G ======
            z = torch.randn(B, latent_dim, device=device)
            with amp.autocast(enabled=(device == "cuda" and use_amp)):
                x_fake = G(z, sb)
                fake_scores = D(x_fake, sb)
                g_loss = g_hinge_loss(fake_scores)

            g_opt.zero_grad(set_to_none=True)
            scaler.scale(g_loss).backward()
            scaler.step(g_opt)
            scaler.update()

            if step % log_every == 0:
                print(f"[ep {epoch}/{epochs} | step {step}] "
                      f"d_loss={d_loss.item():.4f} | g_loss={g_loss.item():.4f}",
                      flush=True)
            step += 1

        # ===== Fine epoca =====
        if epoch % 2 == 0 or epoch == epochs:
            with torch.no_grad():
                z = torch.randn(4, latent_dim, device=device)
                s = torch.tensor(S_train[:4], device=device, dtype=torch.float32)
                _ = G(z, s).detach().cpu().numpy()

            metrics = evaluate_metrics(G, X_train, S_train,
                                       n_samples=200,
                                       latent_dim=latent_dim,
                                       device=device)
            print(f"[ep {epoch}] MMD={metrics['mmd']:.4f}", flush=True)

            cfg = dict(batch_size=batch_size, epochs=epochs, lr=lr, latent_dim=latent_dim,
                       d_model=d_model, depth=depth, dropout=dropout, use_amp=use_amp, seed=seed,
                       seq_len=seq_len, channels=channels, cond_dim=cond_dim)
            save_checkpoint(str(ckpt_dir / f"ckpt_ep{epoch:03d}.pt"),
                            G, D, g_opt, d_opt, epoch, cfg)

        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    return G, D, cfg
