
import os, math, json, time, random
import numpy as np
import torch
from matplotlib import pyplot as plt




def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    



def save_checkpoint(path, G, D, g_opt, d_opt, epoch, cfg):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'G': G.state_dict(),
        'D': D.state_dict(),
        'g_opt': g_opt.state_dict(),
        'd_opt': d_opt.state_dict(),
        'cfg': cfg,
    }, path)




def load_checkpoint(path, G, D, g_opt=None, d_opt=None, map_location='cpu'):
    ckpt = torch.load(path, map_location=map_location)
    G.load_state_dict(ckpt['G'])
    D.load_state_dict(ckpt['D'])
    if g_opt is not None and d_opt is not None:
        g_opt.load_state_dict(ckpt['g_opt'])
        d_opt.load_state_dict(ckpt['d_opt'])
    return ckpt.get('epoch', 0), ckpt.get('cfg', {})




def generate(G, S, n_samples=None, latent_dim=128, device='cuda'):
    G.eval()
    with torch.no_grad():
        S = torch.as_tensor(S, dtype=torch.float32, device=device)
        if n_samples is None:
            n_samples = S.size(0)
        z = torch.randn(n_samples, latent_dim, device=device)
        X_hat = G(z, S[:n_samples]) # [N,L,C]
        return X_hat.detach().cpu().numpy()




def quick_plot_pairs(real, fake, num_series=3, title_prefix=""):
    L = real.shape[0]
    t = np.arange(L)
    cols = min(num_series, real.shape[1])
    plt.figure(figsize=(12, 3*cols))
    for c in range(cols):
        plt.subplot(cols, 1, c+1)
        plt.plot(t, real[:, c], label='real')
        plt.plot(t, fake[:, c], label='fake', alpha=0.8)
        plt.xlabel('t'); plt.ylabel(f'ch{c}')
        plt.legend()
    plt.suptitle(f"{title_prefix} — overlay real vs fake")
    plt.tight_layout(); plt.show()



def evaluate_metrics(G, X_real, S_real, n_samples=100, latent_dim=128, device="cuda"):
    """
    Calcola metriche di base per confrontare i dati reali con quelli generati dal Generatore.

    Parametri:
    - G: modello Generator (già addestrato, in eval mode)
    - X_real: array numpy con sequenze reali (N, T, C)
    - S_real: array numpy con feature statiche (N, F)
    - n_samples: quanti campioni usare per la valutazione (default=200)
    - latent_dim: dimensione del rumore latente
    - device: 'cuda' o 'cpu'

    Ritorna:
    - dizionario con MMD e statistiche base (media/std reali vs fake)
    """
    import numpy as np, torch
    G.eval()

    # 1) Scegliamo un sottoinsieme casuale di dati reali
    n_total = len(X_real)
    idx = np.random.choice(n_total, size=min(n_samples, n_total), replace=False)

    # 2) Convertiamo i dati scelti in tensori PyTorch sul device giusto
    Xr = torch.tensor(X_real[idx], dtype=torch.float32, device=device)  # sequenze reali
    Sr = torch.tensor(S_real[idx], dtype=torch.float32, device=device)  # feature statiche reali

    # 3) Generiamo sequenze finte con il generatore, usando lo stesso subset di condizioni
    with torch.no_grad():
        z = torch.randn(len(idx), latent_dim, device=device)  # rumore latente
        Xg = G(z, Sr).detach().cpu().numpy()                 # sequenze generate

    # 4) Flatten per confronto statistico (collassa tempo e batch in un'unica dimensione)
    real_flat = Xr.cpu().numpy().reshape(-1, Xr.shape[-1])   # (N*T, C)
    fake_flat = Xg.reshape(-1, Xg.shape[-1])                 # (N*T, C)

    # 5) Calcolo MMD (distanza tra distribuzioni reali e generate)
    mmd_val = rbf_mmd(torch.tensor(real_flat), torch.tensor(fake_flat))

    # 6) Statistiche base: media e deviazione standard per ogni canale
    stats = {}
    for c in range(real_flat.shape[1]):
        stats[f"ch{c}_mean_real"] = float(real_flat[:, c].mean())
        stats[f"ch{c}_mean_fake"] = float(fake_flat[:, c].mean())
        stats[f"ch{c}_std_real"]  = float(real_flat[:, c].std())
        stats[f"ch{c}_std_fake"]  = float(fake_flat[:, c].std())

    # 7) Ritorniamo tutte le metriche in un dizionario
    return {"mmd": mmd_val, **stats}



def rbf_mmd(x, y, sigma=1.0):
    # x: [N,D], y: [M,D]
    import torch
    def pdist(a, b):
        a2 = (a*a).sum(1, keepdim=True)
        b2 = (b*b).sum(1, keepdim=True)
        return a2 + b2.T - 2*a@b.T
    Kxx = torch.exp(-pdist(x, x)/(2*sigma**2))
    Kyy = torch.exp(-pdist(y, y)/(2*sigma**2))
    Kxy = torch.exp(-pdist(x, y)/(2*sigma**2))
    mmd = Kxx.mean() + Kyy.mean() - 2*Kxy.mean()
    return mmd.item()