# scripts/train_cli.py
import argparse
import sys
import torch
from src.dataset import get_loaders
from src.train import train_gan
from src.utils import set_seed


def parse_args():
    ap = argparse.ArgumentParser(description="Train hybrid TTS-cGAN model")
    ap.add_argument("--processed_dir", type=str, default="data/processed",
                    help="Directory contenente X.npy e S.npy normalizzati")
    ap.add_argument("--out_dir", type=str, default="Outputs",
                    help="Directory per logs e checkpoints")

    # dimensioni dei dati
    ap.add_argument("--seq_len", type=int, default=120)
    ap.add_argument("--x_dim", type=int, default=5)
    ap.add_argument("--s_dim", type=int, default=4)
    ap.add_argument("--z_dim", type=int, default=64)

    # architettura
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--n_layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1)

    # training
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)

    # vincoli fisici
    ap.add_argument("--v_idx", type=int, default=2, help="Indice della colonna velocità in X")
    ap.add_argument("--vmax", type=float, default=1.0)
    ap.add_argument("--a_idx", type=int, default=4, help="Indice della colonna accelerazione in X")
    ap.add_argument("--amax", type=float, default=1.0)

    # nuovo parametro per selezionare CPU o CUDA
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                    help="Dispositivo su cui eseguire il training (cpu o cuda)")
    # scripts/train_cli.py (aggiungi args)
    ap.add_argument("--export_every", type=int, default=1, help="Salva sample/plot ogni N epoche")
    ap.add_argument("--use_amp", action="store_true", help="Abilita AMP su CUDA")

    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    # verifica disponibilità CUDA
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[ERRORE] Hai richiesto --device cuda ma torch.cuda.is_available() = False.")
        print("→ Verifica con 'nvidia-smi' o reinstallare PyTorch con la versione CUDA compatibile.")
        print("→ In alternativa, lancia con '--device cpu'.")
        sys.exit(1)

    train_loader, val_loader, test_loader = get_loaders(
        args.processed_dir, batch_size=args.batch_size, num_workers=2,
        val_ratio=0.1, test_ratio=0.1, seed=args.seed
    )

    score = train_gan(
        train_loader, val_loader, out_dir=args.out_dir,
        seq_len=args.seq_len, x_dim=args.x_dim, s_dim=args.s_dim, z_dim=args.z_dim,
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers, dropout=args.dropout,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        v_idx=args.v_idx, vmax=args.vmax, a_idx=args.a_idx, amax=args.amax,
        device=args.device, seed=args.seed,
        export_every=args.export_every, use_amp=args.use_amp
    )   

    print(f"\n[TRAIN COMPLETATO] Best composite score (MMD+DTW): {score:.4f}")


if __name__ == "__main__":
    main()
