import sys, os
# permette "from src..." anche se lanci lo script da qualunque cartella
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import argparse
from pathlib import Path
import numpy as np
from src.train import train_gan  # importa la funzione principale


def main(data_dir: Path, proj_dir: Path, epochs: int, batch_size: int, workers: int):
    X = np.load(data_dir / "X_train_norm.npy")
    S = np.load(data_dir / "S_train_norm.npy")

    print("✅ Dati caricati:", X.shape, S.shape)

    train_gan(
        X_train=X, 
        S_train=S,
        proj_dir=proj_dir,
        batch_size=batch_size,
        epochs=epochs,
        lr=1e-4,
        latent_dim=128,
        d_model=128,
        depth=4,
        nhead=8,
        ff_dim=512,
        dropout=0.1,
        workers=workers,   # su Windows: 0
        use_amp=False,
        r1_every=8,
        r1_gamma=1.0,
        seed=42,
    )

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed", type=Path)
    ap.add_argument("--proj", default=".", type=Path)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()
    main(args.data, args.proj, args.epochs, args.batch_size, args.workers)
    