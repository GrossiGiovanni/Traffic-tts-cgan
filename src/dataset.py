# src/dataset.py
import os
import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
# scripts/train_cli.py (in cima)
import argparse
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

print("[train_cli] file:", __file__)
print("[train_cli] cwd:", os.getcwd())
print("[train_cli] has src? ", os.path.isdir(os.path.join(os.path.dirname(__file__), "..", "src")))
try:
    import src
    print("[train_cli] import src: OK")
except Exception as e:
    print("[train_cli] import src FAILED:", e)
    raise

class TimeSeriesCndDataset(Dataset):
    """
    Dataset per (X, S) già preprocessati:
      - X: (N, L, C) float32 (idealmente in [-1, 1])
      - S: (N, d_s) float32 (prime colonne numeriche in [-1, 1], OHE intatte)
    Supporta memmap (np.load(..., mmap_mode='r')) per non saturare la RAM.
    """
    def __init__(self, X: np.ndarray, S: np.ndarray):
        assert isinstance(X, np.ndarray) and isinstance(S, np.ndarray), "Passa gli array già caricati"
        assert X.shape[0] == S.shape[0], f"X and S size mismatch: {X.shape[0]} vs {S.shape[0]}"
        assert X.ndim == 3, f"X shape attesa (N,L,C), trovato {X.shape}"
        assert S.ndim == 2, f"S shape attesa (N,d), trovato {S.shape}"
        self.X = X
        self.S = S

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        # Zero-copy verso torch (evita .copy())
        x = torch.as_tensor(self.X[idx], dtype=torch.float32)
        s = torch.as_tensor(self.S[idx], dtype=torch.float32)
        return x, s


def _safe_splits(N: int, val_ratio: float, test_ratio: float):
    """Calcola split non vuoti e che sommano a N."""
    n_test = int(N * test_ratio)
    n_val  = int(N * val_ratio)
    n_train = N - n_val - n_test
    # garantisci almeno 1 elemento a split se possibile
    if n_train <= 0 and N >= 3:
        n_train = 1
    if n_val <= 0 and N - n_train >= 2:
        n_val = 1
    if n_test <= 0 and N - n_train - n_val >= 1:
        n_test = 1
    # riequilibra per sommare esattamente a N
    diff = N - (n_train + n_val + n_test)
    n_train += diff
    return n_train, n_val, n_test


def get_loaders(
    processed_dir: str,
    batch_size: int = 128,
    num_workers: int = 2,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    pin_memory: bool = True,
):
    """
    Carica X.npy / S.npy UNA SOLA VOLTA (con memmap) e costruisce i DataLoader.
    - train: drop_last=True (per batch perfetti in GAN)
    - val/test: drop_last=False (non perdere dati)
    """
    x_path = os.path.join(processed_dir, "X.npy")
    s_path = os.path.join(processed_dir, "S.npy")
    if not os.path.isfile(x_path) or not os.path.isfile(s_path):
        raise FileNotFoundError(f"Mancano X.npy / S.npy in {processed_dir}")

    # memmap: lettura on-demand, non occupa tutta la RAM
    X = np.load(x_path, mmap_mode="r")
    S = np.load(s_path, mmap_mode="r")

    # dataset unico (niente doppio caricamento)
    ds = TimeSeriesCndDataset(X, S)
    N = len(ds)

    n_train, n_val, n_test = _safe_splits(N, val_ratio, test_ratio)
    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(ds, [n_train, n_val, n_test], generator=g)

    def make_loader(d, shuffle, drop_last):
        return DataLoader(
            d,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            persistent_workers=(num_workers > 0),
        )

    train_loader = make_loader(train_ds, shuffle=True,  drop_last=True)
    val_loader   = make_loader(val_ds,   shuffle=False, drop_last=False)
    test_loader  = make_loader(test_ds,  shuffle=False, drop_last=False)

    # stampa diagnostica utile una volta
    print(f"✅ Dati caricati: X={tuple(X.shape)} S={tuple(S.shape)} | "
          f"split: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    return train_loader, val_loader, test_loader
