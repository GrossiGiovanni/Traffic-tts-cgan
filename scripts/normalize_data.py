import argparse
from pathlib import Path
import numpy as np

def main(raw_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(raw_dir / "X_train.npy")  # (N, T, C)
    S = np.load(raw_dir / "S_train.npy")  # (N, F)

    # --- X: Min-Max su ogni feature → [-1, 1]
    Xf = X.reshape(-1, X.shape[-1]).astype(np.float32)
    X_min = Xf.min(axis=0, keepdims=True)
    X_max = Xf.max(axis=0, keepdims=True)
    Xn = 2 * (Xf - X_min) / (X_max - X_min + 1e-8) - 1
    Xn = Xn.reshape(X.shape).astype(np.float32)

    # --- S: prime 2 colonne numeriche normalizzate, resto (one-hot) intatto
    assert S.shape[1] >= 2
    S_num = S[:, :2].astype(np.float32)
    S_min = S_num.min(axis=0, keepdims=True)
    S_max = S_num.max(axis=0, keepdims=True)
    Sn_num = 2 * (S_num - S_min) / (S_max - S_min + 1e-8) - 1
    S_cat = S[:, 2:]
    Sn = np.concatenate([Sn_num, S_cat], axis=1).astype(np.float32)

    np.save(out_dir / "X_train_norm.npy", Xn)
    np.save(out_dir / "S_train_norm.npy", Sn)
    print("✅ Salvati:", out_dir / "X_train_norm.npy", "e", out_dir / "S_train_norm.npy")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", default="data/raw", type=Path)
    ap.add_argument("--out", dest="out_dir", default="data/processed", type=Path)
    args = ap.parse_args()
    main(args.in_dir, args.out_dir)
