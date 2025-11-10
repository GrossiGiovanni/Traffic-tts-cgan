 # src/metrics.py
import numpy as np
from scipy.spatial.distance import cdist

def gaussian_mmd(x, y, sigma=1.0):
    # x,y: (N, D)
    xx = cdist(x, x, 'sqeuclidean')
    yy = cdist(y, y, 'sqeuclidean')
    xy = cdist(x, y, 'sqeuclidean')
    Kxx = np.exp(-xx/(2*sigma**2)).mean()
    Kyy = np.exp(-yy/(2*sigma**2)).mean()
    Kxy = np.exp(-xy/(2*sigma**2)).mean()
    return float(Kxx + Kyy - 2*Kxy)

def dtw_distance_1d(a, b):
    L1, L2 = len(a), len(b)
    D = np.full((L1+1, L2+1), np.inf); D[0,0]=0
    for i in range(1, L1+1):
        ai = a[i-1]
        for j in range(1, L2+1):
            cost = abs(ai - b[j-1])
            D[i,j] = cost + min(D[i-1,j], D[i,j-1], D[i-1,j-1])
    return float(D[L1, L2])

def acf(x, max_lag=10):
    x = x - x.mean()
    denom = (x*x).sum() + 1e-12
    ac = [1.0]
    for k in range(1, max_lag+1):
        ac.append(float(np.dot(x[:-k], x[k:]) / denom))
    return np.array(ac)

def acf_distance(real, fake, max_lag=10):
    # real,fake: (N, L) 1D feature per sequenza
    r_acf = np.stack([acf(r, max_lag) for r in real], 0).mean(0)
    f_acf = np.stack([acf(f, max_lag) for f in fake], 0).mean(0)
    return float(np.abs(r_acf - f_acf).mean())

def psd_distance(real, fake):
    # confronto spettri medi (L deve essere uguale)
    Fr = np.abs(np.fft.rfft(real, axis=1)).mean(0)
    Ff = np.abs(np.fft.rfft(fake, axis=1)).mean(0)
    Fr /= (Fr.sum() + 1e-12); Ff /= (Ff.sum() + 1e-12)
    return float(np.abs(Fr - Ff).mean())

def violation_rate(x, v_idx=None, vmax=None, a_idx=None, amax=None):
    # x: (N, L, C)
    N, L, C = x.shape
    bad = 0
    tot = N*L
    if v_idx is not None:
        v = x[..., v_idx]
        bad += (v < 0).sum()
        if vmax is not None:
            bad += (v > vmax).sum()
    if a_idx is not None and amax is not None:
        a = np.abs(x[..., a_idx])
        bad += (a > amax).sum()
    return float(bad) / float(tot)

def diversity_ratio(fake, real):
    # var(fake)/var(real) su flatten (per-feature sarebbe meglio, ma semplice è robusto)
    rf = real.reshape(len(real), -1)
    ff = fake.reshape(len(fake), -1)
    vr = (ff.var(axis=0).mean() + 1e-12) / (rf.var(axis=0).mean() + 1e-12)
    return float(vr)
