import torch
from torch.utils.data import Dataset, DataLoader


class TrafficDataset(Dataset):
    def __init__(self, X, S):
        assert X.ndim == 3, "X must be (N, L, C)"
        assert S.ndim == 2, "S must be (N, D)"
        assert X.shape[0] == S.shape[0], "X and S must have same N"
        self.X = torch.from_numpy(X).float()
        self.S = torch.from_numpy(S).float()


    def __len__(self):
        return self.X.shape[0]
    
    
    def __getitem__(self, idx):
        return self.X[idx], self.S[idx]




def make_loader(X, S, batch_size=128, num_workers=2, pin_memory=True, shuffle=True):
    ds = TrafficDataset(X, S)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                        num_workers=num_workers, pin_memory=pin_memory, drop_last=True)