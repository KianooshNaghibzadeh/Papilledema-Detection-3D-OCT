"""
dataset.py
----------
PyTorch Dataset class for loading preprocessed 3D OCT volumes.

Binary classification:
    Label 1 → Papilledema (abnormal: left_abnormal=0, right_abnormal=1)
    Label 0 → Normal      (normal:   left_normal=2,  right_normal=3)

Usage:
    from src.dataset import OCTDataset, load_data

Reference:
    Naghibzadeh, K. et al. "Enhancing Optic Nerve Head Abnormality Detection
    through Deep Learning on 3D OCT Images." (MSc Thesis, Tarbiat Modares
    University, 2023)
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path


# ── Dataset class ──────────────────────────────────────────────────────────────

class OCTDataset(Dataset):
    """
    PyTorch Dataset for 3D OCT volumes.

    Each volume is returned as a tensor of shape (1, D, H, W),
    where the single channel dimension is required by 3D convolutions.

    Args:
        X (np.ndarray): Array of shape (N, D, H, W).
        Y (np.ndarray): Binary labels of shape (N,).
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (N, 1, D, H, W)
        self.Y = torch.tensor(Y, dtype=torch.float32).unsqueeze(1)  # (N, 1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.Y[idx]


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data(npy_dir: str) -> tuple:
    """
    Load and merge preprocessed .npy arrays into a single dataset.

    Binary label assignment:
        Papilledema (label=1): X0.npy (left abnormal), X1.npy (right abnormal)
        Normal      (label=0): X2.npy (left normal),   X3.npy (right normal)

    Args:
        npy_dir (str): Directory containing X0.npy ... X3.npy files.

    Returns:
        tuple: (X, Y) where X has shape (N, D, H, W) and Y has shape (N,).
    """
    npy_dir = Path(npy_dir)

    # Papilledema samples → label 1
    X_abnorm_l = np.load(npy_dir / "X0.npy")
    X_abnorm_r = np.load(npy_dir / "X1.npy")

    # Normal samples → label 0
    X_norm_l = np.load(npy_dir / "X2.npy")
    X_norm_r = np.load(npy_dir / "X3.npy")

    Y_abnorm_l = np.ones(X_abnorm_l.shape[0])
    Y_abnorm_r = np.ones(X_abnorm_r.shape[0])
    Y_norm_l   = np.zeros(X_norm_l.shape[0])
    Y_norm_r   = np.zeros(X_norm_r.shape[0])

    X = np.concatenate([X_abnorm_l, X_abnorm_r, X_norm_l, X_norm_r], axis=0)
    Y = np.concatenate([Y_abnorm_l, Y_abnorm_r, Y_norm_l, Y_norm_r], axis=0)

    print(f"Dataset loaded: X={X.shape}, Y={Y.shape}")
    print(f"  Papilledema: {int(Y.sum())} | Normal: {int((Y == 0).sum())}")
    return X, Y


def compute_class_weights(Y: np.ndarray) -> tuple:
    """
    Compute class weights for handling class imbalance in the loss function.

    Weight for positive class (papilledema) = proportion of normal samples.
    Weight for negative class (normal)      = proportion of papilledema samples.

    Args:
        Y (np.ndarray): Binary label array.

    Returns:
        tuple: (weight_positive, weight_negative) as floats.
    """
    n_total = len(Y)
    n_positive = Y.sum()
    weight_positive = (n_total - n_positive) / n_total  # weight for label=1
    weight_negative = n_positive / n_total               # weight for label=0
    print(f"Class weights -> positive (papilledema): {weight_positive:.4f} | "
          f"negative (normal): {weight_negative:.4f}")
    return float(weight_positive), float(weight_negative)


def get_dataloaders(
    npy_dir: str,
    test_size: float = 0.2,
    batch_size: int = 2,
    random_state: int = 2,
    device: str = "cpu"
) -> tuple:
    """
    Load data, split into train/test sets, and return DataLoaders.

    Args:
        npy_dir (str): Directory containing preprocessed .npy files.
        test_size (float): Fraction of data for testing. Default: 0.2.
        batch_size (int): Batch size for DataLoaders. Default: 2.
        random_state (int): Random seed for reproducibility. Default: 2.
        device (str): Device to load tensors onto ('cpu' or 'cuda').

    Returns:
        tuple: (train_loader, test_loader, weight_positive)
    """
    X, Y = load_data(npy_dir)
    weight_pos, _ = compute_class_weights(Y)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state
    )

    train_dataset = OCTDataset(X_train, Y_train)
    test_dataset  = OCTDataset(X_test,  Y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    print(f"Train samples: {len(train_dataset)} | Test samples: {len(test_dataset)}")
    return train_loader, test_loader, weight_pos
