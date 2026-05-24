"""
train.py
--------
Training script for the 3D ResNet papilledema classifier.

Hyperparameters (from thesis):
    Optimizer:      SGD (momentum=0.9)
    Learning rate:  0.001 (adaptive via ReduceLROnPlateau)
    Epochs:         20
    Batch size:     2
    Loss:           Weighted BCE (addresses class imbalance)
    Threshold:      0.5 (for binary decision from sigmoid output)
    Best model:     Saved based on validation F1 score

Usage:
    python -m src.train --npy_dir /path/to/npy_files --output_dir ./results

Reference:
    Naghibzadeh, K. et al. "Enhancing Optic Nerve Head Abnormality Detection
    through Deep Learning on 3D OCT Images." (MSc Thesis, Tarbiat Modares
    University, 2023)
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
import torchmetrics
from pathlib import Path

from src.model import ResNet3D, WeightedBCELoss
from src.dataset import get_dataloaders


# ── Device setup ───────────────────────────────────────────────────────────────

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ── Training loop ──────────────────────────────────────────────────────────────

def train(
    npy_dir: str,
    output_dir: str,
    epochs: int = 20,
    batch_size: int = 2,
    learning_rate: float = 0.001,
    test_size: float = 0.2,
    threshold: float = 0.5,
    random_state: int = 2,
) -> None:
    """
    Full training loop with validation, LR scheduling, and best-model checkpointing.

    Args:
        npy_dir (str): Directory with preprocessed .npy files.
        output_dir (str): Directory to save model checkpoints.
        epochs (int): Number of training epochs. Default: 20.
        batch_size (int): Batch size. Default: 2.
        learning_rate (float): Initial learning rate. Default: 0.001.
        test_size (float): Fraction of data for validation. Default: 0.2.
        threshold (float): Decision threshold for binary predictions. Default: 0.5.
        random_state (int): Random seed. Default: 2.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")

    # ── Data ──
    train_loader, test_loader, weight_pos = get_dataloaders(
        npy_dir, test_size=test_size, batch_size=batch_size, random_state=random_state
    )

    # ── Model ──
    model = ResNet3D(num_classes=1, input_shape=(1, 110, 110, 110))
    if device == "cuda":
        model = model.cuda()

    # ── Loss, optimizer, scheduler ──
    criterion = WeightedBCELoss(torch.tensor(weight_pos, dtype=torch.float32)).to(device)
    sigmoid   = nn.Sigmoid()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=3, verbose=True
    )

    # ── Training ──
    max_val_f1 = 0.0

    for epoch in range(epochs):
        # — Train phase —
        model.train()
        train_loss = train_f1 = train_acc = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            logits = model(images)
            preds  = sigmoid(logits)
            loss   = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            preds_binary = (preds > threshold).float()
            train_loss += loss.item()
            train_f1   += torchmetrics.functional.f1_score(
                preds_binary.long(), labels.long(), task="binary"
            ).item()
            train_acc  += torchmetrics.functional.accuracy(
                preds_binary.long(), labels.long(), task="binary"
            ).item()

        # — Validation phase —
        model.eval()
        val_loss = val_f1 = val_acc = 0.0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)

                logits = model(images)
                preds  = sigmoid(logits)
                loss   = criterion(preds, labels)

                preds_binary = (preds > threshold).float()
                val_loss += loss.item()
                val_f1   += torchmetrics.functional.f1_score(
                    preds_binary.long(), labels.long(), task="binary"
                ).item()
                val_acc  += torchmetrics.functional.accuracy(
                    preds_binary.long(), labels.long(), task="binary"
                ).item()

        # — Averages —
        n_train = len(train_loader)
        n_val   = len(test_loader)

        avg_train_loss = train_loss / n_train
        avg_train_f1   = train_f1   / n_train
        avg_train_acc  = train_acc  / n_train
        avg_val_loss   = val_loss   / n_val
        avg_val_f1     = val_f1     / n_val
        avg_val_acc    = val_acc    / n_val

        print(
            f"Epoch [{epoch+1:02d}/{epochs}] "
            f"Train Loss: {avg_train_loss:.4f} | Train F1: {avg_train_f1:.4f} | Train Acc: {avg_train_acc:.4f} || "
            f"Val Loss: {avg_val_loss:.4f} | Val F1: {avg_val_f1:.4f} | Val Acc: {avg_val_acc:.4f}"
        )

        scheduler.step(avg_val_f1)

        # — Checkpoint best model —
        if avg_val_f1 > max_val_f1:
            max_val_f1 = avg_val_f1
            ckpt_path = output_dir / f"best_model_epoch{epoch+1}_f1{max_val_f1:.4f}.pth"
            torch.save(model.state_dict(), str(ckpt_path))
            print(f"  ✓ New best model saved: {ckpt_path.name}")

    print(f"\nTraining complete. Best Val F1: {max_val_f1:.4f}")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train 3D ResNet for papilledema detection from OCT volumes."
    )
    parser.add_argument("--npy_dir",      type=str, required=True,
                        help="Directory containing preprocessed .npy files.")
    parser.add_argument("--output_dir",   type=str, default="./results",
                        help="Directory to save model checkpoints.")
    parser.add_argument("--epochs",       type=int, default=20)
    parser.add_argument("--batch_size",   type=int, default=2)
    parser.add_argument("--lr",           type=float, default=0.001)
    parser.add_argument("--test_size",    type=float, default=0.2)
    parser.add_argument("--threshold",    type=float, default=0.5)
    parser.add_argument("--random_state", type=int, default=2)
    args = parser.parse_args()

    train(
        npy_dir=args.npy_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        test_size=args.test_size,
        threshold=args.threshold,
        random_state=args.random_state,
    )
