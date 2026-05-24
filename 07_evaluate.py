"""
evaluate.py
-----------
Evaluation script for the trained 3D ResNet papilledema classifier.

Computes and reports:
    - Accuracy
    - Precision
    - Recall (Sensitivity)
    - F1 Score
    - Confusion Matrix (TP, FP, TN, FN)
    - Full sklearn classification report

Usage:
    python -m src.evaluate --npy_dir /path/to/npy --checkpoint /path/to/model.pth

Reference:
    Naghibzadeh, K. et al. "Enhancing Optic Nerve Head Abnormality Detection
    through Deep Learning on 3D OCT Images." (MSc Thesis, Tarbiat Modares
    University, 2023)

    Results from thesis:
        Test Accuracy: 93%
        Test F1 Score: 0.87
"""

import argparse
import torch
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from pathlib import Path

from src.model import ResNet3D
from src.dataset import get_dataloaders


# ── Device setup ───────────────────────────────────────────────────────────────

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate(
    npy_dir: str,
    checkpoint_path: str,
    threshold: float = 0.5,
    test_size: float = 0.2,
    batch_size: int = 2,
    random_state: int = 2,
    output_dir: str = "./results",
) -> dict:
    """
    Load a trained model checkpoint and evaluate on the test set.

    Args:
        npy_dir (str): Directory containing preprocessed .npy files.
        checkpoint_path (str): Path to saved model .pth checkpoint.
        threshold (float): Decision threshold for binary predictions. Default: 0.5.
        test_size (float): Fraction of data used for testing. Default: 0.2.
        batch_size (int): Batch size. Default: 2.
        random_state (int): Random seed (must match training). Default: 2.
        output_dir (str): Directory to save the classification report.

    Returns:
        dict: Dictionary of evaluation metrics.
    """
    device = get_device()
    print(f"Using device: {device}")

    # ── Load data ──
    _, test_loader, _ = get_dataloaders(
        npy_dir, test_size=test_size, batch_size=batch_size,
        random_state=random_state
    )

    # ── Load model ──
    model = ResNet3D(num_classes=1, input_shape=(1, 110, 110, 110))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")

    sigmoid = torch.nn.Sigmoid()

    # ── Inference ──
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            probs  = sigmoid(logits).cpu().numpy()
            preds  = (probs > threshold).astype(int).flatten()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy().flatten().astype(int))

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    # ── Metrics ──
    acc       = accuracy_score(all_labels, all_preds)
    f1        = f1_score(all_labels, all_preds, zero_division=0)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall    = recall_score(all_labels, all_preds, zero_division=0)
    cm        = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()

    report = classification_report(
        all_labels, all_preds,
        target_names=["Normal", "Papilledema"],
        zero_division=0
    )

    # ── Print results ──
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"  Accuracy  : {acc:.4f} ({acc*100:.1f}%)")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TP: {tp}  FP: {fp}")
    print(f"    FN: {fn}  TN: {tn}")
    print(f"\n{report}")

    # ── Save report ──
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write("EVALUATION RESULTS\n")
        f.write("="*60 + "\n")
        f.write(f"Accuracy  : {acc:.4f}\n")
        f.write(f"F1 Score  : {f1:.4f}\n")
        f.write(f"Precision : {precision:.4f}\n")
        f.write(f"Recall    : {recall:.4f}\n\n")
        f.write(f"Confusion Matrix:\n")
        f.write(f"  TP: {tp}  FP: {fp}\n")
        f.write(f"  FN: {fn}  TN: {tn}\n\n")
        f.write(report)
    print(f"Report saved to: {report_path}")

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "tp": int(tp), "fp": int(fp),
        "fn": int(fn), "tn": int(tn),
    }


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the trained 3D ResNet papilledema classifier."
    )
    parser.add_argument("--npy_dir",     type=str, required=True,
                        help="Directory containing preprocessed .npy files.")
    parser.add_argument("--checkpoint",  type=str, required=True,
                        help="Path to the model checkpoint (.pth file).")
    parser.add_argument("--threshold",   type=float, default=0.5)
    parser.add_argument("--test_size",   type=float, default=0.2)
    parser.add_argument("--batch_size",  type=int, default=2)
    parser.add_argument("--random_state",type=int, default=2)
    parser.add_argument("--output_dir",  type=str, default="./results")
    args = parser.parse_args()

    evaluate(
        npy_dir=args.npy_dir,
        checkpoint_path=args.checkpoint,
        threshold=args.threshold,
        test_size=args.test_size,
        batch_size=args.batch_size,
        random_state=args.random_state,
        output_dir=args.output_dir,
    )
