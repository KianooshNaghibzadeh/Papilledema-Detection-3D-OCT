"""
predict.py
----------
Inference script for the trained 3D ResNet papilledema classifier.

Runs prediction on a folder of preprocessed .npy files or raw .fda files,
outputting per-sample binary predictions (0 = Normal, 1 = Papilledema)
and probability scores.

Trained checkpoint: papilledema_classifier.pth
    - Saved at epoch 5
    - Validation F1: 0.6818 (on suspicious/unlabelled held-out samples)
    - Full test set performance: Accuracy=93%, F1=0.87

Usage:
    # From preprocessed .npy file:
    python -m src.predict \
        --input X4.npy \
        --checkpoint papilledema_classifier.pth

    # From raw .fda files:
    python -m src.predict \
        --input /path/to/fda_folder \
        --checkpoint papilledema_classifier.pth \
        --input_type fda

Reference:
    Naghibzadeh, K. et al. "Enhancing Optic Nerve Head Abnormality Detection
    through Deep Learning on 3D OCT Images." (MSc Thesis, Tarbiat Modares
    University, 2023)
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from skimage.transform import resize

from src.model import ResNet3D
from src.utils.fda_reader import read_fda_volume
from src.preprocessing import preprocess_volume


# ── Constants ──────────────────────────────────────────────────────────────────

LABEL_MAP = {0: "Normal", 1: "Papilledema"}
TARGET_SHAPE = (110, 110, 110)


# ── Device ─────────────────────────────────────────────────────────────────────

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: str) -> nn.Module:
    """
    Load the trained 3D ResNet from a checkpoint file.

    Args:
        checkpoint_path (str): Path to the .pth checkpoint file.
        device (str): Device to load the model onto.

    Returns:
        nn.Module: Model in eval mode.
    """
    model = ResNet3D(num_classes=1, input_shape=(1, 110, 110, 110))
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"Model loaded from: {checkpoint_path}")
    return model


# ── Data loading ───────────────────────────────────────────────────────────────

def load_npy(npy_path: str) -> np.ndarray:
    """
    Load a preprocessed .npy volume array and resize if needed.

    Args:
        npy_path (str): Path to .npy file of shape (N, D, H, W).

    Returns:
        np.ndarray: Array of shape (N, 110, 110, 110).
    """
    X = np.load(npy_path)
    print(f"Loaded .npy: {npy_path} | Shape: {X.shape} | Max: {X.max():.4f}")

    # Resize if not already at target shape
    if X.shape[1:] != TARGET_SHAPE:
        print(f"Resizing from {X.shape[1:]} → {TARGET_SHAPE} ...")
        resized = []
        for i in range(X.shape[0]):
            resized.append(resize(X[i], TARGET_SHAPE, anti_aliasing=True))
            print(f"  Resized [{i+1}/{X.shape[0]}]")
        X = np.array(resized, dtype=np.float32)

    return X.astype(np.float32)


def load_fda_folder(fda_folder: str) -> tuple:
    """
    Load and preprocess all .fda files in a folder.

    Args:
        fda_folder (str): Path to folder containing .fda files.

    Returns:
        tuple: (X array of shape (N, 110, 110, 110), list of filenames)
    """
    folder = Path(fda_folder)
    fda_files = sorted(folder.glob("*.fda"))

    if not fda_files:
        raise FileNotFoundError(f"No .fda files found in: {folder}")

    volumes = []
    filenames = []
    for i, fda_file in enumerate(fda_files, 1):
        try:
            raw = read_fda_volume(str(fda_file))
            processed = preprocess_volume(raw)
            volumes.append(processed.astype(np.float32))
            filenames.append(fda_file.name)
            print(f"[{i}/{len(fda_files)}] Preprocessed: {fda_file.name}")
        except Exception as e:
            print(f"[{i}/{len(fda_files)}] FAILED: {fda_file.name} -> {e}")

    return np.array(volumes, dtype=np.float32), filenames


# ── Inference ──────────────────────────────────────────────────────────────────

def predict(
    X: np.ndarray,
    model: nn.Module,
    device: str,
    threshold: float = 0.5,
    filenames: list = None,
) -> list:
    """
    Run inference on a batch of preprocessed 3D volumes.

    Processes one sample at a time to minimize GPU memory usage.

    Args:
        X (np.ndarray): Array of shape (N, 110, 110, 110).
        model (nn.Module): Trained model in eval mode.
        device (str): Device to run inference on.
        threshold (float): Decision threshold. Default: 0.5.
        filenames (list): Optional list of filenames for reporting.

    Returns:
        list: List of dicts with keys: sample, probability, prediction, label.
    """
    sigmoid = nn.Sigmoid()
    results = []

    print(f"\nRunning inference on {X.shape[0]} samples (threshold={threshold})...")
    print("-" * 60)

    with torch.no_grad():
        for i in range(X.shape[0]):
            # Shape: (1, 1, 110, 110, 110) — batch=1, channel=1
            volume = torch.tensor(X[i:i+1], dtype=torch.float32)
            volume = volume.unsqueeze(1).to(device)  # add channel dim

            logit = model(volume)
            prob  = sigmoid(logit).item()
            pred  = int(prob > threshold)
            label = LABEL_MAP[pred]

            name = filenames[i] if filenames else f"sample_{i:04d}"
            results.append({
                "sample":      name,
                "probability": round(prob, 4),
                "prediction":  pred,
                "label":       label,
            })

            print(f"  [{i+1:>3}/{X.shape[0]}] {name:<30} "
                  f"prob={prob:.4f}  →  {label}")

    return results


def save_results(results: list, output_path: str) -> None:
    """
    Save prediction results to a text file.

    Args:
        results (list): List of result dicts from predict().
        output_path (str): Path to save the output .txt file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_papilledema = sum(1 for r in results if r["prediction"] == 1)
    n_normal      = sum(1 for r in results if r["prediction"] == 0)

    with open(output_path, "w") as f:
        f.write("PREDICTION RESULTS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total samples:  {len(results)}\n")
        f.write(f"Papilledema:    {n_papilledema}\n")
        f.write(f"Normal:         {n_normal}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"{'Sample':<35} {'Probability':>12} {'Prediction':>15}\n")
        f.write("-" * 65 + "\n")
        for r in results:
            f.write(f"{r['sample']:<35} {r['probability']:>12.4f} {r['label']:>15}\n")

    print(f"\nResults saved to: {output_path}")

    # Also print summary
    print(f"\nSUMMARY")
    print(f"  Total:       {len(results)}")
    print(f"  Papilledema: {n_papilledema} ({100*n_papilledema/len(results):.1f}%)")
    print(f"  Normal:      {n_normal} ({100*n_normal/len(results):.1f}%)")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run papilledema prediction on 3D OCT volumes."
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to a .npy file or a folder of .fda files."
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to trained model checkpoint (.pth file)."
    )
    parser.add_argument(
        "--input_type", type=str, default="npy", choices=["npy", "fda"],
        help="Input type: 'npy' for preprocessed array, 'fda' for raw OCT files. Default: npy"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Decision threshold for binary classification. Default: 0.5"
    )
    parser.add_argument(
        "--output", type=str, default="./results/predictions.txt",
        help="Path to save prediction results. Default: ./results/predictions.txt"
    )
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}\n")

    # Load model
    model = load_model(args.checkpoint, device)

    # Load data
    filenames = None
    if args.input_type == "npy":
        X = load_npy(args.input)
    else:
        X, filenames = load_fda_folder(args.input)

    # Run inference
    results = predict(X, model, device, threshold=args.threshold, filenames=filenames)

    # Save results
    save_results(results, args.output)
