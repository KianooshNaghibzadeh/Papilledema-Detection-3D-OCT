"""
preprocessing.py
----------------
Preprocessing pipeline for 3D OCT volumes:
  1. Read .fda files per class folder
  2. Crop to the Optic Nerve Head (ONH) region of interest
  3. Resize to a fixed cubic volume (110 x 110 x 110)
  4. Normalize pixel values to [0, 1]
  5. Save per-class NumPy arrays (.npy)

Crop coordinates (determined via MRIcroGL inspection):
    Axis X (slices):  [40:190]
    Axis Y (height):  [60:850]
    Axis Z (width):   [130:400]

Usage:
    python -m src.preprocessing --data_dir /path/to/data --output_dir /path/to/output

Reference:
    Naghibzadeh, K. et al. "Enhancing Optic Nerve Head Abnormality Detection
    through Deep Learning on 3D OCT Images." (MSc Thesis, Tarbiat Modares
    University, 2023)
"""

import os
import argparse
import numpy as np
from pathlib import Path
from skimage.transform import resize
from src.utils.fda_reader import read_fda_volume

# ── Constants ──────────────────────────────────────────────────────────────────

# ONH crop coordinates identified via MRIcroGL visualization
CROP_X = (40, 190)
CROP_Y = (60, 850)
CROP_Z = (130, 400)

# Target voxel resolution after resizing
TARGET_SHAPE = (110, 110, 110)

# Class label mapping
CLASS_LABELS = {
    "left_abnormal":  0,
    "right_abnormal": 1,
    "left_normal":    2,
    "right_normal":   3,
    "left_suspicious":  4,
    "right_suspicious": 5,
}


# ── Core functions ─────────────────────────────────────────────────────────────

def crop_volume(volume: np.ndarray) -> np.ndarray:
    """
    Crop a 3D OCT volume to the ONH region of interest.

    Args:
        volume (np.ndarray): Raw 3D volume of shape (slices, height, width).

    Returns:
        np.ndarray: Cropped volume.
    """
    return volume[CROP_X[0]:CROP_X[1],
                  CROP_Y[0]:CROP_Y[1],
                  CROP_Z[0]:CROP_Z[1]]


def resize_volume(volume: np.ndarray, target_shape: tuple = TARGET_SHAPE) -> np.ndarray:
    """
    Resize a 3D volume to the target shape using anti-aliased interpolation.

    Args:
        volume (np.ndarray): Input 3D volume.
        target_shape (tuple): Desired output shape. Default: (110, 110, 110).

    Returns:
        np.ndarray: Resized volume as float64.
    """
    return resize(volume, target_shape, anti_aliasing=True)


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    """
    Normalize voxel intensities to [0, 1] by dividing by 255.

    Args:
        volume (np.ndarray): Input 3D volume with values in [0, 255].

    Returns:
        np.ndarray: Normalized volume with values in [0, 1].
    """
    return volume / 255.0


def preprocess_volume(volume: np.ndarray) -> np.ndarray:
    """
    Full preprocessing pipeline: crop → normalize → resize.

    Args:
        volume (np.ndarray): Raw 3D OCT volume.

    Returns:
        np.ndarray: Preprocessed volume of shape TARGET_SHAPE.
    """
    cropped = crop_volume(volume)
    normalized = normalize_volume(cropped.astype(np.float64))
    resized = resize_volume(normalized)
    return resized


def process_class_folder(
    class_folder: str,
    output_dir: str,
    label: int,
    verbose: bool = True
) -> None:
    """
    Process all .fda files in a class folder and save as a single .npy array.

    Args:
        class_folder (str): Path to folder containing .fda files for one class.
        output_dir (str): Directory to save the output .npy file.
        label (int): Integer label for this class.
        verbose (bool): Print progress if True.
    """
    class_folder = Path(class_folder)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fda_files = sorted(class_folder.glob("*.fda"))
    if not fda_files:
        print(f"  No .fda files found in: {class_folder}")
        return

    volumes = []
    for i, fda_file in enumerate(fda_files, 1):
        try:
            raw = read_fda_volume(str(fda_file))
            processed = preprocess_volume(raw)
            volumes.append(processed)
            if verbose:
                print(f"  [{i}/{len(fda_files)}] Processed: {fda_file.name}")
        except Exception as e:
            print(f"  [{i}/{len(fda_files)}] FAILED: {fda_file.name} -> {e}")

    if volumes:
        X = np.array(volumes, dtype=np.float32)
        out_path = output_dir / f"X{label}.npy"
        np.save(str(out_path), X)
        print(f"  Saved: {out_path} | Shape: {X.shape}")


def run_full_pipeline(data_dir: str, output_dir: str) -> None:
    """
    Run the full preprocessing pipeline over all class folders.

    Expected data_dir structure:
        data_dir/
            left_abnormal/   *.fda
            right_abnormal/  *.fda
            left_normal/     *.fda
            right_normal/    *.fda
            left_suspicious/ *.fda (optional)
            right_suspicious/*.fda (optional)

    Args:
        data_dir (str): Root directory containing per-class subdirectories.
        output_dir (str): Directory to save output .npy files.
    """
    data_dir = Path(data_dir)
    for class_name, label in CLASS_LABELS.items():
        class_folder = data_dir / class_name
        if not class_folder.exists():
            print(f"Skipping missing folder: {class_folder}")
            continue
        print(f"\nProcessing class: {class_name} (label={label})")
        process_class_folder(str(class_folder), output_dir, label)


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess 3D OCT .fda files for papilledema classification."
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Root directory with per-class subdirectories of .fda files."
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to save preprocessed .npy arrays."
    )
    args = parser.parse_args()
    run_full_pipeline(args.data_dir, args.output_dir)
