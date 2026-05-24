"""
fda_reader.py
-------------
Utility for reading Topcon .fda OCT files using the oct_converter library.

Usage:
    from src.utils.fda_reader import read_fda_volume, inspect_fda_chunks

Reference:
    Naghibzadeh, K. et al. "Enhancing Optic Nerve Head Abnormality Detection
    through Deep Learning on 3D OCT Images." (MSc Thesis, Tarbiat Modares
    University, 2023)
"""

import numpy as np
from pathlib import Path
from oct_converter.readers import FDA


def inspect_fda_chunks(fda_path: str) -> None:
    """
    Print all internal data chunks stored in an .fda file.
    Useful for exploring file structure and verifying file integrity.

    Args:
        fda_path (str): Path to the .fda file.
    """
    fda_path = Path(fda_path)
    if not fda_path.exists():
        raise FileNotFoundError(f"FDA file not found: {fda_path}")

    print(f"Inspecting: {fda_path.name}")
    FDA(str(fda_path), printing=True)


def read_fda_volume(fda_path: str) -> np.ndarray:
    """
    Read a Topcon .fda OCT file and return the 3D volume as a NumPy array.

    Args:
        fda_path (str): Path to the .fda file.

    Returns:
        np.ndarray: 3D OCT volume array of shape (slices, height, width).

    Raises:
        FileNotFoundError: If the .fda file does not exist.
        ValueError: If the volume cannot be read from the file.
    """
    fda_path = Path(fda_path)
    if not fda_path.exists():
        raise FileNotFoundError(f"FDA file not found: {fda_path}")

    volume = FDA(str(fda_path), printing=False).read_oct_volume().volume

    if volume is None:
        raise ValueError(f"Could not read OCT volume from: {fda_path}")

    return np.array(volume)


def batch_read_fda(folder_path: str, verbose: bool = True) -> list:
    """
    Read all .fda files in a folder and return a list of (filename, volume) tuples.

    Args:
        folder_path (str): Path to the folder containing .fda files.
        verbose (bool): Print progress if True.

    Returns:
        list: List of (filename, np.ndarray) tuples.
    """
    folder = Path(folder_path)
    fda_files = sorted(folder.glob("*.fda"))

    if not fda_files:
        raise FileNotFoundError(f"No .fda files found in: {folder}")

    results = []
    for i, fda_file in enumerate(fda_files, 1):
        try:
            volume = read_fda_volume(str(fda_file))
            results.append((fda_file.name, volume))
            if verbose:
                print(f"[{i}/{len(fda_files)}] Read: {fda_file.name} -> shape: {volume.shape}")
        except Exception as e:
            print(f"[{i}/{len(fda_files)}] FAILED: {fda_file.name} -> {e}")

    return results
