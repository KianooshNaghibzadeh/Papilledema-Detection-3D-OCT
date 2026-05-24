"""
nifti_converter.py
------------------
Utility for converting Topcon .fda OCT files to NIfTI format (.nii.gz)
for visualization in tools such as MRIcroGL.

Usage:
    from src.utils.nifti_converter import convert_fda_to_nifti, batch_convert_to_nifti

Reference:
    Naghibzadeh, K. et al. "Enhancing Optic Nerve Head Abnormality Detection
    through Deep Learning on 3D OCT Images." (MSc Thesis, Tarbiat Modares
    University, 2023)
"""

import numpy as np
import nibabel as nib
from pathlib import Path
from src.utils.fda_reader import read_fda_volume


def convert_fda_to_nifti(
    fda_path: str,
    output_path: str,
    affine: np.ndarray = None
) -> None:
    """
    Convert a single .fda OCT file to a NIfTI (.nii.gz) file.

    Args:
        fda_path (str): Path to the input .fda file.
        output_path (str): Path to save the output .nii.gz file.
        affine (np.ndarray, optional): 4x4 affine matrix. Defaults to identity.
    """
    if affine is None:
        affine = np.eye(4)

    volume = read_fda_volume(fda_path)
    nifti_image = nib.Nifti1Image(volume, affine)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nifti_image, str(output_path))
    print(f"Saved NIfTI: {output_path}")


def batch_convert_to_nifti(
    input_folder: str,
    output_folder: str,
    affine: np.ndarray = None
) -> None:
    """
    Convert all .fda files in a folder to NIfTI format.

    Args:
        input_folder (str): Folder containing .fda files.
        output_folder (str): Folder to save .nii.gz output files.
        affine (np.ndarray, optional): 4x4 affine matrix. Defaults to identity.
    """
    if affine is None:
        affine = np.eye(4)

    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    fda_files = sorted(input_folder.glob("*.fda"))
    if not fda_files:
        raise FileNotFoundError(f"No .fda files found in: {input_folder}")

    for i, fda_file in enumerate(fda_files, 1):
        try:
            output_path = output_folder / f"{fda_file.stem}.nii.gz"
            convert_fda_to_nifti(str(fda_file), str(output_path), affine)
            print(f"[{i}/{len(fda_files)}] Converted: {fda_file.name}")
        except Exception as e:
            print(f"[{i}/{len(fda_files)}] FAILED: {fda_file.name} -> {e}")
