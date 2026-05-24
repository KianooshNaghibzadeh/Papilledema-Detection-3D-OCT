# Papilledema Detection from 3D OCT Images using Deep Learning

A deep learning pipeline for automated binary classification of **Optic Nerve Head (ONH)** 3D Optical Coherence Tomography (OCT) images, detecting the presence or absence of **papilledema** (optic disc swelling), a key biomarker of Idiopathic Intracranial Hypertension (IIH).

---

## Clinical Background

**Papilledema** is swelling of the optic disc caused by elevated intracranial pressure (ICP), most commonly associated with **Idiopathic Intracranial Hypertension (IIH)**. Delayed diagnosis can lead to irreversible vision loss. Manual grading (Frisén scale) via ophthalmoscopy is subjective and time-consuming.

**OCT imaging** of the ONH provides high-resolution 3D volumetric data that can capture subtle structural changes. This project automates the interpretation of these 3D volumes using a custom 3D ResNet, reducing reliance on specialist availability and human error.

---

## Results

| Metric | Train | Test |
|---|---|---|
| Accuracy | 99.5% | **93.0%** |
| F1 Score | 0.899 | **0.870** |
| Loss | 0.007 | 0.106 |

Evaluated on a held-out test set (20% split, random_state=2) from a dataset of 795 3D OCT volumes.

---

## Dataset

- **Source:** 795 `.fda` files collected from two Iranian ophthalmology clinics (*Negah Eye Clinic* and *Abu Rayhan Medical Clinic*, Tehran)
- **Labelling:** Performed independently by an ophthalmologist and a neurologist
- **Class distribution:**

| Class | Count | Proportion |
|---|---|---|
| Papilledema (abnormal) | 561 | 70.6% |
| Normal | 234 | 29.4% |
| **Total** | **795** | — |

- **Format:** Topcon `.fda` proprietary OCT format
- **Class imbalance handling:** Weighted BCE loss (positive weight = 0.294, negative weight = 0.706)

> **Note:** The dataset is not publicly available due to patient privacy. A sample `.fda` file is provided in `data/sample/` for testing the pipeline.

---

## Repository Structure

```
Papilledema-Detection-3D-OCT/
│
├── README.md
├── requirements.txt
│
├── data/
│   └── sample/
│       └── 100.fda                    # Example FDA file for testing
│
├── notebooks/                                        # Exploratory Jupyter notebooks (follow in order)
│   ├── 01_explore_fda_file_structure.ipynb           # Step 1: Inspect internal chunks of a raw .fda file
│   ├── 02_convert_fda_to_nifti.ipynb                 # Step 2: Convert .fda files to NIfTI for visualization
│   ├── 03_read_fda_to_numpy_arrays.ipynb             # Step 3: Crop, resize, normalize and save as .npy
│   └── 04_train_3d_resnet_classifier.ipynb           # Step 4: Train and evaluate the 3D ResNet model
│
├── src/
│   ├── utils/                                        # Reusable file I/O utilities
│   │   ├── 01_fda_reader.py                          # Read Topcon .fda OCT files into NumPy arrays
│   │   └── 02_nifti_converter.py                     # Batch convert .fda files to NIfTI (.nii.gz)
│   │
│   ├── 03_preprocessing.py                           # Full preprocessing pipeline: crop → normalize → resize → save .npy
│   ├── 04_dataset.py                                 # PyTorch Dataset class and DataLoader factory
│   ├── 05_model.py                                   # Custom 3D ResNet architecture (~3.2M parameters)
│   ├── 06_train.py                                   # Training loop: SGD, weighted BCE loss, best-model checkpointing
│   ├── 07_evaluate.py                                # Evaluation: accuracy, F1, precision, recall, confusion matrix
│   └── 08_predict.py                                 # Inference: classify new .fda or .npy samples
│
└── results/
    └── example_classification_report.txt             # Sample output from model evaluation
```

---

## Pipeline Overview

```
Raw .fda files
      │
      ▼
[1] Read OCT volume          oct_converter → np.ndarray (slices, H, W)
      │
      ▼
[2] Convert to NIfTI         nibabel → .nii.gz (for visualization)
      │
      ▼
[3] Crop ONH region          [40:190, 60:850, 130:400]  (identified)
      │
      ▼
[4] Normalize                ÷ 255  → [0, 1]
      │
      ▼
[5] Resize                   skimage.resize → (110, 110, 110)
      │
      ▼
[6] Save as .npy             Per-class arrays: X0.npy ... X3.npy
      │
      ▼
[7] Train 3D ResNet          SGD + Weighted BCE + ReduceLROnPlateau
      │
      ▼
[8] Evaluate                 Accuracy, F1, Precision, Recall, Confusion Matrix
```

---

## Model Architecture

Custom **3D ResNet** with ~3.2M parameters, built from scratch in PyTorch.

| Stage | Layer | Output Shape |
|---|---|---|
| 1 | Conv3d × 3 (32→32→64, stride=2) | (2, 64, 55, 55, 55) |
| 2-3 | Bottleneck × 2 (64→64) | (2, 64, 55, 55, 55) |
| 4 | Conv3d (64→64, stride=2) | (2, 64, 28, 28, 28) |
| 5-6 | Bottleneck × 2 (64→64) | (2, 64, 28, 28, 28) |
| 7 | Conv3d (64→128, stride=2) | (2, 128, 14, 14, 14) |
| 8-9 | Bottleneck × 2 (128→128) | (2, 128, 14, 14, 14) |
| 10 | MaxPool3d (7×7×7) | (2, 128, 2, 2, 2) |
| FC1 | Linear(1024→128) + ReLU | (2, 128) |
| FC2 | Linear(128→1) | (2, 1) |

**Bottleneck block:** Pre-activation design (BN → ReLU → Conv3d × 2) with identity skip connection.

### Hyperparameters

| Parameter | Value |
|---|---|
| Optimizer | SGD (momentum=0.9) |
| Learning rate | 0.001 (adaptive) |
| LR scheduler | ReduceLROnPlateau (patience=3) |
| Loss function | Weighted BCE (BCEWithLogitsLoss) |
| Epochs | 20 |
| Batch size | 2 |
| Decision threshold | 0.5 |
| Train/Test split | 80% / 20% |

---

## Installation

```bash
git clone https://github.com/your-username/Papilledema-Detection-3D-OCT.git
cd Papilledema-Detection-3D-OCT
pip install -r requirements.txt
```

**GPU support (recommended):** Install PyTorch with CUDA from [pytorch.org](https://pytorch.org/get-started/locally/).

---

## Usage

### 1. Preprocess raw `.fda` files

```bash
python -m src.preprocessing \
    --data_dir /path/to/fda_data \
    --output_dir ./data/processed
```

Expected `data_dir` structure:
```
data_dir/
    left_abnormal/    *.fda
    right_abnormal/   *.fda
    left_normal/      *.fda
    right_normal/     *.fda
```

### 2. Convert to NIfTI (for visualization)

```python
from src.utils.nifti_converter import batch_convert_to_nifti

batch_convert_to_nifti(
    input_folder="data/left_abnormal",
    output_folder="data/nifti/left_abnormal"
)
```

### 3. Train the model

```bash
python -m src.train \
    --npy_dir ./data/processed \
    --output_dir ./results \
    --epochs 20 \
    --batch_size 2 \
    --lr 0.001
```

### 4. Predict on new samples

```bash
# From a preprocessed .npy file (e.g. suspicious cases)
python -m src.predict \
    --input X4.npy \
    --checkpoint results/papilledema_classifier.pth \
    --output results/predictions.txt

# From raw .fda files directly
python -m src.predict \
    --input /path/to/fda_folder \
    --checkpoint results/papilledema_classifier.pth \
    --input_type fda \
    --output results/predictions.txt
```

**Output example:**
```
PREDICTION RESULTS
============================================================
Total samples:  45
Papilledema:    31
Normal:         14
============================================================

Sample                              Probability     Prediction
-----------------------------------------------------------------
1843.fda                                 0.8921     Papilledema
110.fda                                  0.1203          Normal
```

### 5. Evaluate

```bash
python -m src.evaluate \
    --npy_dir ./data/processed \
    --checkpoint ./results/best_model.pth \
    --output_dir ./results
```

---

## Trained Model Checkpoint

| File | Epoch | Val F1 | Notes |
|---|---|---|---|
| `papilledema_classifier.pth` | 5 | 0.6818 | Evaluated on suspicious (unlabelled) cases |

> The checkpoint was saved at the epoch with the best validation F1 score during training.
> Full held-out test performance: **Accuracy = 93%, F1 = 0.87**.

Download the checkpoint and place it in the `results/` folder before running `predict.py`.

---

## Hardware Used

| Component | Specification |
|---|---|
| CPU | 2× Intel Xeon E2620-5 (48 cores) |
| GPU | NVIDIA RTX 5000 (3072 CUDA cores, 16 GB VRAM) |
| RAM | 128 GB |

---

## Dependencies

| Library | Purpose |
|---|---|
| `oct-converter` | Reading Topcon `.fda` files |
| `nibabel` | NIfTI file I/O |
| `scikit-image` | Volume resizing |
| `PyTorch` | Model training |
| `torchmetrics` | F1, accuracy computation |
| `scikit-learn` | Train/test split, evaluation metrics |

---

## Citation

If you use this code in your research, please cite:

```bibtex
@mastersthesis{naghibzadeh2023papilledema,
  author    = {Naghibzadeh, Seyed Kianoosh},
  title     = {Evaluation of the Images of Optic Nerve Head Using Image Processing and Artificial Intelligence},
  school    = {Tarbiat Modares University},
  year      = {2023},
  address   = {Tehran, Iran},
  note      = {Supervised by Dr. Parviz Abdolmaleki}
}
```

---

## Author

**Kianoosh Naghibzadeh**
PhD Candidate, Biomedical Engineering — Toronto Metropolitan University
[LinkedIn](https://linkedin.com/in/kianooshnaghibzadeh) · kianoosh.naghibzadeh@torontomu.ca
