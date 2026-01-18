# MRI-based Alzheimer's Disease Screening with 3D CNN

This repository provides the implementation of a lightweight 3D convolutional neural network for screening-oriented classification of Alzheimer's disease from structural MRI scans. The work focuses on high-sensitivity detection suitable for clinical screening scenarios where minimizing false negatives is prioritized.

## Overview

- Task: Binary classification (Normal vs Dementia)
- Input: 3D T1-weighted MRI volumes
- Resolution: 64 × 64 × 64
- Model: Custom 3D CNN (~450k parameters)
- Evaluation: Held-out test set (n = 162)

## Key Results

- AUC-ROC: **0.824**
- Recall (Dementia): **0.943**
- Accuracy: 0.679

High recall was achieved through class-weighted training to reduce false negatives, which is critical for medical screening applications.

## Dataset

Structural MRI scans were preprocessed and resampled to 64³ resolution. Due to data privacy and licensing restrictions, raw MRI data are not included.

The code assumes preprocessed NumPy arrays:
- `combined_scans.npy`
- `combined_labels.npy`

## Usage

### Training
```bash
python scripts/train_final.py
```

## Dataset

Structural MRI scans were preprocessed and resampled to 64³ resolution. Due to data privacy and licensing restrictions, raw MRI data are not included.

The code assumes preprocessed NumPy arrays:
- `combined_scans.npy`
- `combined_labels.npy`

## Usage

### Training
```bash
python scripts/train_final.py
```

## Paper

The accompanying paper describing the method, experiments, and clinical interpretation is provided in the paper/ directory.

arXiv preprint: (link will be added after submission)

## Author
Georgii A. Erokhin

Contact:
georgii.erokhin@gmail.com
george.erokhin@proton.me

## Disclaimer
This project is intended for research and educational purposes only. It is not a certified medical diagnostic tool.
