# Static IR Drop Prediction with Attention U-Net

Based on "[Static IR Drop Prediction with Attention U-Net and Saliency-Based Explainability](https://www.arxiv.org/abs/2408.03292)" by Lizi Zhang and Azadeh Davoodi. Original repository: [lzzh97/Static-IR-Drop-Prediction](https://github.com/lzzh97/Static-IR-Drop-Prediction).

This fork includes bug fixes, preprocessing optimizations, and Google Colab compatibility for the ASU Semiconductor Solutions Challenge 2026 (Problem D).

## Model

**VCAttUNet** — a U-Net architecture with embedded attention gates for predicting static IR drop in power delivery networks (PDNs).

- **Input**: 12-channel feature maps (current density, effective distance, PDN density, per-layer resistance for metal layers m1/m4/m7/m8/m9, and inter-layer via density for m1-m4/m4-m7/m7-m8/m8-m9)
- **Output**: Predicted IR drop heatmap
- **Strategy**: Pretrain on 100 synthetic circuits, finetune on 10 real circuits
- **Metrics**: F1 score (hotspot detection) and MAE (regression accuracy)

## Project Structure

```
AttUnet/
  model.py                  # VCAttUNet and VCAttUNet_Large architectures
  train.py                  # Training script (pretrain + finetune)
  evaluate.py               # Standalone evaluation script
  preprocess.py             # Convert CSV data to fast .npy format
  extract_features.py       # Extract resistance/via features from SPICE netlists
  DataLoad_normalization.py # Dataset classes for CSV and .npy loading
  metrics.py                # F1 score and MAE computation
  utilities.py              # Helper functions for netlist parsing
data/
  fake-circuit-data-plus/   # 100 synthetic circuit samples (CSV)
  real-circuit-data-plus/   # 10 real circuit samples for training (CSV)
  hidden-real-circuit-data/ # 10 real circuit samples for testing (CSV)
```

## Quick Start (Google Colab)

### 1. Setup
```python
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/ejom/ASU-proj-Static-IR-Drop-Prediction.git
!pip install scikit-image
```

### 2. Extract missing features (first time only)
```python
%cd /content/ASU-proj-Static-IR-Drop-Prediction/AttUnet
!python extract_features.py
```

### 3. Preprocess data (first time only)
```python
!python preprocess.py
```

### 4. Train
```python
!python train.py
```

### 5. Evaluate
```python
!python evaluate.py --model /content/drive/MyDrive/ir-drop-saved/ft_real/599.pth
```

## Training Details

| Phase | Epochs | Learning Rate | Scheduler |
|-------|--------|--------------|-----------|
| Pretrain | 450 | 0.0005 | Constant |
| Finetune | 600 | 0.0005 → 0.00001 | CosineAnnealingLR |

- **Loss**: Asymmetric MSE (2x penalty for underestimation)
- **Optimizer**: Adam
- **Batch size**: 8
- **Input resolution**: 512x512

## Data

Originally provided by the [ICCAD 2023 Contest Problem C](https://github.com/ASU-VDA-Lab/ML-for-IR-drop). Each circuit sample includes:

| Feature | Description |
|---------|-------------|
| current_map | Current density per grid cell |
| eff_dist_map | Effective distance to nearest power pin |
| pdn_density | Power delivery network wire density |
| resistance_m{1,4,7,8,9} | Per-layer metal resistance |
| via_m{1m4,4m7,7m8,8m9} | Inter-layer via resistance |
| ir_drop_map | Ground truth IR drop (target) |

## Requirements

```
torch>=2.0
numpy
scikit-image
scikit-learn
matplotlib
seaborn
scipy
```
