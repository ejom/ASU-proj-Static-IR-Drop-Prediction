# -*- coding: utf-8 -*-
"""
Static IR Drop Prediction - Training Script
Based on: "Static IR Drop Prediction with Attention U-Net and
Saliency-Based Explainability" by Zhang & Davoodi (arXiv:2408.03292)

Two-phase training:
  1. Pretrain on 100 synthetic circuits (450 epochs, high dropout, asymmetric L1)
  2. Finetune on real circuits (600 epochs, low dropout, hotspot-weighted L1)

Uses 8/2 train/val split on real data for checkpoint selection.

Usage:
  python preprocess.py   # run once to convert CSVs to .npy
  python train.py        # train the model
  python evaluate.py --model <checkpoint>  # final eval on hidden test set
"""

import os
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import Subset
from torch.amp import autocast, GradScaler
import numpy as np

from DataLoad_normalization import load_real, load_fake, load_npy
from metrics import F1_Score
from model import VCAttUNet, set_dropout_rate


# =====================================================================
# Save Directory
# =====================================================================

DRIVE_SAVE = '/content/drive/MyDrive/ir-drop-saved'
if os.path.isdir('/content/drive/MyDrive'):
    SAVE_DIR = DRIVE_SAVE
    print(f'Saving checkpoints to Google Drive: {SAVE_DIR}')
else:
    SAVE_DIR = '../saved'
    print(f'Google Drive not mounted — saving locally: {SAVE_DIR}')

os.makedirs(f'{SAVE_DIR}/pt', exist_ok=True)
os.makedirs(f'{SAVE_DIR}/ft_real', exist_ok=True)


# =====================================================================
# Reproducibility & Device
# =====================================================================

np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device.type)
use_amp = device.type == 'cuda'
pin = device.type == 'cuda'
num_workers = 2


# =====================================================================
# Data Loading
# =====================================================================

use_npy = os.path.isdir('../data/fake-npy') and len(os.listdir('../data/fake-npy')) > 0
if use_npy:
    print('Using preprocessed .npy data (fast)')
    dataset_fake = load_npy('../data/fake-npy')
    dataset_real = load_npy('../data/real-npy')
else:
    print('Using raw CSV data (slow — run preprocess.py first for faster training)')
    dataset_fake = load_fake('../data/fake-circuit-data-plus/')
    dataset_real = load_real('../data/real-circuit-data-plus/', mode='train', testcase=[])

dataloader_fake = torch.utils.data.DataLoader(
    dataset=dataset_fake, batch_size=8, shuffle=True,
    num_workers=num_workers, pin_memory=pin, persistent_workers=(num_workers > 0))

# 8/2 train/val split on real data for checkpoint selection
n_real = len(dataset_real)
n_val = max(2, n_real // 5)
n_train = n_real - n_val
g = torch.Generator().manual_seed(0)
perm = torch.randperm(n_real, generator=g).tolist()
dataset_real_train = Subset(dataset_real, perm[:n_train])
dataset_real_val = Subset(dataset_real, perm[n_train:])
dataloader_real = torch.utils.data.DataLoader(
    dataset=dataset_real_train, batch_size=8, shuffle=True,
    num_workers=num_workers, pin_memory=pin, persistent_workers=(num_workers > 0))
dataloader_real_val = torch.utils.data.DataLoader(
    dataset=dataset_real_val, batch_size=1, shuffle=False,
    num_workers=num_workers, pin_memory=pin)
print(f'Real data: {n_train} train, {n_val} val')


# =====================================================================
# Loss Functions
#
# Pretrain: Asymmetric L1 (paper Section 2.3)
#   - L1 base preserves dynamic range (MSE compresses it)
#   - lambda=2 penalty for underestimation
#
# Finetune: Asymmetric L1 + hotspot weighting
#   - Same L1 base as pretrain (consistent loss landscape)
#   - Additional weight on hotspot pixels (top 10% by IR drop)
#   - Directly aligns with F1 evaluation metric
# =====================================================================

class AsymmetricL1Loss(nn.Module):
    """Asymmetric L1: penalizes underestimation by lambda=2.
    Paper eq: Loss = mean(|pred-gt|) where underestimation gets 2x weight."""
    def __init__(self, lam=2.0):
        super().__init__()
        self.lam = lam

    def forward(self, pred, target):
        error = torch.abs(pred - target)
        underest = pred < target
        error = torch.where(underest, error * self.lam, error)
        return torch.mean(error)


class HotspotAsymmetricL1Loss(nn.Module):
    """Asymmetric L1 with extra weight on hotspot pixels.
    Hotspot = pixel with IR drop > 90% of sample max (contest definition).
    Uses L1 base (not MSE) to preserve output dynamic range."""
    def __init__(self, lam=2.0, hotspot_weight=10.0):
        super().__init__()
        self.lam = lam
        self.hotspot_weight = hotspot_weight

    def forward(self, pred, target):
        # Asymmetric L1: 2x penalty for underestimation
        error = torch.abs(pred - target)
        underest = pred < target
        error = torch.where(underest, error * self.lam, error)

        # Hotspot weighting: upweight top 10% pixels per sample
        with torch.no_grad():
            B = target.shape[0]
            t_flat = target.reshape(B, -1)
            thresholds = 0.9 * t_flat.max(dim=1)[0]
            thresholds = thresholds.view(B, 1, 1, 1)
            hotspot_mask = (target >= thresholds).float()
            weights = 1.0 + (self.hotspot_weight - 1.0) * hotspot_mask

        return torch.mean(error * weights)


def augment_batch(maps, ir):
    """Random flips and 90 degree rotations for data augmentation."""
    if torch.rand(1).item() > 0.5:
        maps = torch.flip(maps, [-1])
        ir = torch.flip(ir, [-1])
    if torch.rand(1).item() > 0.5:
        maps = torch.flip(maps, [-2])
        ir = torch.flip(ir, [-2])
    k = torch.randint(0, 4, (1,)).item()
    if k > 0:
        maps = torch.rot90(maps, k, [-2, -1])
        ir = torch.rot90(ir, k, [-2, -1])
    return maps, ir


# =====================================================================
# Hyperparameters (paper Table 1)
# =====================================================================

num_epochs_pt = 450
num_epochs_ft = 600
learning_rate = 0.0005      # paper: 0.0005 for both phases
learning_rate_min = 0.00001
scale = 100

MSE = nn.MSELoss()
L1 = nn.L1Loss()
criterion_pt = AsymmetricL1Loss(lam=2.0)           # pretrain: paper-consistent
criterion_ft = HotspotAsymmetricL1Loss(lam=2.0, hotspot_weight=10.0)  # finetune: + hotspot focus
scaler = GradScaler('cuda', enabled=use_amp)


# =====================================================================
# Validation helper
# =====================================================================

def evaluate_on_val(model):
    """Evaluate on val split. Returns (avg_l1, avg_f1)."""
    model.eval()
    l1_sum = 0
    f1_sum = 0
    n = 0
    with torch.no_grad():
        for data in dataloader_real_val:
            maps = data[:, :-1, :, :].to(device, non_blocking=True)
            ir = data[:, -1:, :, :].to(device, non_blocking=True) * scale
            with autocast(device_type=device.type, enabled=use_amp):
                output, _ = model(maps)
            output = output.float()
            l1_sum += L1(output, ir).item()
            f1_sum += F1_Score(
                output.cpu().numpy().copy(),
                ir.cpu().numpy().copy()
            )[0]
            n += 1
    if n == 0:
        return 0.0, 0.0
    return l1_sum / n, f1_sum / n


# =====================================================================
# Phase 1: PRETRAINING on fake circuit data
# Paper: 450 epochs, LR=0.0005, dropout 0.3-0.5, asymmetric L1
# =====================================================================

print('\n' + '='*50)
print('PRETRAINING on fake circuit data')
print('='*50)

model = VCAttUNet(in_ch=12, out_ch=1, dropout_rate=0.5).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

model.train()
for epoch in range(num_epochs_pt):
    loss_sum = 0
    mse_sum = 0
    l1_sum = 0

    for i, data in enumerate(dataloader_fake):
        maps = data[:, :-1, :, :].to(device, non_blocking=True)
        ir = data[:, -1, :, :].unsqueeze(1).to(device, non_blocking=True) * scale
        maps, ir = augment_batch(maps, ir)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=use_amp):
            output, _ = model(maps)
            loss = criterion_pt(output, ir)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            out_fp32 = output.detach().float()
            mse_sum += MSE(out_fp32, ir).item()
            l1_sum += L1(out_fp32, ir).item()
        loss_sum += loss.item()

    if (epoch + 1) % 50 == 0 or epoch == 0:
        torch.save(model.state_dict(), f'{SAVE_DIR}/pt/{epoch}.pth')

    n_batches = len(dataloader_fake)
    print('Epoch [{}/{}], Loss: {:.4f}, MSE: {:.4f}, L1: {:.4f}'.format(
        epoch + 1, num_epochs_pt,
        loss_sum / n_batches, mse_sum / n_batches, l1_sum / n_batches))

# Check pretrain quality on real val data
avg_l1, avg_f1 = evaluate_on_val(model)
print('****** Pretraining complete. Val L1: {:.8f}, Val F1: {:.4f}'.format(avg_l1, avg_f1))


# =====================================================================
# Phase 2: FINETUNING on real circuit data
# Paper: 600 epochs, LR=0.0005->0.00001, dropout 0.1
# =====================================================================

print('\n' + '='*50)
print('FINETUNING on real circuit data')
print('='*50)

# CRITICAL: reduce dropout from 0.5 to 0.1 for finetuning (paper Table 1)
set_dropout_rate(model, 0.1)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=2, eta_min=learning_rate_min)

best_val_f1 = -float('inf')
best_val_l1 = float('inf')
best_val_combo = -float('inf')

for epoch in range(num_epochs_ft):
    loss_sum = 0
    mse_sum = 0
    l1_sum = 0
    model.train()

    for i, data in enumerate(dataloader_real):
        maps = data[:, :-1, :, :].to(device, non_blocking=True)
        ir = data[:, -1, :, :].unsqueeze(1).to(device, non_blocking=True) * scale
        maps, ir = augment_batch(maps, ir)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=use_amp):
            output, _ = model(maps)
            loss = criterion_ft(output, ir)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            out_fp32 = output.detach().float()
            mse_sum += MSE(out_fp32, ir).item()
            l1_sum += L1(out_fp32, ir).item()
        loss_sum += loss.item()

    scheduler.step(epoch + 1)

    n_batches = len(dataloader_real)
    print('Epoch [{}/{}], Loss: {:.4f}, MSE: {:.4f}, L1: {:.4f}'.format(
        epoch + 1, num_epochs_ft,
        loss_sum / n_batches, mse_sum / n_batches, l1_sum / n_batches))

    # Checkpoint + val every 25 epochs (and epoch 1)
    do_save = epoch == 0 or (epoch + 1) % 25 == 0

    if do_save:
        torch.save(model.state_dict(), f'{SAVE_DIR}/ft_real/{epoch}.pth')

        avg_l1, avg_f1 = evaluate_on_val(model)
        print('****** Epoch {}, Val L1: {:.8f}, Val F1: {:.4f}'.format(
            epoch + 1, avg_l1, avg_f1))
        if avg_f1 > best_val_f1:
            best_val_f1 = avg_f1
            torch.save(model.state_dict(), f'{SAVE_DIR}/ft_real/best_f1.pth')
            print(f'  >> New best val F1 (F1={best_val_f1:.4f})')
        if avg_l1 < best_val_l1:
            best_val_l1 = avg_l1
            torch.save(model.state_dict(), f'{SAVE_DIR}/ft_real/best_l1.pth')
            print(f'  >> New best val L1 (L1={best_val_l1:.8f})')
        combo = avg_f1 - 2.0 * avg_l1
        if combo > best_val_combo:
            best_val_combo = combo
            torch.save(model.state_dict(), f'{SAVE_DIR}/ft_real/best_combo.pth')
            print(f'  >> New best combo (score={best_val_combo:.4f})')

# Save final model
torch.save(model.state_dict(), f'{SAVE_DIR}/ft_real/final.pth')
print('****** Finetuning complete. Final model saved.')
print(f'****** Best val F1={best_val_f1:.4f}, Best val L1={best_val_l1:.8f}')
print('****** Evaluate with: python evaluate.py --model .../ft_real/best_f1.pth')
