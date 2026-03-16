# -*- coding: utf-8 -*-
"""
Static IR Drop Prediction - Training Script
Based on "Static IR Drop Prediction with Attention U-Net and Saliency-Based Explainability"
by Lizi Zhang and Azadeh Davoodi (arXiv:2408.03292)

Trains VCAttUNet using a pretrain-finetune strategy:
  1. Pretrain on synthetic (fake) circuit data (450 epochs)
  2. Finetune on real circuit data (600 epochs)

The model takes 12 input feature maps (current, distance, PDN density,
resistances for 5 metal layers, vias for 4 layer pairs) and predicts
a single IR drop heatmap.

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
from torch.amp import autocast, GradScaler  # mixed precision training (float16 on GPU)
import numpy as np

from DataLoad_normalization import load_real, load_fake, load_npy
from metrics import F1_Score
from model import VCAttUNet


# =====================================================================
# Save Directory — use Google Drive on Colab so checkpoints survive
# runtime disconnects, otherwise save locally
# =====================================================================

DRIVE_SAVE = '/content/drive/MyDrive/ir-drop-saved'
if os.path.isdir('/content/drive/MyDrive'):
    SAVE_DIR = DRIVE_SAVE
    print(f'Saving checkpoints to Google Drive: {SAVE_DIR}')
else:
    SAVE_DIR = '../saved'
    print(f'Google Drive not mounted — saving locally: {SAVE_DIR}')

# pt/ = pretrain checkpoints, ft_real/ = finetune checkpoints
os.makedirs(f'{SAVE_DIR}/pt', exist_ok=True)
os.makedirs(f'{SAVE_DIR}/ft_real', exist_ok=True)


# =====================================================================
# Reproducibility — seed all random number generators so results are
# repeatable across runs. benchmark=True lets cuDNN auto-tune convolution
# algorithms for our fixed 512x512 input size (faster on GPU).
# =====================================================================

np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = False  # allow non-deterministic ops for speed
torch.backends.cudnn.benchmark = True       # auto-tune conv algorithms for fixed input size

# Use GPU if available, otherwise CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device.type)
use_amp = device.type == 'cuda'  # automatic mixed precision only on GPU
pin = device.type == 'cuda'      # pin_memory speeds up CPU→GPU data transfer


# =====================================================================
# Data Loading
#
# Each sample is a tensor of shape (13, 512, 512):
#   - Channels 0-11: 12 input feature maps (current, distance, PDN, etc.)
#   - Channel 12:    ground truth IR drop map (the target we predict)
#
# We prefer preprocessed .npy files (fast) over raw CSVs (slow).
# =====================================================================

use_npy = os.path.isdir('../data/fake-npy') and len(os.listdir('../data/fake-npy')) > 0
if use_npy:
    print('Using preprocessed .npy data (fast)')
    dataset_fake = load_npy('../data/fake-npy')   # 100 synthetic circuit samples
    dataset_real = load_npy('../data/real-npy')    # 10 real circuit samples
else:
    print('Using raw CSV data (slow — run preprocess.py first for faster training)')
    dataset_fake = load_fake('../data/fake-circuit-data-plus/')
    dataset_real = load_real('../data/real-circuit-data-plus/', mode='train', testcase=[])

# --- Train/val split from real data (80% train, 20% val) ---
# We use a seeded random permutation so the split is random but reproducible.
# The validation set is used for model selection (best checkpoint).
# The hidden test set is NEVER seen during training — only used in evaluate.py.
n_real = len(dataset_real)
n_val = max(2, n_real // 5)      # at least 2 val samples
n_train = n_real - n_val          # remaining for training
g = torch.Generator().manual_seed(0)
perm = torch.randperm(n_real, generator=g).tolist()  # random shuffle with fixed seed
train_indices = perm[:n_train]
val_indices = perm[n_train:]
dataset_real_train = Subset(dataset_real, train_indices)
dataset_real_val = Subset(dataset_real, val_indices)
print(f'Real data split: {n_train} train, {n_val} val')

# --- DataLoaders wrap datasets and handle batching, shuffling, parallel loading ---
# batch_size=8: process 8 samples at once
# shuffle=True: randomize order each epoch (important for training)
# num_workers=2: load data in 2 parallel background processes
# pin_memory=True: pre-allocate GPU-accessible memory for faster transfers
# persistent_workers=True: keep worker processes alive between epochs (avoids restart overhead)
dataloader_fake = torch.utils.data.DataLoader(
    dataset=dataset_fake, batch_size=8, shuffle=True,
    num_workers=2, pin_memory=pin, persistent_workers=True)

dataloader_real_train = torch.utils.data.DataLoader(
    dataset=dataset_real_train, batch_size=8, shuffle=True,
    num_workers=2, pin_memory=pin, persistent_workers=True)

# Val loader: batch_size=1 for per-sample evaluation, no shuffle needed
dataloader_real_val = torch.utils.data.DataLoader(
    dataset=dataset_real_val, batch_size=1, shuffle=False,
    num_workers=2, pin_memory=pin)


# =====================================================================
# Loss Function
#
# The contest evaluates F1 score on "hotspot" pixels — pixels where
# IR drop exceeds 90% of that sample's maximum. Standard MSE treats
# all pixels equally, so the model ignores hotspot locations (only ~10%
# of pixels). This custom loss upweights those hotspot pixels by 10x,
# directly aligning training with the F1 evaluation metric.
# =====================================================================

class HotspotWeightedLoss(nn.Module):
    """MSE loss that upweights pixels whose target IR drop exceeds
    90% of that sample's maximum IR drop, matching the contest hotspot definition."""
    def __init__(self, hotspot_weight=10.0, underestimate_scale=2.0):
        super().__init__()
        self.hotspot_weight = hotspot_weight        # extra weight for hotspot pixels
        self.underestimate_scale = underestimate_scale  # extra penalty when pred < target

    def forward(self, pred, target):
        # Squared error for each pixel
        error = (pred - target) ** 2

        # Penalize underestimation (pred < target) more heavily, because
        # missing a hotspot is worse than slightly over-predicting
        underest = pred < target
        error = torch.where(underest, error * self.underestimate_scale, error)

        # Identify hotspot pixels per sample (top ~10% by IR drop value)
        # This runs without gradients since it's just computing a mask
        with torch.no_grad():
            B = target.shape[0]  # batch size
            t_flat = target.reshape(B, -1)                      # flatten spatial dims
            thresholds = 0.9 * t_flat.max(dim=1)[0]             # 90% of max per sample
            thresholds = thresholds.view(B, 1, 1, 1)            # reshape for broadcasting
            hotspot_mask = (target >= thresholds).float()        # 1.0 for hotspot, 0.0 otherwise
            # Non-hotspot pixels get weight 1.0, hotspot pixels get weight hotspot_weight
            weights = 1.0 + (self.hotspot_weight - 1.0) * hotspot_mask

        return torch.mean(error * weights)


def augment_batch(maps, ir):
    """Apply random geometric transforms to a batch for data augmentation.

    Both input maps and target IR drop are transformed identically so they
    stay aligned. With 2 flip axes and 4 rotation angles, this gives up to
    8x effective dataset size from the same samples.

    Args:
        maps: input feature maps tensor (B, 12, H, W)
        ir:   target IR drop tensor (B, 1, H, W)
    Returns:
        transformed (maps, ir) tuple
    """
    # 50% chance horizontal flip
    if torch.rand(1).item() > 0.5:
        maps = torch.flip(maps, [-1])  # flip last dim (width)
        ir = torch.flip(ir, [-1])
    # 50% chance vertical flip
    if torch.rand(1).item() > 0.5:
        maps = torch.flip(maps, [-2])  # flip second-to-last dim (height)
        ir = torch.flip(ir, [-2])
    # Random 0/90/180/270 degree rotation
    k = torch.randint(0, 4, (1,)).item()  # k=0 means no rotation
    if k > 0:
        maps = torch.rot90(maps, k, [-2, -1])  # rotate in the H,W plane
        ir = torch.rot90(ir, k, [-2, -1])
    return maps, ir


# =====================================================================
# Hyperparameters
# =====================================================================

num_epochs_pt = 450        # pretrain epochs on fake data
num_epochs_ft = 600        # finetune epochs on real data
learning_rate_pt = 0.0005  # pretrain learning rate
learning_rate_ft = 0.0002  # finetune learning rate (lower to preserve pretrained features)
learning_rate_min = 0.00001  # minimum LR for cosine annealing
scale = 100                # multiply IR drop targets by this so values aren't tiny
                           # (helps numerical stability; divided out at evaluation time)

MSE = nn.MSELoss()         # standard mean squared error (for logging only)
L1 = nn.L1Loss()           # mean absolute error (for logging only)
criterion = HotspotWeightedLoss()  # actual training loss

# GradScaler scales the loss up before backward() to prevent float16 gradients
# from underflowing to zero, then scales them back down before optimizer.step()
scaler = GradScaler('cuda', enabled=use_amp)


# =====================================================================
# Helper: Evaluate on validation set
#
# Runs the model on the held-out validation samples (2 real circuits)
# and computes average L1 error and F1 score. This is used for:
#   - Monitoring training progress
#   - Selecting the best checkpoint (best_f1.pth, best_l1.pth)
# =====================================================================

def evaluate_on_val(model):
    """Run model on validation split and return average L1 and F1 at 512x512."""
    model.eval()  # switch to evaluation mode (changes BatchNorm and Dropout behavior)
    l1_sum = 0
    f1_sum = 0
    n = 0
    with torch.no_grad():  # disable gradient computation (saves memory and time)
        for data in dataloader_real_val:
            # Split tensor: first 12 channels = input features, last channel = IR drop target
            maps = data[:, :-1, :, :].to(device)
            ir = data[:, -1:, :, :].to(device) * scale  # scale target to match training range
            with autocast(device_type=device.type, enabled=use_amp):
                output, _ = model(maps)
            output = output.float()  # ensure float32 for accurate metrics
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
# Phase 1: PRETRAINING on synthetic (fake) circuit data
#
# The fake data is cheaper to generate and provides 100 samples.
# Pretraining teaches the model general circuit patterns before we
# finetune on the scarce real data (only 8 training samples).
# =====================================================================

print('\n' + '='*50)
print('PRETRAINING on fake circuit data')
print('='*50)

# Create model: 12 input channels → 1 output channel (IR drop prediction)
model = VCAttUNet(in_ch=12, out_ch=1).to(device)

# Adam optimizer with small weight decay (L2 regularization) to prevent overfitting
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_pt, weight_decay=1e-5)

model.train()  # set model to training mode (enables BatchNorm updates, dropout, etc.)
for epoch in range(num_epochs_pt):
    loss_sum = 0
    mse_sum = 0
    l1_sum_train = 0

    for i, data in enumerate(dataloader_fake):
        # Split the 13-channel tensor into input (12 ch) and target (1 ch)
        # non_blocking=True allows CPU→GPU transfer to overlap with computation
        maps = data[:, :-1, :, :].to(device, non_blocking=True)          # (B, 12, 512, 512)
        ir = data[:, -1, :, :].unsqueeze(1).to(device, non_blocking=True) * scale  # (B, 1, 512, 512)

        # Apply random flips/rotations to both input and target
        maps, ir = augment_batch(maps, ir)

        # --- Forward pass with mixed precision ---
        # set_to_none=True is slightly faster than zeroing gradients
        optimizer.zero_grad(set_to_none=True)
        # autocast automatically uses float16 where safe (convolutions, matmuls)
        # and float32 where needed (reductions, loss) for ~2x GPU speedup
        with autocast(device_type=device.type, enabled=use_amp):
            output, _ = model(maps)      # model prediction
            loss = criterion(output, ir)  # hotspot-weighted loss

        # --- Backward pass with gradient scaling ---
        scaler.scale(loss).backward()              # compute gradients (scaled up for fp16 safety)
        scaler.unscale_(optimizer)                  # scale gradients back to real values
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # prevent exploding gradients
        scaler.step(optimizer)                      # update model weights
        scaler.update()                             # adjust scale factor for next iteration

        # --- Log metrics (no gradients needed) ---
        with torch.no_grad():
            out_fp32 = output.detach().float()
            mse_sum += MSE(out_fp32, ir).item()        # .item() extracts Python number from tensor
            l1_sum_train += L1(out_fp32, ir).item()
        loss_sum += loss.item()

    # Save checkpoint every 50 epochs (and the first epoch)
    if (epoch + 1) % 50 == 0 or epoch == 0:
        torch.save(model.state_dict(), f'{SAVE_DIR}/pt/{epoch}.pth')

    # Print epoch summary (averaged over all batches)
    n_batches = len(dataloader_fake)
    print('Epoch [{}/{}], Loss: {:.4f}, MSE: {:.4f}, L1: {:.4f}'.format(
        epoch + 1, num_epochs_pt,
        loss_sum / n_batches,
        mse_sum / n_batches, l1_sum_train / n_batches))

# Check how well the pretrained model does on real validation data
avg_l1, avg_f1 = evaluate_on_val(model)
print('****** After pretraining, Val L1: {:.8f}, Val F1: {:.4f}'.format(avg_l1, avg_f1))


# =====================================================================
# Phase 2: FINETUNING on real circuit data
#
# Starting from the pretrained weights, we now train on real circuit
# data with a lower learning rate to adapt without losing pretrained
# knowledge. Only 8 real training samples, so augmentation is critical.
#
# Cosine annealing with warm restarts periodically resets the learning
# rate to escape local minima. T_0=50 means the first cycle is 50
# epochs, T_mult=2 doubles each subsequent cycle (50, 100, 200, ...).
# =====================================================================

print('\n' + '='*50)
print('FINETUNING on real circuit data')
print('='*50)

# Fresh optimizer for finetuning (new LR, resets momentum)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_ft, weight_decay=1e-5)

# Cosine annealing with warm restarts: LR oscillates between learning_rate_ft
# and learning_rate_min in cycles of increasing length
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=learning_rate_min)

# Track best validation metrics for checkpoint selection
best_val_f1 = -float('inf')  # any F1 beats this
best_val_l1 = float('inf')   # any L1 beats this

for epoch in range(num_epochs_ft):
    loss_sum = 0
    mse_sum = 0
    l1_sum_train = 0
    model.train()  # re-enable training mode (evaluate_on_val sets it to eval)

    for i, data in enumerate(dataloader_real_train):
        # Same data splitting as pretrain: 12 input channels + 1 target channel
        maps = data[:, :-1, :, :].to(device, non_blocking=True)
        ir = data[:, -1, :, :].unsqueeze(1).to(device, non_blocking=True) * scale
        maps, ir = augment_batch(maps, ir)

        # Forward + backward pass (identical structure to pretrain)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=use_amp):
            output, _ = model(maps)
            loss = criterion(output, ir)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            out_fp32 = output.detach().float()
            mse_sum += MSE(out_fp32, ir).item()
            l1_sum_train += L1(out_fp32, ir).item()
        loss_sum += loss.item()

    # Step the LR scheduler (adjusts learning rate for next epoch)
    scheduler.step(epoch + 1)

    # Print epoch summary
    n_batches = len(dataloader_real_train)
    print('Epoch [{}/{}], Loss: {:.4f}, MSE: {:.4f}, L1: {:.4f}'.format(
        epoch + 1, num_epochs_ft,
        loss_sum / n_batches,
        mse_sum / n_batches, l1_sum_train / n_batches))

    # Every 25 epochs (and epoch 1): save checkpoint + run validation
    if (epoch + 1) % 25 == 0 or epoch == 0:
        torch.save(model.state_dict(), f'{SAVE_DIR}/ft_real/{epoch}.pth')
        avg_l1, avg_f1 = evaluate_on_val(model)
        print('****** After Finetuning Epoch: {}, Val L1: {:.8f}, Val F1: {:.4f}'.format(
            epoch + 1, avg_l1, avg_f1))

        # Save best-F1 checkpoint (optimizes hotspot detection accuracy)
        if avg_f1 > best_val_f1:
            best_val_f1 = avg_f1
            torch.save(model.state_dict(), f'{SAVE_DIR}/ft_real/best_f1.pth')
            print(f'  >> New best val F1 model saved (F1={best_val_f1:.4f})')

        # Save best-L1 checkpoint (optimizes overall prediction accuracy)
        if avg_l1 < best_val_l1:
            best_val_l1 = avg_l1
            torch.save(model.state_dict(), f'{SAVE_DIR}/ft_real/best_l1.pth')
            print(f'  >> New best val L1 model saved (L1={best_val_l1:.8f})')
