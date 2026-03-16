# -*- coding: utf-8 -*-
"""
Static IR Drop Prediction - Training Script
Based on "Static IR Drop Prediction with Attention U-Net and Saliency-Based Explainability"
by Lizi Zhang and Azadeh Davoodi (arXiv:2408.03292)

Trains VCAttUNet using a pretrain-finetune strategy:
  1. Pretrain on synthetic (fake) circuit data (450 epochs)
  2. Finetune on real circuit data (600 epochs)

Usage:
  python preprocess.py   # run once to convert CSVs to .npy
  python train.py        # train the model
"""

import os
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from skimage.transform import resize
import numpy as np

from DataLoad_normalization import load_real, load_fake, load_real_original_size, load_npy
from metrics import F1_Score
from model import VCAttUNet


######## Save Directory ########

DRIVE_SAVE = '/content/drive/MyDrive/ir-drop-saved'
if os.path.isdir('/content/drive/MyDrive'):
    SAVE_DIR = DRIVE_SAVE
    print(f'Saving checkpoints to Google Drive: {SAVE_DIR}')
else:
    SAVE_DIR = '../saved'
    print(f'Google Drive not mounted — saving locally: {SAVE_DIR}')

os.makedirs(f'{SAVE_DIR}/pt', exist_ok=True)
os.makedirs(f'{SAVE_DIR}/ft_real', exist_ok=True)


######## Reproducibility ########

np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device.type)


######## Data Loading ########

use_npy = os.path.isdir('../data/fake-npy') and len(os.listdir('../data/fake-npy')) > 0
if use_npy:
    print('Using preprocessed .npy data (fast)')
    dataset_fake = load_npy('../data/fake-npy')
    dataset_real = load_npy('../data/real-npy')
    dataset_test = load_npy('../data/hidden-npy')
    dataset_test_original_size = load_npy('../data/hidden-npy-orig')
else:
    print('Using raw CSV data (slow — run preprocess.py first for faster training)')
    dataset_fake = load_fake('../data/fake-circuit-data-plus/')
    dataset_real = load_real('../data/real-circuit-data-plus/', mode='train', testcase=[])
    dataset_test = load_real('../data/hidden-real-circuit-data/', mode='train', testcase=[])
    dataset_test_original_size = load_real_original_size('../data/hidden-real-circuit-data/', mode='train', testcase=[])

dataloader_fake = torch.utils.data.DataLoader(
    dataset=dataset_fake, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)

dataloader_real = torch.utils.data.DataLoader(
    dataset=dataset_real, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)

dataloader_test = torch.utils.data.DataLoader(
    dataset=dataset_test, batch_size=1, shuffle=False)

dataloader_test_original_size = torch.utils.data.DataLoader(
    dataset=dataset_test_original_size, batch_size=1, shuffle=False)


######## Loss Functions ########
# Contest scoring: 60% MAE, 30% F1, 10% runtime.
# Pretrain: asymmetric L1 (directly optimizes MAE, 2x underestimation penalty)
# Finetune: asymmetric L1 + mild hotspot weight (2x, not 10x) to help F1
#           without distorting MAE too much

class AsymmetricL1Loss(nn.Module):
    """L1 loss with 2x penalty for underestimation (pred < target)."""
    def __init__(self, lam=2.0):
        super(AsymmetricL1Loss, self).__init__()
        self.lam = lam

    def forward(self, prediction, target):
        error = torch.abs(prediction - target)
        underestimation = (prediction - target) < 0
        scaled_error = torch.where(underestimation, error * self.lam, error)
        return torch.mean(scaled_error)


class HotspotAsymmetricL1Loss(nn.Module):
    """Asymmetric L1 with mild hotspot weighting for finetuning.

    hotspot_weight=2.0 gives a gentle nudge toward F1 (30% of contest score)
    without distorting overall MAE (60% of contest score).
    """
    def __init__(self, lam=2.0, hotspot_weight=2.0):
        super(HotspotAsymmetricL1Loss, self).__init__()
        self.lam = lam
        self.hotspot_weight = hotspot_weight

    def forward(self, prediction, target):
        error = torch.abs(prediction - target)
        underestimation = (prediction - target) < 0
        error = torch.where(underestimation, error * self.lam, error)

        with torch.no_grad():
            B = target.shape[0]
            t_flat = target.reshape(B, -1)
            thresholds = 0.9 * t_flat.max(dim=1)[0]
            thresholds = thresholds.view(B, 1, 1, 1)
            hotspot_mask = (target >= thresholds).float()
            weights = 1.0 + (self.hotspot_weight - 1.0) * hotspot_mask

        return torch.mean(error * weights)


######## Hyperparameters ########

num_epochs_pt = 450
num_epochs_ft = 600
learning_rate_pt = 0.0005
learning_rate_ft = 0.0005
learning_rate_min = 0.00001
scale = 100

MSE = nn.MSELoss()
L1 = nn.L1Loss()
criterion_pt = AsymmetricL1Loss(lam=2.0)
criterion_ft = HotspotAsymmetricL1Loss(lam=2.0, hotspot_weight=2.0)


######## Helper: Evaluate on hidden test set ########

def evaluate_on_test(model):
    """Run model on hidden test set and return average L1 and F1."""
    assert len(dataloader_test) == len(dataloader_test_original_size), \
        f'Test loader mismatch: {len(dataloader_test)} vs {len(dataloader_test_original_size)}'
    model.eval()
    l1_sum = 0
    f1_sum = 0
    with torch.no_grad():
        for data, data_org in zip(dataloader_test, dataloader_test_original_size):
            maps = data[:, :-1, :, :].to(device)
            ir = data_org[:, -1, :, :].unsqueeze(1)
            output, _ = model(maps)
            output = output.cpu().detach().numpy()[0, 0] / scale
            ir_np = ir.numpy()[0, 0]
            output_resized = resize(output, ir_np.shape, preserve_range=True, anti_aliasing=True)
            output = torch.tensor(output_resized, dtype=ir.dtype).unsqueeze(0).unsqueeze(0)
            l1_sum += L1(output, ir).item()
            f1_sum += F1_Score(output.numpy().copy(), ir.numpy().copy())[0]
    n = len(dataloader_test)
    return l1_sum / n, f1_sum / n


######## Pretrain ########

print('\n' + '='*50)
print('PRETRAINING on fake circuit data')
print('='*50)

model = VCAttUNet(in_ch=12, out_ch=1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_pt)

model.train()
for epoch in range(num_epochs_pt):
    loss_sum = 0
    f_score = 0
    mse_sum = 0
    l1_sum_train = 0
    for i, data in enumerate(dataloader_fake):
        maps = data[:, :-1, :, :].to(device)
        ir = data[:, -1, :, :].unsqueeze(1).to(device) * scale
        output, _ = model(maps)

        loss = criterion_pt(output, ir)
        mse = MSE(output, ir)
        l1 = L1(output, ir)
        loss_sum += loss.item()
        mse_sum += mse.item()
        l1_sum_train += l1.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        f_score += F1_Score(output.cpu().detach().numpy().copy(), ir.cpu().numpy().copy())[0]

    if (epoch + 1) % 50 == 0 or epoch == 0:
        torch.save(model.state_dict(), f'{SAVE_DIR}/pt/{epoch}.pth')

    n_batches = len(dataloader_fake)
    print('Epoch [{}/{}], Loss: {:.4f}, F1: {:.4f}, MSE: {:.4f}, L1: {:.4f}'.format(
        epoch + 1, num_epochs_pt,
        loss_sum / n_batches, f_score / n_batches,
        mse_sum / n_batches, l1_sum_train / n_batches))

avg_l1, avg_f1 = evaluate_on_test(model)
print('****** After pretraining, L1 Loss: {:.8f}, F1 Score: {:.4f}'.format(avg_l1, avg_f1))


######## Finetune ########

print('\n' + '='*50)
print('FINETUNING on real circuit data')
print('='*50)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_ft)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs_ft, eta_min=learning_rate_min)

best_mae = float('inf')
best_f1 = -float('inf')
best_contest = -float('inf')

for epoch in range(num_epochs_ft):
    loss_sum = 0
    f_score = 0
    mse_sum = 0
    l1_sum_train = 0
    model.train()
    for i, data in enumerate(dataloader_real):
        maps = data[:, :-1, :, :].to(device)
        ir = data[:, -1, :, :].unsqueeze(1).to(device) * scale
        output, _ = model(maps)

        loss = criterion_ft(output, ir)
        mse = MSE(output, ir)
        l1 = L1(output, ir)
        loss_sum += loss.item()
        mse_sum += mse.item()
        l1_sum_train += l1.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        f_score += F1_Score(output.cpu().detach().numpy().copy(), ir.cpu().numpy().copy())[0]

    scheduler.step()

    n_batches = len(dataloader_real)
    print('Epoch [{}/{}], Loss: {:.4f}, F1: {:.4f}, MSE: {:.4f}, L1: {:.4f}'.format(
        epoch + 1, num_epochs_ft,
        loss_sum / n_batches, f_score / n_batches,
        mse_sum / n_batches, l1_sum_train / n_batches))

    if (epoch + 1) % 50 == 0 or epoch == 0:
        torch.save(model.state_dict(), f'{SAVE_DIR}/ft_real/{epoch}.pth')
        avg_l1, avg_f1 = evaluate_on_test(model)
        print('****** Epoch: {}, MAE: {:.8f}, F1: {:.4f}'.format(
            epoch + 1, avg_l1, avg_f1))

        # Save best by MAE (60% of contest score)
        if avg_l1 < best_mae:
            best_mae = avg_l1
            torch.save(model.state_dict(), f'{SAVE_DIR}/ft_real/best_mae.pth')
            print(f'  >> New best MAE: {best_mae:.8f}')

        # Save best by F1 (30% of contest score)
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            torch.save(model.state_dict(), f'{SAVE_DIR}/ft_real/best_f1.pth')
            print(f'  >> New best F1: {best_f1:.4f}')

        # Save best by approximate contest score (lower MAE = better, higher F1 = better)
        # Normalize: contest_score ~ -0.6*mae + 0.3*f1 (higher is better)
        contest_score = -0.6 * avg_l1 + 0.3 * avg_f1
        if contest_score > best_contest:
            best_contest = contest_score
            torch.save(model.state_dict(), f'{SAVE_DIR}/ft_real/best_contest.pth')
            print(f'  >> New best contest proxy (MAE={avg_l1:.8f}, F1={avg_f1:.4f})')

print(f'\nBest MAE: {best_mae:.8f}, Best F1: {best_f1:.4f}')
print(f'Checkpoints: best_mae.pth, best_f1.pth, best_contest.pth')
