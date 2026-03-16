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


######## Loss Function ########

class CustomMSELoss(nn.Module):
    """MSE loss with heavier penalty for underestimation (pred < target)."""
    def __init__(self, negative_scale=2.0):
        super(CustomMSELoss, self).__init__()
        self.negative_scale = negative_scale

    def forward(self, prediction, target):
        squared_error = (prediction - target) ** 2
        underestimation = (prediction - target) < 0
        scaled_error = torch.where(underestimation, squared_error * self.negative_scale, squared_error)
        return torch.mean(scaled_error)


######## Hyperparameters (matched to paper) ########

num_epochs_pt = 450
num_epochs_ft = 600
learning_rate_pt = 0.0005
learning_rate_ft = 0.0005
learning_rate_min = 0.00001
scale = 100

MSE = nn.MSELoss()
L1 = nn.L1Loss()
criterion = CustomMSELoss()


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
            output_resized = resize(output, ir_np.shape, preserve_range=True)
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

        loss = criterion(output, ir)
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

        loss = criterion(output, ir)
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
        print('****** After Finetuning Epoch: {}, L1 Loss: {:.8f}, F1 Score: {:.4f}'.format(
            epoch + 1, avg_l1, avg_f1))
