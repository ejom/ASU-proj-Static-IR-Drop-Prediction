# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 22:11:33 2024

@author: Lizi Zhang
"""
# import wandb

import os
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from skimage.transform import resize

import numpy as np

from DataLoad_normalization import load_real, load_fake, load_real_original_size, load_npy
from metrics import F1_Score

import seaborn as sns
from matplotlib import pyplot as plt

# Save checkpoints to Google Drive if available, otherwise local
DRIVE_SAVE = '/content/drive/MyDrive/ir-drop-saved'
if os.path.isdir('/content/drive/MyDrive'):
    SAVE_DIR = DRIVE_SAVE
    print(f'Saving checkpoints to Google Drive: {SAVE_DIR}')
else:
    SAVE_DIR = '../saved'
    print(f'Google Drive not mounted — saving locally: {SAVE_DIR}')

os.makedirs(f'{SAVE_DIR}/pt', exist_ok=True)
os.makedirs(f'{SAVE_DIR}/ft_real', exist_ok=True)

torch.cuda.empty_cache()
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device.type)


######## dataloader ########

# Use preprocessed .npy data if available, otherwise fall back to CSV
use_npy = os.path.isdir('../data/fake-npy') and len(os.listdir('../data/fake-npy')) > 0
if use_npy:
    print('Using preprocessed .npy data (fast)')
    dataset_fake = load_npy('../data/fake-npy')
    dataset_real = load_npy('../data/real-npy')
    dataset_test = load_npy('../data/hidden-npy')
    dataset_test_original_size = load_npy('../data/hidden-npy-orig')
else:
    print('Using raw CSV data (slow — run preprocess.py first for faster training)')
    datapath_fake='../data/fake-circuit-data-plus/'
    dataset_fake = load_fake(datapath_fake)

    datapath_real='../data/real-circuit-data-plus/'
    dataset_real = load_real(datapath_real, mode='train', testcase=[])

    datapath_test='../data/hidden-real-circuit-data/'
    dataset_test = load_real(datapath_test, mode='train', testcase=[])
    dataset_test_original_size = load_real_original_size(datapath_test, mode='train', testcase=[])

dataloader_fake = torch.utils.data.DataLoader(dataset = dataset_fake,
                                        batch_size = 8,
                                        shuffle = True,
                                        num_workers = 2,
                                        pin_memory = True)

dataloader_real = torch.utils.data.DataLoader(dataset = dataset_real,
                                        batch_size = 8,
                                        shuffle = True,
                                        num_workers = 2,
                                        pin_memory = True)

dataloader_test = torch.utils.data.DataLoader(dataset = dataset_test,
                                        batch_size = 1,
                                        shuffle = False)

dataloader_test_original_size = torch.utils.data.DataLoader(dataset = dataset_test_original_size,
                                        batch_size = 1,
                                        shuffle = False)


######## CustomMSELoss ########


class CustomMSELoss(nn.Module):
    def __init__(self, negative_scale=2.0):
        super(CustomMSELoss, self).__init__()
        self.negative_scale = negative_scale

    def forward(self, prediction, target):
        # Calculate the squared error per element
        squared_error = (prediction - target) ** 2

        # Penalize underestimation (pred < target) more heavily
        underestimation = (prediction - target) < 0
        squared_error[underestimation] *= self.negative_scale

        return torch.mean(squared_error)


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

# wandb.init(
#     project='IR_pre',
#     name= 'VCAttUNet',
#     config={
#         'pt_lr':learning_rate_pt,
#         'ft_lr':learning_rate_ft,
#         'pt_epoch':num_epochs_pt,
#         'ft_epoch':num_epochs_ft})


######## Pretrain ########

from model import VCAttUNet as net

model = net(in_ch=12, out_ch=1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_pt)



model.train()
for epoch in range(num_epochs_pt):
    loss_sum = 0
    f_score = 0
    for i, data in enumerate(dataloader_fake):
        maps = data[:,:-1,:,:]
        maps = maps.to(device)
        ir = data[:,-1,:,:].unsqueeze(1).to(device)*scale
        output,_ = model(maps)
        
        loss = criterion(output, ir)
        mse = MSE(output, ir)
        l1 = L1(output, ir)
        loss_sum += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        f_score += F1_Score(output.cpu().detach().numpy().copy(), ir.cpu().numpy().copy())[0]
        
    if (epoch+1) % 50 == 0 or epoch == 0:
        torch.save(model.state_dict(), f'{SAVE_DIR}/pt/{epoch}.pth')
        
    print('Epoch [{}/{}], Loss: {:.4f}, F1 Score: {:.4f}, MSE: {:.4f}, L1: {:.4f}'
            .format(epoch+1, num_epochs_pt, loss_sum/len(dataloader_fake), f_score/len(dataloader_fake), mse.item(), l1.item()))
    
    # wandb.log({'pt_loss':loss_sum/len(dataloader_fake), 'pt_f1':f_score/len(dataloader_fake)})


l1_sum=0
f1_sum=0
model.eval()
with torch.no_grad():
    for i,(data, data_org) in enumerate(zip(dataloader_test, dataloader_test_original_size)):
        maps = data[:,:-1,:,:].to(device)
        ir = data_org[:,-1,:,:].unsqueeze(1)
        output, x = model(maps)
        output = output.cpu().detach().numpy()[0, 0] / scale        # (512, 512)
        ir_np = ir.numpy()[0, 0]                                     # (H, W)
        output_resized = resize(output, ir_np.shape, preserve_range=True)
        output = torch.tensor(output_resized).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        l1_sum += L1(output, ir).item()
        f1_sum += F1_Score(output.numpy().copy(), ir.numpy().copy())[0]

print('****** After pretraining, L1 Loss: {:.8f}, F1 Score: {:.4f}'.format(l1_sum/len(dataloader_test), f1_sum/len(dataloader_test)))
# wandb.log({'after_pt_l1':l1_sum/len(dataloader_test), 'after_pt_f1':f1_sum/len(dataloader_test)})




######## Finetune ########
# model.load_state_dict(torch.load('../saved/pt/449.pth'))

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_ft)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs_ft, eta_min=learning_rate_min)


for epoch in range(num_epochs_ft):
    loss_sum = 0
    f_score = 0
    model.train()
    for i, data in enumerate(dataloader_real):
        maps = data[:,:-1,:,:]
        maps = maps.to(device)
        ir = data[:,-1,:,:].unsqueeze(1).to(device)*scale
        output,_ = model(maps)

        loss = criterion(output, ir)
        mse = MSE(output, ir)
        l1 = L1(output, ir)
        loss_sum += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        f_score += F1_Score(output.cpu().detach().numpy().copy(), ir.cpu().numpy().copy())[0]

    scheduler.step()

    print('Epoch [{}/{}], Loss: {:.4f}, F1 Score: {:.4f}, MSE: {:.4f}, L1: {:.4f}'
            .format(epoch+1, num_epochs_ft, loss_sum/len(dataloader_real), f_score/len(dataloader_real), mse.item(), l1.item()))
    # wandb.log({'ft_loss':loss_sum/len(dataloader_real), 'ft_f1':f_score/len(dataloader_real)})

        
    if (epoch+1) % 50 == 0 or epoch == 0:
        torch.save(model.state_dict(), f'{SAVE_DIR}/ft_real/{epoch}.pth')
        
        l1_sum=0
        f1_sum=0
        model.eval()
        with torch.no_grad():
            for i,(data, data_org) in enumerate(zip(dataloader_test, dataloader_test_original_size)):
                maps = data[:,:-1,:,:].to(device)
                ir = data_org[:,-1,:,:].unsqueeze(1)
                output, x = model(maps)
                output = output.cpu().detach().numpy()[0, 0] / scale        # (512, 512)
                ir_np = ir.numpy()[0, 0]                                     # (H, W)
                output_resized = resize(output, ir_np.shape, preserve_range=True)
                output = torch.tensor(output_resized).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
                l1_sum += L1(output, ir).item()
                f1_sum += F1_Score(output.numpy().copy(), ir.numpy().copy())[0]
        
        # fig, axs = plt.subplots(1,2, sharex=True, sharey=True, figsize=(10, 4))
        # sns.heatmap(output.numpy()[0,0,:],ax=axs[0])
        # sns.heatmap(ir.numpy()[0,0,:],ax=axs[1])
        # wandb_fig = wandb.Image(fig)
        # wandb.log({'After Finetuning Epoch: '+str(epoch+1):wandb_fig})

        print('****** After Finetuning Epoch: {}, L1 Loss: {:.8f}, F1 Score: {:.4f}'.format(epoch+1,l1_sum/len(dataloader_test), f1_sum/len(dataloader_test)))
        # wandb.log({'after_ft_l1':l1_sum/len(dataloader_test), 'after_ft_f1':f1_sum/len(dataloader_test)})

































