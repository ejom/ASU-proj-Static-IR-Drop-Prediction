"""
Evaluate old model checkpoints (trained before the architecture changes).

The old model used:
  - VCAttUNet with n1=16, pre=conv_block(12->8), no dropout
  - Filters: [16, 32, 64, 128, 256]

Usage:
  python evaluate_old.py --model /content/drive/MyDrive/ir-drop-test/ft_real/599.pth
"""

import torch
import torch.nn as nn
import numpy as np
import argparse
import os
import warnings
import matplotlib.pyplot as plt
from skimage.transform import resize
from sklearn.metrics import f1_score
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# =====================================================================
# Old model architecture (n1=16, pre=conv_block, no dropout)
# =====================================================================

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi, (x, psi)


class VCAttUNet_Old(nn.Module):
    """Old model: n1=16, pre=conv_block(12->8), no dropout."""

    def __init__(self, in_ch=12, out_ch=1):
        super(VCAttUNet_Old, self).__init__()

        n1 = 16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.pre = conv_block(ch_in=in_ch, ch_out=8)

        self.Conv1 = conv_block(8, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Up5 = up_conv(filters[4], filters[3])
        self.Att5 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=8)
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.pre(x)

        e1 = self.Conv1(x)
        e2 = self.Maxpool1(e1)

        e2 = self.Conv2(e2)
        e3 = self.Maxpool2(e2)

        e3 = self.Conv3(e3)
        e4 = self.Maxpool3(e3)

        e4 = self.Conv4(e4)
        e5 = self.Maxpool4(e4)

        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        e4, att5 = self.Att5(g=d5, x=e4)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        e3, att4 = self.Att4(g=d4, x=e3)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        e2, att3 = self.Att3(g=d3, x=e2)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        e1, att2 = self.Att2(g=d2, x=e1)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        return out, x


# =====================================================================
# F1 Score (same as current metrics.py)
# =====================================================================

def F1_Score(x, y):
    scores = []
    for i in range(x.shape[0]):
        pred = x[i, 0].copy()
        gt = y[i, 0].copy()
        pred_bin = (pred > 0.9 * pred.max()).astype(np.uint8)
        gt_bin = (gt > 0.9 * gt.max()).astype(np.uint8)
        scores.append(
            f1_score(gt_bin.flatten(), pred_bin.flatten(), average='binary', zero_division=0)
        )
    return scores


# =====================================================================
# Data loading (reuse current load_npy)
# =====================================================================

from DataLoad_normalization import load_npy, load_real, load_real_original_size

parser = argparse.ArgumentParser(description='Evaluate old VCAttUNet checkpoint')
parser.add_argument('--model', type=str, required=True, help='Path to old model checkpoint')
parser.add_argument('--save-dir', type=str, default=None)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.save_dir:
    SAVE_DIR = args.save_dir
elif os.path.isdir('/content/drive/MyDrive'):
    SAVE_DIR = '/content/drive/MyDrive/ir-drop-saved'
else:
    SAVE_DIR = '../results'
os.makedirs(SAVE_DIR, exist_ok=True)

datapath_test_512 = '../data/hidden-npy'
datapath_test_orig = '../data/hidden-npy-orig'

use_npy = os.path.isdir(datapath_test_512) and len(os.listdir(datapath_test_512)) > 0
if use_npy:
    dataset_test_512 = load_npy(datapath_test_512)
    dataset_test_orig = load_npy(datapath_test_orig)
else:
    datapath_test = '../data/hidden-real-circuit-data/'
    dataset_test_512 = load_real(datapath_test, mode='train')
    dataset_test_orig = load_real_original_size(datapath_test, mode='train')

dataloader_test_512 = torch.utils.data.DataLoader(dataset=dataset_test_512, batch_size=1, shuffle=False)
dataloader_test_orig = torch.utils.data.DataLoader(dataset=dataset_test_orig, batch_size=1, shuffle=False)

scale = 100


def evaluate_model(model, dataloader_512, dataloader_orig):
    assert len(dataloader_512) == len(dataloader_orig)

    model.eval()
    num_samples = len(dataloader_512)
    total_mae = 0
    total_f1 = 0

    predictions = []
    ground_truths = []
    errors = []
    sample_names = []
    sample_metrics = []

    with torch.no_grad():
        for i, (data, data_org) in enumerate(zip(dataloader_512, dataloader_orig)):
            maps = data[:, :-1, :, :].to(device)
            ir = data_org[:, -1:, :, :]

            output, _ = model(maps)
            pred = output.cpu().detach().numpy()[0, 0] / scale
            ir_np = ir.numpy()[0, 0]

            pred_resized = resize(pred, ir_np.shape, preserve_range=True, anti_aliasing=True)

            mae = np.mean(np.abs(pred_resized - ir_np))
            total_mae += mae

            pred_4d = pred_resized[np.newaxis, np.newaxis, :, :]
            gt_4d = ir_np[np.newaxis, np.newaxis, :, :]
            f1 = F1_Score(pred_4d.copy(), gt_4d.copy())[0]
            total_f1 += f1

            predictions.append(pred_resized)
            ground_truths.append(ir_np)
            errors.append(np.abs(pred_resized - ir_np))
            sample_names.append(f'Sample {i+1}')
            sample_metrics.append((mae, f1))

            print(f'  {sample_names[-1]}: MAE={mae:.6f}, F1={f1:.4f}, '
                  f'pred=[{pred_resized.min():.6f}, {pred_resized.max():.6f}], '
                  f'gt=[{ir_np.min():.6f}, {ir_np.max():.6f}]')

    avg_mae = total_mae / num_samples
    avg_f1 = total_f1 / num_samples

    print(f'\n{"="*50}')
    print(f'Average MAE:  {avg_mae:.6f}')
    print(f'Average F1:   {avg_f1:.4f}')
    print(f'{"="*50}')

    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        mae, f1 = sample_metrics[i]
        axes[i, 0].imshow(predictions[i], cmap='hot')
        axes[i, 0].set_title(f'{sample_names[i]} - Predicted')
        axes[i, 0].axis('off')
        axes[i, 1].imshow(ground_truths[i], cmap='hot')
        axes[i, 1].set_title(f'Ground Truth')
        axes[i, 1].axis('off')
        axes[i, 2].imshow(np.abs(predictions[i] - ground_truths[i]), cmap='hot')
        axes[i, 2].set_title(f'|Error| MAE={mae:.6f} F1={f1:.4f}')
        axes[i, 2].axis('off')

    plt.tight_layout()
    fig_path = os.path.join(SAVE_DIR, 'eval_results_old.png')
    plt.savefig(fig_path, dpi=150)
    print(f'\nHeatmap saved to {fig_path}')
    plt.show()
    plt.close(fig)


def main():
    model = VCAttUNet_Old(in_ch=12, out_ch=1)

    try:
        model.load_state_dict(torch.load(args.model, map_location=device))
        print(f'Loaded old model from {args.model}')
    except Exception as e:
        print(f'Error loading model: {e}')
        return

    model = model.to(device)
    print(f'Device: {device}')
    print(f'Save directory: {SAVE_DIR}\n')

    evaluate_model(model, dataloader_test_512, dataloader_test_orig)


if __name__ == '__main__':
    main()
