"""
Evaluate a trained VCAttUNet model on the hidden test set.

Prints per-sample and average L1 (MAE) and F1 scores, and saves
a heatmap visualization comparing predictions to ground truth.

Usage:
  python evaluate.py --model <path_to_checkpoint>
  python evaluate.py --model /content/drive/MyDrive/ir-drop-saved/ft_real/599.pth
  python evaluate.py --model ../saved/ft_real/599.pth --save-dir ../results
"""

import torch
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from skimage.transform import resize

from model import VCAttUNet
from DataLoad_normalization import load_real_original_size, load_npy
from metrics import F1_Score

# Argument parser
parser = argparse.ArgumentParser(description='Evaluate VCAttUNet for static IR drop prediction')
parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
parser.add_argument('--save-dir', type=str, default=None,
                    help='Directory to save results (default: Google Drive if available, else ../results)')
args = parser.parse_args()

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Determine save directory
if args.save_dir:
    SAVE_DIR = args.save_dir
elif os.path.isdir('/content/drive/MyDrive'):
    SAVE_DIR = '/content/drive/MyDrive/ir-drop-saved'
else:
    SAVE_DIR = '../results'
os.makedirs(SAVE_DIR, exist_ok=True)

# Load test data
datapath_test_512 = '../data/hidden-npy'
datapath_test_orig = '../data/hidden-npy-orig'

use_npy = os.path.isdir(datapath_test_512) and len(os.listdir(datapath_test_512)) > 0
if use_npy:
    dataset_test_512 = load_npy(datapath_test_512)
    dataset_test_orig = load_npy(datapath_test_orig)
else:
    datapath_test = '../data/hidden-real-circuit-data/'
    from DataLoad_normalization import load_real
    dataset_test_512 = load_real(datapath_test, mode='train')
    dataset_test_orig = load_real_original_size(datapath_test, mode='train')

dataloader_test_512 = torch.utils.data.DataLoader(dataset=dataset_test_512, batch_size=1, shuffle=False)
dataloader_test_orig = torch.utils.data.DataLoader(dataset=dataset_test_orig, batch_size=1, shuffle=False)

scale = 100


def evaluate_model(model, dataloader_512, dataloader_orig):
    assert len(dataloader_512) == len(dataloader_orig), \
        f'Test loader mismatch: {len(dataloader_512)} vs {len(dataloader_orig)}'

    model.eval()
    num_samples = len(dataloader_512)
    total_mae = 0
    total_f1 = 0

    # Collect results for plotting
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

            pred_resized = resize(pred, ir_np.shape, preserve_range=True)

            # Compute MAE
            mae = np.mean(np.abs(pred_resized - ir_np))
            total_mae += mae

            # Compute F1 (top 10% hotspot threshold)
            pred_4d = pred_resized[np.newaxis, np.newaxis, :, :]
            gt_4d = ir_np[np.newaxis, np.newaxis, :, :]
            f1 = F1_Score(pred_4d.copy(), gt_4d.copy())[0]
            total_f1 += f1

            # Store for plotting
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

    # Generate heatmap visualization
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

        axes[i, 2].imshow(errors[i], cmap='hot')
        axes[i, 2].set_title(f'|Error| MAE={mae:.6f} F1={f1:.4f}')
        axes[i, 2].axis('off')

    plt.tight_layout()
    fig_path = os.path.join(SAVE_DIR, 'eval_results.png')
    plt.savefig(fig_path, dpi=150)
    print(f'\nHeatmap saved to {fig_path}')
    plt.show()
    plt.close(fig)

    return avg_mae, avg_f1


def main():
    model = VCAttUNet(in_ch=12, out_ch=1)

    try:
        model.load_state_dict(torch.load(args.model, map_location=device))
        print(f'Loaded model from {args.model}')
    except Exception as e:
        print(f'Error loading model: {e}')
        return

    model = model.to(device)
    print(f'Device: {device}')
    print(f'Save directory: {SAVE_DIR}\n')

    evaluate_model(model, dataloader_test_512, dataloader_test_orig)


if __name__ == '__main__':
    main()
