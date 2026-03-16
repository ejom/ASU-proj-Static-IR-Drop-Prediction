"""
Evaluate a trained VCAttUNet model on the hidden test set.

This script is run AFTER training is complete. It loads a saved checkpoint,
runs the model on the hidden test circuits (never seen during training),
and reports per-sample and average metrics. It also generates a visual
comparison of predicted vs ground truth IR drop heatmaps.

Metrics:
  - MAE (Mean Absolute Error): average pixel-wise prediction error
  - F1 Score: overlap between predicted and actual hotspot regions
    (hotspot = pixel with IR drop > 90% of that sample's maximum)

Usage:
  python evaluate.py --model /content/drive/MyDrive/ir-drop-saved/ft_real/best_f1.pth
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

# --- Parse command-line arguments ---
parser = argparse.ArgumentParser(description='Evaluate VCAttUNet for static IR drop prediction')
parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint (.pth file)')
parser.add_argument('--save-dir', type=str, default=None,
                    help='Directory to save results (default: Google Drive if available, else ../results)')
args = parser.parse_args()

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Determine where to save output ---
if args.save_dir:
    SAVE_DIR = args.save_dir
elif os.path.isdir('/content/drive/MyDrive'):
    SAVE_DIR = '/content/drive/MyDrive/ir-drop-saved'
else:
    SAVE_DIR = '../results'
os.makedirs(SAVE_DIR, exist_ok=True)

# --- Load hidden test data ---
# We need two versions of the test data:
#   1. 512x512 version: fed into the model (which expects fixed 512x512 input)
#   2. Original-size version: for computing metrics at the native resolution
#      (predictions are resized back to original size for fair comparison)
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

# batch_size=1 because each test sample may have a different original resolution
dataloader_test_512 = torch.utils.data.DataLoader(dataset=dataset_test_512, batch_size=1, shuffle=False)
dataloader_test_orig = torch.utils.data.DataLoader(dataset=dataset_test_orig, batch_size=1, shuffle=False)

# The model predicts values scaled by 100 during training, so we divide by 100
# to get back to the original IR drop scale for evaluation
scale = 100


def evaluate_model(model, dataloader_512, dataloader_orig):
    """Run the model on all test samples and compute metrics.

    For each sample:
      1. Feed the 512x512 input features into the model
      2. Get the 512x512 prediction, divide by scale to undo training scaling
      3. Resize prediction to the original test circuit resolution
      4. Compare against ground truth at original resolution

    Args:
        model: trained VCAttUNet
        dataloader_512: test data at 512x512 (model input)
        dataloader_orig: test data at original resolution (for ground truth)
    """
    assert len(dataloader_512) == len(dataloader_orig), \
        f'Test loader mismatch: {len(dataloader_512)} vs {len(dataloader_orig)}'

    model.eval()  # disable BatchNorm updates and dropout
    num_samples = len(dataloader_512)
    total_mae = 0
    total_f1 = 0

    # Collect results for plotting
    predictions = []
    ground_truths = []
    errors = []
    sample_names = []
    sample_metrics = []

    with torch.no_grad():  # no gradient computation needed for evaluation
        for i, (data, data_org) in enumerate(zip(dataloader_512, dataloader_orig)):
            # Input features from 512x512 data (first 12 channels)
            maps = data[:, :-1, :, :].to(device)
            # Ground truth IR drop from original-size data (last channel)
            ir = data_org[:, -1:, :, :]

            # Run model and convert prediction to numpy at original scale
            output, _ = model(maps)
            pred = output.cpu().detach().numpy()[0, 0] / scale  # (512, 512)
            ir_np = ir.numpy()[0, 0]                             # (H_orig, W_orig)

            # Resize prediction from 512x512 back to the original circuit resolution
            # anti_aliasing=True smooths the result to avoid artifacts
            pred_resized = resize(pred, ir_np.shape, preserve_range=True, anti_aliasing=True)

            # --- Compute MAE (Mean Absolute Error) ---
            mae = np.mean(np.abs(pred_resized - ir_np))
            total_mae += mae

            # --- Compute F1 (hotspot detection accuracy) ---
            # F1_Score expects 4D arrays: (batch, channel, height, width)
            pred_4d = pred_resized[np.newaxis, np.newaxis, :, :]
            gt_4d = ir_np[np.newaxis, np.newaxis, :, :]
            f1 = F1_Score(pred_4d.copy(), gt_4d.copy())[0]
            total_f1 += f1

            # Store for heatmap visualization
            predictions.append(pred_resized)
            ground_truths.append(ir_np)
            errors.append(np.abs(pred_resized - ir_np))
            sample_names.append(f'Sample {i+1}')
            sample_metrics.append((mae, f1))

            # Print per-sample results with value ranges for debugging
            print(f'  {sample_names[-1]}: MAE={mae:.6f}, F1={f1:.4f}, '
                  f'pred=[{pred_resized.min():.6f}, {pred_resized.max():.6f}], '
                  f'gt=[{ir_np.min():.6f}, {ir_np.max():.6f}]')

    # --- Print summary ---
    avg_mae = total_mae / num_samples
    avg_f1 = total_f1 / num_samples

    print(f'\n{"="*50}')
    print(f'Average MAE:  {avg_mae:.6f}')
    print(f'Average F1:   {avg_f1:.4f}')
    print(f'{"="*50}')

    # --- Generate side-by-side heatmap visualization ---
    # Three columns per sample: Predicted | Ground Truth | Absolute Error
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        mae, f1 = sample_metrics[i]

        # Column 1: Model prediction
        axes[i, 0].imshow(predictions[i], cmap='hot')
        axes[i, 0].set_title(f'{sample_names[i]} - Predicted')
        axes[i, 0].axis('off')

        # Column 2: Ground truth
        axes[i, 1].imshow(ground_truths[i], cmap='hot')
        axes[i, 1].set_title(f'Ground Truth')
        axes[i, 1].axis('off')

        # Column 3: Absolute error (brighter = larger error)
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
    # Create model and load trained weights
    # dropout_rate doesn't matter for eval (dropout is disabled in eval mode)
    model = VCAttUNet(in_ch=12, out_ch=1, dropout_rate=0.0)

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
