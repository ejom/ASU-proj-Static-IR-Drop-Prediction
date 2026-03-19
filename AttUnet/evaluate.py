import torch
import torch.nn as nn
import numpy as np
import argparse
import os
import glob
import seaborn as sns
from matplotlib import pyplot as plt
from skimage.transform import resize

from DataLoad_normalization import load_real, load_real_original_size
from metrics import F1_Score
from model import VCAttUNet as net

# Args
parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--model', type=str, help='Path to a single .pth model')
group.add_argument('--model_dir', type=str, help='Directory of .pth models to sweep (picks best by F1)')
parser.add_argument('--data', type=str, default='../data/hidden-real-circuit-data/', help='Path to test data')
parser.add_argument('--save_plots', action='store_true', help='Save heatmaps to disk')
parser.add_argument('--plot_dir', type=str, default='./plots/', help='Directory to save plots')
args = parser.parse_args()

# Device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# Load test data - both use batch_size=1 so they stay aligned
dataset_test = load_real(args.data, mode='train', testcase=[])
dataloader_test = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=1, shuffle=False)

dataset_test_org = load_real_original_size(args.data, mode='train', testcase=[], print_name=True)
dataloader_test_org = torch.utils.data.DataLoader(dataset=dataset_test_org, batch_size=1, shuffle=False)

# Metrics
L1 = nn.L1Loss()
MSE = nn.MSELoss()

# Get test case names for labeling
testcase_names = dataset_test_org.folder_list


def evaluate_model(model_path, store_tensors=True):
    """Evaluate a single model, return avg L1, avg F1, and per-case results.

    Set store_tensors=False during sweeps to save memory.
    """
    model = net(in_ch=12, out_ch=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    l1_sum = 0
    f1_sum = 0
    n = len(dataloader_test)
    per_case = []

    with torch.no_grad():
        for i, (data, data_org) in enumerate(zip(dataloader_test, dataloader_test_org)):
            maps = data[:, :-1, :, :].to(device)
            ir = data_org[:, -1, :, :].unsqueeze(1)
            shape = ir.shape

            output, _ = model(maps)
            output = output / 100
            output = output.cpu().numpy()
            output = torch.tensor(resize(output, shape))

            l1 = L1(output, ir).item()
            mse = MSE(output, ir).item()
            f1 = F1_Score(output.numpy().copy(), ir.numpy().copy())[0]

            l1_sum += l1
            f1_sum += f1

            name = testcase_names[i] if i < len(testcase_names) else f'test_{i}'
            entry = {'name': name, 'l1': l1, 'f1': f1, 'mse': mse}
            if store_tensors:
                entry['output'] = output
                entry['ir'] = ir
            per_case.append(entry)

    return l1_sum / n, f1_sum / n, per_case


def print_detailed_results(model_path, avg_l1, avg_f1, per_case):
    """Print per-case results and generate plots."""
    print(f'\nLoaded model from {model_path}')
    print(f'\n{"Case":<15} {"L1":>10} {"F1":>10} {"MSE":>10}')
    print('-' * 50)

    for c in per_case:
        print(f'{c["name"]:<15} {c["l1"]:>10.6f} {c["f1"]:>10.4f} {c["mse"]:>10.6f}')

        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        sns.heatmap(c['output'].numpy()[0, 0, :], ax=axs[0])
        axs[0].set_title('Predicted IR Drop')
        sns.heatmap(c['ir'].numpy()[0, 0, :], ax=axs[1])
        axs[1].set_title('Ground Truth')
        sns.heatmap(np.abs(c['output'].numpy()[0, 0, :] - c['ir'].numpy()[0, 0, :]), ax=axs[2])
        axs[2].set_title('Absolute Error')
        plt.suptitle(f'{c["name"]} | L1: {c["l1"]:.6f} | F1: {c["f1"]:.4f}')
        plt.tight_layout()

        if args.save_plots:
            plt.savefig(f'{args.plot_dir}/{c["name"]}.png', dpi=150, bbox_inches='tight')
        plt.show()

    print('-' * 50)
    print(f'{"Average":<15} {avg_l1:>10.6f} {avg_f1:>10.4f}')


if args.save_plots:
    os.makedirs(args.plot_dir, exist_ok=True)

if args.model:
    # Single model evaluation
    avg_l1, avg_f1, per_case = evaluate_model(args.model)
    print_detailed_results(args.model, avg_l1, avg_f1, per_case)

else:
    # Sweep all .pth files in directory
    pth_files = sorted(glob.glob(os.path.join(args.model_dir, '*.pth')))
    if not pth_files:
        print(f'No .pth files found in {args.model_dir}')
        exit(1)

    print(f'Found {len(pth_files)} models in {args.model_dir}\n')
    print(f'{"Model":<40} {"Avg L1":>10} {"Avg F1":>10}')
    print('-' * 65)

    best_f1 = -1
    best_path = None
    best_l1 = None

    for path in pth_files:
        avg_l1, avg_f1, _ = evaluate_model(path, store_tensors=False)
        basename = os.path.basename(path)
        print(f'{basename:<40} {avg_l1:>10.6f} {avg_f1:>10.4f}')

        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_path = path
            best_l1 = avg_l1

    print('-' * 65)
    print(f'\nBest model: {os.path.basename(best_path)} (F1: {best_f1:.4f}, L1: {best_l1:.6f})')
    print('\n' + '=' * 65)
    print('DETAILED RESULTS FOR BEST MODEL')
    print('=' * 65)

    # Re-run best model with tensors for plots
    avg_l1, avg_f1, per_case = evaluate_model(best_path, store_tensors=True)
    print_detailed_results(best_path, avg_l1, avg_f1, per_case)
