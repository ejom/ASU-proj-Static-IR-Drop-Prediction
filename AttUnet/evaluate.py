import torch
import torch.nn as nn
import numpy as np
import argparse
import seaborn as sns
from matplotlib import pyplot as plt
from skimage.transform import resize

from DataLoad_normalization import load_real, load_real_original_size
from metrics import F1_Score
from model import VCAttUNet as net

# Args
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, help='Path to saved .pth model')
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

# Load model
model = net(in_ch=12, out_ch=1).to(device)
model.load_state_dict(torch.load(args.model, map_location=device))
model.eval()
print(f'Loaded model from {args.model}')

# Metrics
L1 = nn.L1Loss()
MSE = nn.MSELoss()

# Get test case names for labeling
testcase_names = dataset_test_org.folder_list

# Evaluate
l1_sum = 0
f1_sum = 0
n = len(dataloader_test)

if args.save_plots:
    import os
    os.makedirs(args.plot_dir, exist_ok=True)

print(f'\n{"Case":<15} {"L1":>10} {"F1":>10} {"MSE":>10}')
print('-' * 50)

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
        print(f'{name:<15} {l1:>10.6f} {f1:>10.4f} {mse:>10.6f}')

        # Plot
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))

        sns.heatmap(output.numpy()[0, 0, :], ax=axs[0])
        axs[0].set_title('Predicted IR Drop')

        sns.heatmap(ir.numpy()[0, 0, :], ax=axs[1])
        axs[1].set_title('Ground Truth')

        sns.heatmap(np.abs(output.numpy()[0, 0, :] - ir.numpy()[0, 0, :]), ax=axs[2])
        axs[2].set_title('Absolute Error')

        plt.suptitle(f'{name} | L1: {l1:.6f} | F1: {f1:.4f}')
        plt.tight_layout()

        if args.save_plots:
            plt.savefig(f'{args.plot_dir}/{name}.png', dpi=150, bbox_inches='tight')
        plt.show()

print('-' * 50)
print(f'{"Average":<15} {l1_sum/n:>10.6f} {f1_sum/n:>10.4f}')