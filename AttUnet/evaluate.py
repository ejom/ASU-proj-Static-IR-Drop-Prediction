import torch
import torch.nn as nn
import argparse
from model import VCAttUNet
from DataLoad_normalization import load_real_original_size, load_npy
from metrics import F1_Score
from skimage.transform import resize
import os

# Argument parser to handle --model argument
parser = argparse.ArgumentParser(description='Evaluate VCAttUNet model for static IR drop prediction')
parser.add_argument('--model', type=str, required=True, help='Path to the model checkpoint to evaluate')
args = parser.parse_args()

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
datapath_test_512 = '../data/hidden-npy'
datapath_test_orig = '../data/hidden-npy-orig'

# Load test data
use_npy = os.path.isdir(datapath_test_512) and len(os.listdir(datapath_test_512)) > 0
if use_npy:
    dataset_test_512 = load_npy(datapath_test_512)
    dataset_test_orig = load_npy(datapath_test_orig)
else:
    datapath_test = '../data/hidden-real-circuit-data/'
    from DataLoad_normalization import load_real
    dataset_test_512 = load_real(datapath_test, mode='train', testcase=[])
    dataset_test_orig = load_real_original_size(datapath_test, mode='train', testcase=[])

dataloader_test_512 = torch.utils.data.DataLoader(dataset=dataset_test_512, batch_size=1, shuffle=False)
dataloader_test_orig = torch.utils.data.DataLoader(dataset=dataset_test_orig, batch_size=1, shuffle=False)

scale = 100
L1 = nn.L1Loss()


def evaluate_model(model, dataloader_512, dataloader_orig):
    assert len(dataloader_512) == len(dataloader_orig), \
        f'Test loader mismatch: {len(dataloader_512)} vs {len(dataloader_orig)}'
    model.eval()
    total_l1 = 0
    total_f1 = 0
    num_samples = 0

    with torch.no_grad():
        for (data, data_org) in zip(dataloader_512, dataloader_orig):
            maps = data[:, :-1, :, :].to(device)
            ir = data_org[:, -1:, :, :]

            output, _ = model(maps)
            output = output.cpu().detach().numpy()[0, 0] / scale
            ir_np = ir.numpy()[0, 0]

            output_resized = resize(output, ir_np.shape, preserve_range=True)
            output_t = torch.tensor(output_resized, dtype=ir.dtype).unsqueeze(0).unsqueeze(0)

            l1 = L1(output_t, ir).item()
            f1 = F1_Score(output_t.numpy().copy(), ir.numpy().copy())[0]

            total_l1 += l1
            total_f1 += f1
            num_samples += 1

            print(f'  Sample {num_samples} - L1: {l1:.6f}, F1: {f1:.4f}')

    avg_l1 = total_l1 / num_samples
    avg_f1 = total_f1 / num_samples

    print(f'\nAverage L1 (MAE): {avg_l1:.6f}')
    print(f'Average F1 Score: {avg_f1:.4f}')


def main():
    model = VCAttUNet(in_ch=12, out_ch=1)

    try:
        model.load_state_dict(torch.load(args.model, map_location=device))
        print(f"Loaded model from {args.model}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model = model.to(device)

    print("Starting evaluation...")
    evaluate_model(model, dataloader_test_512, dataloader_test_orig)


if __name__ == '__main__':
    main()
