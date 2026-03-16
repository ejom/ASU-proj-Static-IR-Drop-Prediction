"""
Preprocess CSV data into .npy files for fast loading during training.

WHY: Loading hundreds of CSV files every epoch is very slow. This script
reads all CSVs once, normalizes and resizes them, then saves as compact
.npy (NumPy binary) files. Training then loads these .npy files which is
~100x faster.

Run once before training:
  python preprocess.py

Creates:
  ../data/fake-npy/       - synthetic circuit data (512x512)
  ../data/real-npy/       - real circuit data (512x512)
  ../data/hidden-npy/     - hidden test data (512x512, for model input)
  ../data/hidden-npy-orig/ - hidden test data (original resolution, for evaluation)

Each .npy file is a single sample: a (13, H, W) float32 array with:
  - Channels 0-11: normalized input features (divided by their max)
  - Channel 12:    raw IR drop values (NOT normalized)
"""

import os
import re
import numpy as np
from skimage.transform import resize
from collections import defaultdict

DATA_ROOT = '../data'
TARGET_SIZE = 512  # all training data is resized to 512x512


def _load_and_normalize(path):
    """Load a CSV and normalize to [0, 1] by dividing by its max value."""
    arr = np.genfromtxt(path, delimiter=',')
    if arr.max() > 0:
        arr = arr / arr.max()
    return arr


def preprocess_fake(src_dir, dst_dir):
    """Convert fake (synthetic) circuit CSV files to .npy format.

    Fake data files are named like: circuit_001_current.csv, circuit_001_ir_drop.csv
    They are grouped by circuit ID, and each group produces one (13, 512, 512) .npy file.

    Args:
        src_dir: directory with CSV files
        dst_dir: output directory for .npy files
    """
    os.makedirs(dst_dir, exist_ok=True)

    files = os.listdir(src_dir)
    # Group files by circuit ID (number after first underscore)
    grouped = defaultdict(list)
    for f in files:
        prefix = re.split(r'_|\.', f)[1]
        grouped[prefix].append(f)

    for idx, (prefix, group) in enumerate(sorted(grouped.items())):
        # Use the .sp (SPICE) file to determine the base name for this circuit
        sp_files = [f for f in group if f.endswith('.sp')]
        if not sp_files:
            continue
        base_name = sp_files[0].replace('.sp', '')
        out_path = os.path.join(dst_dir, f'{base_name}.npy')
        if os.path.exists(out_path):
            continue  # skip if already preprocessed

        print(f'  [{idx+1}] {base_name}')

        # Load each of the 12 input feature channels
        channels = []
        for key in ['_current.csv', '_eff_dist.csv', '_pdn_density.csv',
                     '_resistance_m1.csv', '_resistance_m4.csv', '_resistance_m7.csv',
                     '_resistance_m8.csv', '_resistance_m9.csv',
                     '_via_m1m4.csv', '_via_m4m7.csv', '_via_m7m8.csv', '_via_m8m9.csv']:
            path = os.path.join(src_dir, base_name + key)
            if os.path.exists(path):
                arr = _load_and_normalize(path)  # normalize to [0, 1]
            else:
                arr = np.zeros((TARGET_SIZE, TARGET_SIZE))  # missing feature → zeros
            arr = resize(arr, (TARGET_SIZE, TARGET_SIZE), preserve_range=True)
            channels.append(arr.astype(np.float32))

        # Load IR drop (channel 12) — NOT normalized, raw values preserved
        ir_path = os.path.join(src_dir, base_name + '_ir_drop.csv')
        ir = np.genfromtxt(ir_path, delimiter=',')
        ir = resize(ir, (TARGET_SIZE, TARGET_SIZE), preserve_range=True).astype(np.float32)
        channels.append(ir)

        # Stack into (13, 512, 512) and save
        stacked = np.stack(channels, axis=0)
        np.save(out_path, stacked)


def preprocess_real(src_dir, dst_dir, dst_dir_orig=None):
    """Convert real circuit data folders to .npy format.

    Real data is organized as folders (one per testcase), each containing CSV files.
    This creates two versions:
      1. 512x512 version (for training/model input)
      2. Original resolution version (for evaluation — predictions are resized
         back to this resolution for fair metric comparison)

    Args:
        src_dir:      root directory with testcase folders
        dst_dir:      output dir for 512x512 .npy files
        dst_dir_orig: output dir for original-size .npy files (optional)
    """
    os.makedirs(dst_dir, exist_ok=True)
    if dst_dir_orig:
        os.makedirs(dst_dir_orig, exist_ok=True)

    folders = sorted([f for f in os.listdir(src_dir)
                      if os.path.isdir(os.path.join(src_dir, f))])

    # List of (filename, should_normalize) pairs
    # All features are normalized except ir_drop_map which needs raw values
    feature_files = [
        ('current_map.csv', True),
        ('eff_dist_map.csv', True),
        ('pdn_density.csv', True),
        ('resistance_m1.csv', True),
        ('resistance_m4.csv', True),
        ('resistance_m7.csv', True),
        ('resistance_m8.csv', True),
        ('resistance_m9.csv', True),
        ('via_m1m4.csv', True),
        ('via_m4m7.csv', True),
        ('via_m7m8.csv', True),
        ('via_m8m9.csv', True),
        ('ir_drop_map.csv', False),  # target — NOT normalized
    ]

    for idx, folder in enumerate(folders):
        out_path = os.path.join(dst_dir, f'{folder}.npy')
        out_orig = os.path.join(dst_dir_orig, f'{folder}.npy') if dst_dir_orig else None
        # Skip if already preprocessed
        if dst_dir_orig:
            if os.path.exists(out_path) and os.path.exists(out_orig):
                continue
        else:
            if os.path.exists(out_path):
                continue

        print(f'  [{idx+1}/{len(folders)}] {folder}')
        folder_path = os.path.join(src_dir, folder)

        # Get native resolution from current_map (all features share this base shape)
        current = np.genfromtxt(os.path.join(folder_path, 'current_map.csv'), delimiter=',')
        base_shape = current.shape

        channels_512 = []   # 512x512 versions (for training)
        channels_orig = []  # original-size versions (for evaluation)

        for fname, normalize in feature_files:
            fpath = os.path.join(folder_path, fname)
            if os.path.exists(fpath):
                arr = np.genfromtxt(fpath, delimiter=',')
                if normalize and arr.max() > 0:
                    arr = arr / arr.max()
            else:
                arr = np.zeros(base_shape)

            # 512x512 version for training
            arr_512 = resize(arr, (TARGET_SIZE, TARGET_SIZE), preserve_range=True).astype(np.float32)
            channels_512.append(arr_512)

            # Original-size version for evaluation
            # current, dist, pdn, ir_drop are already at base_shape
            # resistance/via may differ and need resizing to match
            if fname in ('current_map.csv', 'eff_dist_map.csv', 'pdn_density.csv', 'ir_drop_map.csv'):
                channels_orig.append(arr.astype(np.float32))
            else:
                arr_orig = resize(arr, base_shape, preserve_range=True).astype(np.float32)
                channels_orig.append(arr_orig)

        # Save 512x512 version
        stacked_512 = np.stack(channels_512, axis=0)  # (13, 512, 512)
        np.save(out_path, stacked_512)

        # Save original-size version (if output dir provided)
        if dst_dir_orig:
            out_orig = os.path.join(dst_dir_orig, f'{folder}.npy')
            stacked_orig = np.stack(channels_orig, axis=0)  # (13, H_orig, W_orig)
            np.save(out_orig, stacked_orig)


if __name__ == '__main__':
    print('Preprocessing fake circuit data...')
    preprocess_fake(
        f'{DATA_ROOT}/fake-circuit-data-plus',
        f'{DATA_ROOT}/fake-npy'
    )

    print('Preprocessing real circuit data...')
    preprocess_real(
        f'{DATA_ROOT}/real-circuit-data-plus',
        f'{DATA_ROOT}/real-npy'
    )

    print('Preprocessing hidden test data...')
    preprocess_real(
        f'{DATA_ROOT}/hidden-real-circuit-data',
        f'{DATA_ROOT}/hidden-npy',
        f'{DATA_ROOT}/hidden-npy-orig'  # also save original-size for evaluation
    )

    print('Done! Preprocessed .npy files saved.')
