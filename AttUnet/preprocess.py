"""
Preprocess CSV data into .npy files for fast loading during training.
Run once before training: python preprocess.py

Creates:
  ../data/fake-npy/       - fake circuit data (512x512)
  ../data/real-npy/        - real circuit data (512x512)
  ../data/hidden-npy/      - hidden test data (512x512)
  ../data/hidden-npy-orig/ - hidden test data (original resolution)
"""

import os
import re
import numpy as np
from skimage.transform import resize
from collections import defaultdict

DATA_ROOT = '../data'
TARGET_SIZE = 512


def _load_and_normalize(path):
    arr = np.genfromtxt(path, delimiter=',')
    if arr.max() > 0:
        arr = arr / arr.max()
    return arr


def _load_or_zeros(path, shape):
    if os.path.exists(path):
        return _load_and_normalize(path)
    return np.zeros(shape)


def preprocess_fake(src_dir, dst_dir):
    """Convert fake-circuit-data-plus CSVs to (13, 512, 512) .npy files."""
    os.makedirs(dst_dir, exist_ok=True)

    files = os.listdir(src_dir)
    grouped = defaultdict(list)
    for f in files:
        prefix = re.split(r'_|\.', f)[1]
        grouped[prefix].append(f)

    for idx, (prefix, group) in enumerate(sorted(grouped.items())):
        sp_files = [f for f in group if f.endswith('.sp')]
        if not sp_files:
            continue
        base_name = sp_files[0].replace('.sp', '')
        out_path = os.path.join(dst_dir, f'{base_name}.npy')
        if os.path.exists(out_path):
            continue

        print(f'  [{idx+1}] {base_name}')

        file_map = {}
        for f in group:
            file_map[f] = os.path.join(src_dir, f)

        def find(keyword):
            for f, p in file_map.items():
                if keyword in f and f.endswith('.csv'):
                    return p
            return None

        channels = []
        for key in ['_current.csv', '_eff_dist.csv', '_pdn_density.csv',
                     '_resistance_m1.csv', '_resistance_m4.csv', '_resistance_m7.csv',
                     '_resistance_m8.csv', '_resistance_m9.csv',
                     '_via_m1m4.csv', '_via_m4m7.csv', '_via_m7m8.csv', '_via_m8m9.csv']:
            path = os.path.join(src_dir, base_name + key)
            if os.path.exists(path):
                arr = _load_and_normalize(path)
            else:
                arr = np.zeros((TARGET_SIZE, TARGET_SIZE))
            arr = resize(arr, (TARGET_SIZE, TARGET_SIZE), preserve_range=True)
            channels.append(arr.astype(np.float32))

        # ir_drop is NOT normalized
        ir_path = os.path.join(src_dir, base_name + '_ir_drop.csv')
        ir = np.genfromtxt(ir_path, delimiter=',')
        ir = resize(ir, (TARGET_SIZE, TARGET_SIZE), preserve_range=True).astype(np.float32)
        channels.append(ir)

        stacked = np.stack(channels, axis=0)  # (13, 512, 512)
        np.save(out_path, stacked)


def preprocess_real(src_dir, dst_dir, dst_dir_orig=None):
    """Convert real circuit data folders to .npy files.

    dst_dir: saves at 512x512 resolution
    dst_dir_orig: if provided, also saves at original resolution
    """
    os.makedirs(dst_dir, exist_ok=True)
    if dst_dir_orig:
        os.makedirs(dst_dir_orig, exist_ok=True)

    folders = sorted([f for f in os.listdir(src_dir)
                      if os.path.isdir(os.path.join(src_dir, f))])

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
        ('ir_drop_map.csv', False),  # not normalized
    ]

    for idx, folder in enumerate(folders):
        out_path = os.path.join(dst_dir, f'{folder}.npy')
        if os.path.exists(out_path):
            continue

        print(f'  [{idx+1}/{len(folders)}] {folder}')
        folder_path = os.path.join(src_dir, folder)

        # Get base shape from current_map
        current = np.genfromtxt(os.path.join(folder_path, 'current_map.csv'), delimiter=',')
        base_shape = current.shape

        channels_512 = []
        channels_orig = []

        for fname, normalize in feature_files:
            fpath = os.path.join(folder_path, fname)
            if os.path.exists(fpath):
                arr = np.genfromtxt(fpath, delimiter=',')
                if normalize and arr.max() > 0:
                    arr = arr / arr.max()
            else:
                arr = np.zeros(base_shape)

            # 512x512 version
            arr_512 = resize(arr, (TARGET_SIZE, TARGET_SIZE), preserve_range=True).astype(np.float32)
            channels_512.append(arr_512)

            # Original size version (resize resistance/via to match base_shape)
            if fname in ('current_map.csv', 'eff_dist_map.csv', 'pdn_density.csv', 'ir_drop_map.csv'):
                channels_orig.append(arr.astype(np.float32))
            else:
                arr_orig = resize(arr, base_shape, preserve_range=True).astype(np.float32)
                channels_orig.append(arr_orig)

        stacked_512 = np.stack(channels_512, axis=0)
        np.save(out_path, stacked_512)

        if dst_dir_orig:
            out_orig = os.path.join(dst_dir_orig, f'{folder}.npy')
            stacked_orig = np.stack(channels_orig, axis=0)
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
        f'{DATA_ROOT}/hidden-npy-orig'
    )

    print('Done! Preprocessed .npy files saved.')
