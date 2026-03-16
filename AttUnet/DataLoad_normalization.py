"""
Data loading and normalization for IR drop prediction.

This module provides PyTorch Dataset classes that load circuit data and return
tensors ready for training. Each sample is a multi-channel tensor:

  Channels 0-11 (input features):
    0:  current_map     — current draw at each grid point (normalized 0-1)
    1:  eff_dist_map    — effective distance to nearest power pin (normalized 0-1)
    2:  pdn_density     — power delivery network density (normalized 0-1)
    3:  resistance_m1   — metal layer 1 resistance (normalized 0-1)
    4:  resistance_m4   — metal layer 4 resistance
    5:  resistance_m7   — metal layer 7 resistance
    6:  resistance_m8   — metal layer 8 resistance
    7:  resistance_m9   — metal layer 9 resistance
    8:  via_m1m4        — via resistance between layers 1-4
    9:  via_m4m7        — via resistance between layers 4-7
    10: via_m7m8        — via resistance between layers 7-8
    11: via_m8m9        — via resistance between layers 8-9

  Channel 12 (target):
    12: ir_drop         — ground truth IR drop map (NOT normalized, raw values)

Three dataset classes:
  - load_npy:                 fast loading from preprocessed .npy files (preferred)
  - load_fake:                loads synthetic circuit data from CSVs
  - load_real:                loads real circuit data from CSVs, resized to 512x512
  - load_real_original_size:  loads real circuit data at native resolution (for evaluation)
"""

import torch
from torch.utils import data
import os
import numpy as np
from skimage.transform import resize
import re
from collections import defaultdict


class load_npy(data.Dataset):
    """Fast dataset that loads preprocessed .npy files.

    Each .npy file contains a single sample as a (13, H, W) float32 array.
    This is much faster than loading from CSVs because the data is already
    parsed, normalized, and resized.

    Args:
        npy_dir:  directory containing .npy files
        mode:     'train' to include all samples EXCEPT testcase list,
                  'test' to include ONLY testcase list
        testcase: list of sample names to include/exclude (without .npy extension)
    """
    def __init__(self, npy_dir, mode='train', testcase=None):
        if testcase is None:
            testcase = []
        all_files = sorted([f for f in os.listdir(npy_dir) if f.endswith('.npy')])
        # Filter files based on mode and testcase list
        if mode == 'train':
            self.files = [f for f in all_files if os.path.splitext(f)[0] not in testcase]
        else:
            self.files = [f for f in all_files if os.path.splitext(f)[0] in testcase]
        self.files = [os.path.join(npy_dir, f) for f in self.files]
        print(f'load_npy: {len(self.files)} samples from {npy_dir}')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """Load and return one sample as a PyTorch tensor."""
        return torch.from_numpy(np.load(self.files[idx]))


def _safe_normalize(arr):
    """Normalize array to [0, 1] by dividing by its max value.

    Safely handles all-zero arrays (returns zeros instead of NaN).
    Used for input features but NOT for IR drop (which needs raw values).
    """
    if arr.max() > 0:
        return arr / arr.max()
    return arr


def _load_csv_or_zeros(filepath, shape):
    """Load a CSV file as a numpy array, or return zeros if file doesn't exist.

    Some circuit designs may not have all resistance/via layers, so missing
    files are treated as zero-filled maps.
    """
    if os.path.exists(filepath):
        arr = np.genfromtxt(filepath, delimiter=',')
        return _safe_normalize(arr)
    else:
        return np.zeros(shape)


def extract_data(line):
    """Parse a SPICE netlist line to extract resistance value and node coordinates.

    SPICE lines look like: R1 node_x1_y1 node_x2_y2 resistance_value
    Node names encode grid coordinates which are converted to array indices.

    Returns:
        (resistance, node1_coords, node2_coords) or (None, None, None) if not a resistor line
    """
    components = line.split()
    if len(components) >= 4 and components[0].startswith('R'):
        resistance = float(components[3])
        # Extract x,y coordinates from node names and convert to grid indices
        # The //2000 converts from physical coordinates to grid positions
        node1_coords = tuple(map(int, np.array(components[1].split('_')[-2:], dtype=float) // 2000))
        node2_coords = tuple(map(int, np.array(components[2].split('_')[-2:], dtype=float) // 2000))
        return resistance, node1_coords, node2_coords
    else:
        return None, None, None


def get_resistance(file_path):
    """Parse a SPICE netlist file to build resistance and via grids.

    Reads all resistor definitions and distributes resistance values to
    a 2D grid. Via connections (where both nodes already have resistance)
    are tracked separately.

    Returns:
        (resistance_grid, via_grid) — both are 2D numpy arrays
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # First pass: find grid dimensions
    max_x = 0
    max_y = 0
    for line in lines:
        _, node1_coords, node2_coords = extract_data(line)
        if node1_coords and node2_coords:
            max_x = max(max_x, max(node1_coords[0], node2_coords[0]))
            max_y = max(max_y, max(node1_coords[1], node2_coords[1]))

    resistance_grid = np.zeros((max_x + 1, max_y + 1))
    via_grid = np.zeros((max_x + 1, max_y + 1))

    # Second pass: populate grids
    # Each resistor's value is split equally between its two endpoint nodes
    for line in lines:
        resistance, node1_coords, node2_coords = extract_data(line)
        if resistance and node1_coords and node2_coords:
            # If both nodes already have resistance, this is a via connection
            if resistance_grid[node1_coords] != 0 and resistance_grid[node2_coords] != 0:
                via_grid[node1_coords] += resistance / 2
                via_grid[node2_coords] += resistance / 2

            resistance_grid[node1_coords] += resistance / 2
            resistance_grid[node2_coords] += resistance / 2

    return resistance_grid, via_grid


class load_fake(data.Dataset):
    """Dataset for synthetic (fake) circuit data stored as individual CSV files.

    Fake data files are named like: circuit_001_current.csv, circuit_001_ir_drop.csv, etc.
    This groups files by circuit ID and loads all 13 channels per sample.

    Note: This is the slow path. Use preprocess.py to convert to .npy first.
    """
    def __init__(self, root):
        maps = os.listdir(root)
        # Group files by circuit ID (the number after the first underscore)
        grouped_items = defaultdict(list)
        for m in maps:
            prefix = re.split(r'_|\.', m)[1]  # extract circuit ID
            grouped_items[prefix].append(m)

        grouped_list = list(grouped_items.values())
        for i in range(len(grouped_list)):
            grouped_list[i] = [os.path.join(root, m) for m in grouped_list[i]]
        self.maps = grouped_list
        self.maps.sort(reverse=False)

    def __getitem__(self, index):
        """Load all feature CSVs for one circuit, normalize, resize to 512x512."""
        collect_m = self.maps[index]
        for m_path in collect_m:
            # Match each file to its feature type and load accordingly
            # All features except ir_drop are normalized to [0, 1]
            if '_current.csv' in m_path:
                current = np.genfromtxt(m_path, delimiter=',')
                current = _safe_normalize(current)
                current = torch.tensor(resize(current, [512, 512])).float().unsqueeze(0)
            elif 'dist' in m_path:
                dist = np.genfromtxt(m_path, delimiter=',')
                dist = _safe_normalize(dist)
                dist = torch.tensor(resize(dist, [512, 512])).float().unsqueeze(0)
            elif 'pdn' in m_path:
                pdn = np.genfromtxt(m_path, delimiter=',')
                pdn = _safe_normalize(pdn)
                pdn = torch.tensor(resize(pdn, [512, 512])).float().unsqueeze(0)
            elif 'ir_drop' in m_path:
                # IR drop is NOT normalized — we need raw values for the loss function
                ir_drop = np.genfromtxt(m_path, delimiter=',')
                ir_drop = torch.tensor(resize(ir_drop, [512, 512], preserve_range=True)).float().unsqueeze(0)
            elif 'resistance_m1' in m_path:
                resistance_m1 = np.genfromtxt(m_path, delimiter=',')
                resistance_m1 = _safe_normalize(resistance_m1)
                resistance_m1 = torch.tensor(resize(resistance_m1, [512, 512])).float().unsqueeze(0)
            elif 'resistance_m4' in m_path:
                resistance_m4 = np.genfromtxt(m_path, delimiter=',')
                resistance_m4 = _safe_normalize(resistance_m4)
                resistance_m4 = torch.tensor(resize(resistance_m4, [512, 512])).float().unsqueeze(0)
            elif 'resistance_m7' in m_path:
                resistance_m7 = np.genfromtxt(m_path, delimiter=',')
                resistance_m7 = _safe_normalize(resistance_m7)
                resistance_m7 = torch.tensor(resize(resistance_m7, [512, 512])).float().unsqueeze(0)
            elif 'resistance_m8' in m_path:
                resistance_m8 = np.genfromtxt(m_path, delimiter=',')
                resistance_m8 = _safe_normalize(resistance_m8)
                resistance_m8 = torch.tensor(resize(resistance_m8, [512, 512])).float().unsqueeze(0)
            elif 'resistance_m9' in m_path:
                resistance_m9 = np.genfromtxt(m_path, delimiter=',')
                resistance_m9 = _safe_normalize(resistance_m9)
                resistance_m9 = torch.tensor(resize(resistance_m9, [512, 512])).float().unsqueeze(0)
            elif 'via_m1m4' in m_path:
                via_m1m4 = np.genfromtxt(m_path, delimiter=',')
                via_m1m4 = _safe_normalize(via_m1m4)
                via_m1m4 = torch.tensor(resize(via_m1m4, [512, 512])).float().unsqueeze(0)
            elif 'via_m4m7' in m_path:
                via_m4m7 = np.genfromtxt(m_path, delimiter=',')
                via_m4m7 = _safe_normalize(via_m4m7)
                via_m4m7 = torch.tensor(resize(via_m4m7, [512, 512])).float().unsqueeze(0)
            elif 'via_m7m8' in m_path:
                via_m7m8 = np.genfromtxt(m_path, delimiter=',')
                via_m7m8 = _safe_normalize(via_m7m8)
                via_m7m8 = torch.tensor(resize(via_m7m8, [512, 512])).float().unsqueeze(0)
            elif 'via_m8m9' in m_path:
                via_m8m9 = np.genfromtxt(m_path, delimiter=',')
                via_m8m9 = _safe_normalize(via_m8m9)
                via_m8m9 = torch.tensor(resize(via_m8m9, [512, 512])).float().unsqueeze(0)

        # Stack all 12 input features + 1 target into a (13, 512, 512) tensor
        data = torch.concat([current, dist, pdn, resistance_m1, resistance_m4, resistance_m7,
                             resistance_m8, resistance_m9, via_m1m4, via_m4m7, via_m7m8, via_m8m9,
                             ir_drop], dim=0)

        return data

    def __len__(self):
        return len(self.maps)


class load_real(data.Dataset):
    """Dataset for real circuit data, resized to 512x512 for training.

    Real data is organized in folders: one folder per test case, each containing
    CSV files for the 12 features + IR drop ground truth.

    Args:
        folder_path: root directory containing testcase folders
        mode:       'train' to exclude testcase list, 'test' to include only testcase list
        testcase:   list of folder names to include/exclude
    """
    def __init__(self, folder_path, mode='train', testcase=None):
        if testcase is None:
            testcase = []
        self.folder_path = folder_path
        self.folder_list = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
        self.folder_list = sorted(self.folder_list)
        print(self.folder_list)
        if mode == 'train':
            self.folder_list = [f for f in self.folder_list if f not in testcase]
        else:
            self.folder_list = [f for f in self.folder_list if f in testcase]

    def __len__(self):
        return len(self.folder_list)

    def __getitem__(self, index):
        """Load all features for one testcase, normalize, and resize to 512x512."""
        folder_name = self.folder_list[index]
        folder_dir = os.path.join(self.folder_path, folder_name)

        # Load current map first to get the native resolution
        current = np.genfromtxt(os.path.join(folder_dir, 'current_map.csv'), delimiter=',')
        base_shape = current.shape  # native resolution (varies per testcase)
        current = _safe_normalize(current)
        current = torch.tensor(resize(current, [512, 512])).float().unsqueeze(0)

        dist = np.genfromtxt(os.path.join(folder_dir, 'eff_dist_map.csv'), delimiter=',')
        dist = _safe_normalize(dist)
        dist = torch.tensor(resize(dist, [512, 512])).float().unsqueeze(0)

        pdn = np.genfromtxt(os.path.join(folder_dir, 'pdn_density.csv'), delimiter=',')
        pdn = _safe_normalize(pdn)
        pdn = torch.tensor(resize(pdn, [512, 512])).float().unsqueeze(0)

        # Resistance and via maps may not exist for all designs — use zeros if missing
        resistance_m1 = _load_csv_or_zeros(os.path.join(folder_dir, 'resistance_m1.csv'), base_shape)
        resistance_m1 = torch.tensor(resize(resistance_m1, [512, 512])).float().unsqueeze(0)

        resistance_m4 = _load_csv_or_zeros(os.path.join(folder_dir, 'resistance_m4.csv'), base_shape)
        resistance_m4 = torch.tensor(resize(resistance_m4, [512, 512])).float().unsqueeze(0)

        resistance_m7 = _load_csv_or_zeros(os.path.join(folder_dir, 'resistance_m7.csv'), base_shape)
        resistance_m7 = torch.tensor(resize(resistance_m7, [512, 512])).float().unsqueeze(0)

        resistance_m8 = _load_csv_or_zeros(os.path.join(folder_dir, 'resistance_m8.csv'), base_shape)
        resistance_m8 = torch.tensor(resize(resistance_m8, [512, 512])).float().unsqueeze(0)

        resistance_m9 = _load_csv_or_zeros(os.path.join(folder_dir, 'resistance_m9.csv'), base_shape)
        resistance_m9 = torch.tensor(resize(resistance_m9, [512, 512])).float().unsqueeze(0)

        via_m1m4 = _load_csv_or_zeros(os.path.join(folder_dir, 'via_m1m4.csv'), base_shape)
        via_m1m4 = torch.tensor(resize(via_m1m4, [512, 512])).float().unsqueeze(0)

        via_m4m7 = _load_csv_or_zeros(os.path.join(folder_dir, 'via_m4m7.csv'), base_shape)
        via_m4m7 = torch.tensor(resize(via_m4m7, [512, 512])).float().unsqueeze(0)

        via_m7m8 = _load_csv_or_zeros(os.path.join(folder_dir, 'via_m7m8.csv'), base_shape)
        via_m7m8 = torch.tensor(resize(via_m7m8, [512, 512])).float().unsqueeze(0)

        via_m8m9 = _load_csv_or_zeros(os.path.join(folder_dir, 'via_m8m9.csv'), base_shape)
        via_m8m9 = torch.tensor(resize(via_m8m9, [512, 512])).float().unsqueeze(0)

        # IR drop is NOT normalized — raw values preserved with preserve_range=True
        ir_drop = np.genfromtxt(os.path.join(folder_dir, 'ir_drop_map.csv'), delimiter=',')
        ir_drop = torch.tensor(resize(ir_drop, [512, 512], preserve_range=True)).float().unsqueeze(0)

        # Stack into (13, 512, 512) tensor: 12 features + 1 target
        data = torch.concat([current, dist, pdn, resistance_m1, resistance_m4, resistance_m7,
                             resistance_m8, resistance_m9, via_m1m4, via_m4m7, via_m7m8, via_m8m9,
                             ir_drop], dim=0)

        return data


class load_real_original_size(data.Dataset):
    """Dataset for real circuit data at ORIGINAL (native) resolution.

    Same data as load_real, but NOT resized to 512x512. Used during evaluation
    to compare predictions at the actual circuit resolution. The model output
    (at 512x512) gets resized back to this resolution for fair comparison.

    The native resolution varies per testcase (e.g., 200x300, 400x500, etc.).
    """
    def __init__(self, folder_path, mode='train', testcase=None, print_name=False):
        if testcase is None:
            testcase = []
        self.folder_path = folder_path
        self.folder_list = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
        self.folder_list = sorted(self.folder_list)
        if mode == 'train':
            self.folder_list = [f for f in self.folder_list if f not in testcase]
        else:
            self.folder_list = [f for f in self.folder_list if f in testcase]
        if print_name:
            print(self.folder_list)

    def __len__(self):
        return len(self.folder_list)

    def __folderlist__(self):
        return self.folder_list

    def __getitem__(self, index):
        """Load all features at native resolution (no resizing except resistance/via)."""
        folder_name = self.folder_list[index]
        folder_dir = os.path.join(self.folder_path, folder_name)

        # Load at native resolution — NO resize to 512x512
        current = np.genfromtxt(os.path.join(folder_dir, 'current_map.csv'), delimiter=',')
        shape = current.shape
        current = _safe_normalize(current)
        current = torch.tensor(current).float().unsqueeze(0)

        dist = np.genfromtxt(os.path.join(folder_dir, 'eff_dist_map.csv'), delimiter=',')
        dist = _safe_normalize(dist)
        dist = torch.tensor(dist).float().unsqueeze(0)

        pdn = np.genfromtxt(os.path.join(folder_dir, 'pdn_density.csv'), delimiter=',')
        pdn = _safe_normalize(pdn)
        pdn = torch.tensor(pdn).float().unsqueeze(0)

        # Resistance/via maps may have different native sizes — resize to match current_map
        resistance_m1 = _load_csv_or_zeros(os.path.join(folder_dir, 'resistance_m1.csv'), shape)
        resistance_m1 = torch.tensor(resize(resistance_m1, shape)).float().unsqueeze(0)

        resistance_m4 = _load_csv_or_zeros(os.path.join(folder_dir, 'resistance_m4.csv'), shape)
        resistance_m4 = torch.tensor(resize(resistance_m4, shape)).float().unsqueeze(0)

        resistance_m7 = _load_csv_or_zeros(os.path.join(folder_dir, 'resistance_m7.csv'), shape)
        resistance_m7 = torch.tensor(resize(resistance_m7, shape)).float().unsqueeze(0)

        resistance_m8 = _load_csv_or_zeros(os.path.join(folder_dir, 'resistance_m8.csv'), shape)
        resistance_m8 = torch.tensor(resize(resistance_m8, shape)).float().unsqueeze(0)

        resistance_m9 = _load_csv_or_zeros(os.path.join(folder_dir, 'resistance_m9.csv'), shape)
        resistance_m9 = torch.tensor(resize(resistance_m9, shape)).float().unsqueeze(0)

        via_m1m4 = _load_csv_or_zeros(os.path.join(folder_dir, 'via_m1m4.csv'), shape)
        via_m1m4 = torch.tensor(resize(via_m1m4, shape)).float().unsqueeze(0)

        via_m4m7 = _load_csv_or_zeros(os.path.join(folder_dir, 'via_m4m7.csv'), shape)
        via_m4m7 = torch.tensor(resize(via_m4m7, shape)).float().unsqueeze(0)

        via_m7m8 = _load_csv_or_zeros(os.path.join(folder_dir, 'via_m7m8.csv'), shape)
        via_m7m8 = torch.tensor(resize(via_m7m8, shape)).float().unsqueeze(0)

        via_m8m9 = _load_csv_or_zeros(os.path.join(folder_dir, 'via_m8m9.csv'), shape)
        via_m8m9 = torch.tensor(resize(via_m8m9, shape)).float().unsqueeze(0)

        # IR drop at native resolution (raw values, no normalization)
        ir_drop = np.genfromtxt(os.path.join(folder_dir, 'ir_drop_map.csv'), delimiter=',')
        ir_drop = torch.tensor(ir_drop).float().unsqueeze(0)

        data = torch.concat([current, dist, pdn, resistance_m1, resistance_m4, resistance_m7,
                             resistance_m8, resistance_m9, via_m1m4, via_m4m7, via_m7m8, via_m8m9,
                             ir_drop], dim=0)

        return data
