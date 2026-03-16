import torch
from torch.utils import data
import os
import numpy as np
from skimage.transform import resize
import torchvision.transforms as transforms
import re
from collections import defaultdict


class load_npy(data.Dataset):
    """Fast dataset that loads preprocessed .npy files."""
    def __init__(self, npy_dir, mode='train', testcase=[]):
        all_files = sorted([f for f in os.listdir(npy_dir) if f.endswith('.npy')])
        if mode == 'train':
            self.files = [f for f in all_files if os.path.splitext(f)[0] not in testcase]
        else:
            self.files = [f for f in all_files if os.path.splitext(f)[0] in testcase]
        self.files = [os.path.join(npy_dir, f) for f in self.files]
        print(f'load_npy: {len(self.files)} samples from {npy_dir}')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return torch.from_numpy(np.load(self.files[idx]))


def _safe_normalize(arr):
    """Normalize array by its max, safely handling all-zero arrays."""
    if arr.max() > 0:
        return arr / arr.max()
    return arr


def _load_csv_or_zeros(filepath, shape):
    """Load a CSV file, or return a zero array if the file doesn't exist."""
    if os.path.exists(filepath):
        arr = np.genfromtxt(filepath, delimiter=',')
        return _safe_normalize(arr)
    else:
        return np.zeros(shape)



# Create a function to extract resistance and node coordinates
def extract_data(line):
    components = line.split()
    if len(components) >= 4 and components[0].startswith('R'):
        resistance = float(components[3])  # Resistance value
        node1_coords = tuple(map(int, np.array(components[1].split('_')[-2:], dtype=float) // 2000))  # Coordinates of node 1
        node2_coords = tuple(map(int, np.array(components[2].split('_')[-2:], dtype=float) // 2000))  # Coordinates of node 2
        return resistance, node1_coords, node2_coords
    else:
        return None, None, None


def get_resistance(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    # Find the grid size (maximum x, y coordinates)
    max_x = 0
    max_y = 0
    for line in lines:
        _, node1_coords, node2_coords = extract_data(line)
        if node1_coords and node2_coords:
            max_x = max(max_x, max(node1_coords[0], node2_coords[0]))
            max_y = max(max_y, max(node1_coords[1], node2_coords[1]))

    # Create a grid for resistance distribution
    resistance_grid = np.zeros((max_x + 1, max_y + 1))
    via_grid = np.zeros((max_x + 1, max_y + 1))

    # Populate the resistance grid
    for line in lines:
        resistance, node1_coords, node2_coords = extract_data(line)
        if resistance and node1_coords and node2_coords:
            if resistance_grid[node1_coords] != 0 and resistance_grid[node2_coords] != 0:
                via_grid[node1_coords] += resistance/2
                via_grid[node2_coords] += resistance/2

            resistance_grid[node1_coords] += resistance / 2
            resistance_grid[node2_coords] += resistance / 2

    # Print or visualize the resistance grid
    return resistance_grid, via_grid


class load_fake(data.Dataset):
    def __init__(self, root):
        maps = os.listdir(root)
        grouped_items = defaultdict(list)
        for m in maps:
            prefix = re.split(r'_|\.',m)[1]
            grouped_items[prefix].append(m)
            
        grouped_list = list(grouped_items.values())
        for i in range(len(grouped_list)):
            grouped_list[i] = [os.path.join(root, m) for m in grouped_list[i]]
        self.maps = grouped_list
        self.maps.sort(reverse=False)
    
    def __getitem__(self, index):
        collect_m = self.maps[index]
        for m_path in collect_m:
            if '_current.csv' in m_path:
                current = np.genfromtxt(m_path, delimiter = ',')
                current = _safe_normalize(current)
                current = torch.tensor(resize(current, [512,512])).float().unsqueeze(0)
            elif 'dist' in m_path:
                dist = np.genfromtxt(m_path, delimiter = ',')
                dist = _safe_normalize(dist)
                dist = torch.tensor(resize(dist, [512,512])).float().unsqueeze(0)
            elif 'pdn' in m_path:
                pdn = np.genfromtxt(m_path, delimiter = ',')
                pdn = _safe_normalize(pdn)
                pdn = torch.tensor(resize(pdn, [512,512])).float().unsqueeze(0)
            elif 'ir_drop' in m_path:
                ir_drop = np.genfromtxt(m_path, delimiter = ',')
                ir_drop = torch.tensor(resize(ir_drop, [512,512])).float().unsqueeze(0)
            elif 'resistance_m1' in m_path:
                resistance_m1 = np.genfromtxt(m_path, delimiter = ',')
                resistance_m1 = _safe_normalize(resistance_m1)
                resistance_m1 = torch.tensor(resize(resistance_m1, [512,512])).float().unsqueeze(0)
            elif 'resistance_m4' in m_path:
                resistance_m4 = np.genfromtxt(m_path, delimiter = ',')
                resistance_m4 = _safe_normalize(resistance_m4)
                resistance_m4 = torch.tensor(resize(resistance_m4, [512,512])).float().unsqueeze(0)
            elif 'resistance_m7' in m_path:
                resistance_m7 = np.genfromtxt(m_path, delimiter = ',')
                resistance_m7 = _safe_normalize(resistance_m7)
                resistance_m7 = torch.tensor(resize(resistance_m7, [512,512])).float().unsqueeze(0)
            elif 'resistance_m8' in m_path:
                resistance_m8 = np.genfromtxt(m_path, delimiter = ',')
                resistance_m8 = _safe_normalize(resistance_m8)
                resistance_m8 = torch.tensor(resize(resistance_m8, [512,512])).float().unsqueeze(0)
            elif 'resistance_m9' in m_path:
                resistance_m9 = np.genfromtxt(m_path, delimiter = ',')
                resistance_m9 = _safe_normalize(resistance_m9)
                resistance_m9 = torch.tensor(resize(resistance_m9, [512,512])).float().unsqueeze(0)
            elif 'via_m1m4' in m_path:
                via_m1m4 = np.genfromtxt(m_path, delimiter = ',')
                via_m1m4 = _safe_normalize(via_m1m4)
                via_m1m4 = torch.tensor(resize(via_m1m4, [512,512])).float().unsqueeze(0)
            elif 'via_m4m7' in m_path:
                via_m4m7 = np.genfromtxt(m_path, delimiter = ',')
                via_m4m7 = _safe_normalize(via_m4m7)
                via_m4m7 = torch.tensor(resize(via_m4m7, [512,512])).float().unsqueeze(0)
            elif 'via_m7m8' in m_path:
                via_m7m8 = np.genfromtxt(m_path, delimiter = ',')
                via_m7m8 = _safe_normalize(via_m7m8)
                via_m7m8 = torch.tensor(resize(via_m7m8, [512,512])).float().unsqueeze(0)
            elif 'via_m8m9' in m_path:
                via_m8m9 = np.genfromtxt(m_path, delimiter = ',')
                via_m8m9 = _safe_normalize(via_m8m9)
                via_m8m9 = torch.tensor(resize(via_m8m9, [512,512])).float().unsqueeze(0)
            # elif 'current_source' in m_path:
            #     current_source = np.genfromtxt(m_path, delimiter = ',')
            #     current_source = current_source/current_source.max()
            #     current_source = torch.tensor(resize(current_source, [512,512])).float().unsqueeze(0)
                    
        data = torch.concat([current, dist, pdn, resistance_m1, resistance_m4, resistance_m7, 
                             resistance_m8, resistance_m9, via_m1m4, via_m4m7, via_m7m8, via_m8m9, 
                             ir_drop], dim=0)

        return data    
    
    def __len__(self):
        return len(self.maps)
    
class load_real(data.Dataset):
    def __init__(self, folder_path, mode='train', testcase = []):
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
        folder_name = self.folder_list[index]
        folder_dir = os.path.join(self.folder_path, folder_name)

        current = np.genfromtxt(os.path.join(folder_dir,'current_map.csv'), delimiter = ',')
        base_shape = current.shape
        current = _safe_normalize(current)
        current = torch.tensor(resize(current, [512,512])).float().unsqueeze(0)

        dist = np.genfromtxt(os.path.join(folder_dir,'eff_dist_map.csv'), delimiter = ',')
        dist = _safe_normalize(dist)
        dist = torch.tensor(resize(dist, [512,512])).float().unsqueeze(0)

        pdn = np.genfromtxt(os.path.join(folder_dir,'pdn_density.csv'), delimiter = ',')
        pdn = _safe_normalize(pdn)
        pdn = torch.tensor(resize(pdn, [512,512])).float().unsqueeze(0)

        resistance_m1 = _load_csv_or_zeros(os.path.join(folder_dir,'resistance_m1.csv'), base_shape)
        resistance_m1 = torch.tensor(resize(resistance_m1, [512,512])).float().unsqueeze(0)

        resistance_m4 = _load_csv_or_zeros(os.path.join(folder_dir,'resistance_m4.csv'), base_shape)
        resistance_m4 = torch.tensor(resize(resistance_m4, [512,512])).float().unsqueeze(0)

        resistance_m7 = _load_csv_or_zeros(os.path.join(folder_dir,'resistance_m7.csv'), base_shape)
        resistance_m7 = torch.tensor(resize(resistance_m7, [512,512])).float().unsqueeze(0)

        resistance_m8 = _load_csv_or_zeros(os.path.join(folder_dir,'resistance_m8.csv'), base_shape)
        resistance_m8 = torch.tensor(resize(resistance_m8, [512,512])).float().unsqueeze(0)

        resistance_m9 = _load_csv_or_zeros(os.path.join(folder_dir,'resistance_m9.csv'), base_shape)
        resistance_m9 = torch.tensor(resize(resistance_m9, [512,512])).float().unsqueeze(0)

        via_m1m4 = _load_csv_or_zeros(os.path.join(folder_dir,'via_m1m4.csv'), base_shape)
        via_m1m4 = torch.tensor(resize(via_m1m4, [512,512])).float().unsqueeze(0)

        via_m4m7 = _load_csv_or_zeros(os.path.join(folder_dir,'via_m4m7.csv'), base_shape)
        via_m4m7 = torch.tensor(resize(via_m4m7, [512,512])).float().unsqueeze(0)

        via_m7m8 = _load_csv_or_zeros(os.path.join(folder_dir,'via_m7m8.csv'), base_shape)
        via_m7m8 = torch.tensor(resize(via_m7m8, [512,512])).float().unsqueeze(0)

        via_m8m9 = _load_csv_or_zeros(os.path.join(folder_dir,'via_m8m9.csv'), base_shape)
        via_m8m9 = torch.tensor(resize(via_m8m9, [512,512])).float().unsqueeze(0)

        ir_drop = np.genfromtxt(os.path.join(folder_dir,'ir_drop_map.csv'), delimiter = ',')
        ir_drop = torch.tensor(resize(ir_drop, [512,512])).float().unsqueeze(0)

        data = torch.concat([current, dist, pdn, resistance_m1, resistance_m4, resistance_m7,
                             resistance_m8, resistance_m9, via_m1m4, via_m4m7, via_m7m8, via_m8m9,
                             ir_drop], dim=0)

        return data


class load_real_original_size(data.Dataset):
    def __init__(self, folder_path, mode='train', testcase = [], print_name = False):
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
        folder_name = self.folder_list[index]
        folder_dir = os.path.join(self.folder_path, folder_name)

        current = np.genfromtxt(os.path.join(folder_dir,'current_map.csv'), delimiter = ',')
        shape = current.shape
        current = _safe_normalize(current)
        current = torch.tensor(current).float().unsqueeze(0)

        dist = np.genfromtxt(os.path.join(folder_dir,'eff_dist_map.csv'), delimiter = ',')
        dist = _safe_normalize(dist)
        dist = torch.tensor(dist).float().unsqueeze(0)

        pdn = np.genfromtxt(os.path.join(folder_dir,'pdn_density.csv'), delimiter = ',')
        pdn = _safe_normalize(pdn)
        pdn = torch.tensor(pdn).float().unsqueeze(0)

        resistance_m1 = _load_csv_or_zeros(os.path.join(folder_dir,'resistance_m1.csv'), shape)
        resistance_m1 = torch.tensor(resize(resistance_m1, shape)).float().unsqueeze(0)

        resistance_m4 = _load_csv_or_zeros(os.path.join(folder_dir,'resistance_m4.csv'), shape)
        resistance_m4 = torch.tensor(resize(resistance_m4, shape)).float().unsqueeze(0)

        resistance_m7 = _load_csv_or_zeros(os.path.join(folder_dir,'resistance_m7.csv'), shape)
        resistance_m7 = torch.tensor(resize(resistance_m7, shape)).float().unsqueeze(0)

        resistance_m8 = _load_csv_or_zeros(os.path.join(folder_dir,'resistance_m8.csv'), shape)
        resistance_m8 = torch.tensor(resize(resistance_m8, shape)).float().unsqueeze(0)

        resistance_m9 = _load_csv_or_zeros(os.path.join(folder_dir,'resistance_m9.csv'), shape)
        resistance_m9 = torch.tensor(resize(resistance_m9, shape)).float().unsqueeze(0)

        via_m1m4 = _load_csv_or_zeros(os.path.join(folder_dir,'via_m1m4.csv'), shape)
        via_m1m4 = torch.tensor(resize(via_m1m4, shape)).float().unsqueeze(0)

        via_m4m7 = _load_csv_or_zeros(os.path.join(folder_dir,'via_m4m7.csv'), shape)
        via_m4m7 = torch.tensor(resize(via_m4m7, shape)).float().unsqueeze(0)

        via_m7m8 = _load_csv_or_zeros(os.path.join(folder_dir,'via_m7m8.csv'), shape)
        via_m7m8 = torch.tensor(resize(via_m7m8, shape)).float().unsqueeze(0)

        via_m8m9 = _load_csv_or_zeros(os.path.join(folder_dir,'via_m8m9.csv'), shape)
        via_m8m9 = torch.tensor(resize(via_m8m9, shape)).float().unsqueeze(0)

        ir_drop = np.genfromtxt(os.path.join(folder_dir,'ir_drop_map.csv'), delimiter = ',')
        ir_drop = torch.tensor(ir_drop).float().unsqueeze(0)

        data = torch.concat([current, dist, pdn, resistance_m1, resistance_m4, resistance_m7,
                             resistance_m8, resistance_m9, via_m1m4, via_m4m7, via_m7m8, via_m8m9,
                             ir_drop], dim=0)

        return data
        
        