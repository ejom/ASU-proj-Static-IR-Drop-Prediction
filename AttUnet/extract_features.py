"""
Extract per-layer resistance and via features from netlist.sp files.
Generates the missing resistance_m*.csv and via_m*m*.csv files for
test cases that only have current_map, eff_dist, pdn_density, ir_drop.

Usage: python extract_features.py

After running, re-run preprocess.py to regenerate the .npy files
with the new features included.
"""

import os
import re
import numpy as np


METAL_LAYERS = ['m1', 'm4', 'm7', 'm8', 'm9']
VIA_PAIRS = [('m1', 'm4'), ('m4', 'm7'), ('m7', 'm8'), ('m8', 'm9')]


def parse_netlist(sp_path):
    """Parse a SPICE netlist and extract per-layer resistance and via grids."""
    with open(sp_path, 'r') as f:
        lines = f.readlines()

    # First pass: find grid bounds per layer and globally
    # Node format: n1_m1_X_Y  where X,Y are coordinates
    coord_step = 2000  # coordinates are divided by this to get grid indices

    global_max_x = 0
    global_max_y = 0

    entries = []
    for line in lines:
        parts = line.split()
        if len(parts) < 4 or not parts[0].startswith('R'):
            continue

        resistance = float(parts[3])
        node1 = parts[1]
        node2 = parts[2]

        # Extract layer and coordinates from node names
        # Format: n1_m1_X_Y
        m1 = re.match(r'n\d+_(m\d+)_(\d+)_(\d+)', node1)
        m2 = re.match(r'n\d+_(m\d+)_(\d+)_(\d+)', node2)

        if not m1 or not m2:
            continue

        layer1 = m1.group(1)
        x1 = int(float(m1.group(2)) // coord_step)
        y1 = int(float(m1.group(3)) // coord_step)

        layer2 = m2.group(1)
        x2 = int(float(m2.group(2)) // coord_step)
        y2 = int(float(m2.group(3)) // coord_step)

        global_max_x = max(global_max_x, x1, x2)
        global_max_y = max(global_max_y, y1, y2)

        entries.append((layer1, x1, y1, layer2, x2, y2, resistance))

    grid_size = (global_max_x + 1, global_max_y + 1)

    # Initialize grids
    resistance_grids = {layer: np.zeros(grid_size) for layer in METAL_LAYERS}
    via_grids = {f'{a}{b}': np.zeros(grid_size) for a, b in VIA_PAIRS}

    # Second pass: populate grids
    for layer1, x1, y1, layer2, x2, y2, resistance in entries:
        if layer1 == layer2:
            # Same-layer resistor
            if layer1 in resistance_grids:
                resistance_grids[layer1][x1, y1] += resistance / 2
                resistance_grids[layer1][x2, y2] += resistance / 2
        else:
            # Via (inter-layer connection)
            # Normalize layer order
            pair_key = None
            for a, b in VIA_PAIRS:
                if (layer1 == a and layer2 == b) or (layer1 == b and layer2 == a):
                    pair_key = f'{a}{b}'
                    break
            if pair_key and pair_key in via_grids:
                via_grids[pair_key][x1, y1] += resistance / 2
                via_grids[pair_key][x2, y2] += resistance / 2

    return resistance_grids, via_grids


def process_directory(data_dir):
    """Extract features for all test cases in a directory."""
    folders = sorted([f for f in os.listdir(data_dir)
                      if os.path.isdir(os.path.join(data_dir, f))])

    for folder in folders:
        folder_path = os.path.join(data_dir, folder)
        sp_file = os.path.join(folder_path, 'netlist.sp')

        if not os.path.exists(sp_file):
            print(f'  Skipping {folder} — no netlist.sp')
            continue

        # Check if already has all per-layer resistance and via features
        has_features = all(
            os.path.exists(os.path.join(folder_path, f'resistance_{layer}.csv'))
            for layer in METAL_LAYERS
        ) and all(
            os.path.exists(os.path.join(folder_path, f'via_{a}{b}.csv'))
            for a, b in VIA_PAIRS
        )
        if has_features:
            print(f'  {folder} — already has features, skipping')
            continue

        print(f'  Extracting {folder}...')
        resistance_grids, via_grids = parse_netlist(sp_file)

        # Save resistance per layer
        for layer, grid in resistance_grids.items():
            out_path = os.path.join(folder_path, f'resistance_{layer}.csv')
            np.savetxt(out_path, grid, delimiter=',')

        # Save via per layer pair
        for pair_key, grid in via_grids.items():
            out_path = os.path.join(folder_path, f'via_{pair_key}.csv')
            np.savetxt(out_path, grid, delimiter=',')

        print(f'    Grid size: {resistance_grids["m1"].shape}, '
              f'resistors: m1={resistance_grids["m1"].sum():.0f} '
              f'm4={resistance_grids["m4"].sum():.0f} '
              f'm7={resistance_grids["m7"].sum():.0f} '
              f'm8={resistance_grids["m8"].sum():.0f} '
              f'm9={resistance_grids["m9"].sum():.0f}')


if __name__ == '__main__':
    print('Extracting features for hidden test data...')
    process_directory('../data/hidden-real-circuit-data')

    print('\nDone! Now re-run preprocess.py to regenerate .npy files.')
    print('  rm -rf ../data/hidden-npy ../data/hidden-npy-orig')
    print('  python preprocess.py')
