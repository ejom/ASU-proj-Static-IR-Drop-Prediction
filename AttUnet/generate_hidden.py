import numpy as np
import os
import re

def parse_node(node_str):
    """Parse 'n1_m1_4000_0' -> (layer, x, y)"""
    parts = node_str.split('_')
    layer = parts[1]  # e.g., 'm1'
    x = int(float(parts[2])) // 2000
    y = int(float(parts[3])) // 2000
    return layer, x, y

def generate_layer_csvs(spice_path, output_dir):
    with open(spice_path, 'r') as f:
        lines = f.readlines()

    # First pass: find grid size and collect data per layer
    metal_layers = {'m1', 'm4', 'm7', 'm8', 'm9'}
    same_layer = {}   # layer -> list of (resistance, coords1, coords2)
    cross_layer = {}  # (layer1, layer2) -> list of (resistance, coords1, coords2)

    max_x, max_y = 0, 0

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 4 or not parts[0].startswith('R'):
            continue

        resistance = float(parts[3])
        layer1, x1, y1 = parse_node(parts[1])
        layer2, x2, y2 = parse_node(parts[2])

        max_x = max(max_x, x1, x2)
        max_y = max(max_y, y1, y2)

        if layer1 == layer2:
            # Same layer -> resistance
            if layer1 not in same_layer:
                same_layer[layer1] = []
            same_layer[layer1].append((resistance, (x1, y1), (x2, y2)))
        else:
            # Cross layer -> via
            key = tuple(sorted([layer1, layer2]))
            if key not in cross_layer:
                cross_layer[key] = []
            cross_layer[key].append((resistance, (x1, y1), (x2, y2)))

    grid_shape = (max_x + 1, max_y + 1)

    # Generate resistance CSVs per metal layer
    for layer in metal_layers:
        grid = np.zeros(grid_shape)
        if layer in same_layer:
            for res, (x1, y1), (x2, y2) in same_layer[layer]:
                grid[x1, y1] += res / 2
                grid[x2, y2] += res / 2
        out_path = os.path.join(output_dir, f'resistance_{layer}.csv')
        np.savetxt(out_path, grid, delimiter=',')
        print(f'Saved {out_path} (shape: {grid.shape})')

    # Generate via CSVs
    via_map = {
        ('m1', 'm4'): 'via_m1m4',
        ('m4', 'm7'): 'via_m4m7',
        ('m7', 'm8'): 'via_m7m8',
        ('m8', 'm9'): 'via_m8m9',
    }
    for key, name in via_map.items():
        grid = np.zeros(grid_shape)
        if key in cross_layer:
            for res, (x1, y1), (x2, y2) in cross_layer[key]:
                grid[x1, y1] += res / 2
                grid[x2, y2] += res / 2
        out_path = os.path.join(output_dir, f'{name}.csv')
        np.savetxt(out_path, grid, delimiter=',')
        print(f'Saved {out_path} (shape: {grid.shape})')

# Run for all hidden test cases
hidden_dir = '../data/hidden-real-circuit-data/'
for tc in sorted(os.listdir(hidden_dir)):
    tc_path = os.path.join(hidden_dir, tc)
    sp_file = os.path.join(tc_path, 'netlist.sp')
    if os.path.exists(sp_file):
        print(f'\nProcessing {tc}...')
        generate_layer_csvs(sp_file, tc_path)