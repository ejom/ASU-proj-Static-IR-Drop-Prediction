"""
Run once to pre-process all datasets from CSV into .pt files.
Usage: python precache.py
"""
import os
import torch
from DataLoad_normalization import load_fake, load_real, load_real_original_size


DATASETS = {
    '../data/cache/fake': lambda: load_fake('../data/fake-circuit-data-plus/'),
    '../data/cache/real': lambda: load_real('../data/real-circuit-data-plus/', mode='train', testcase=[]),
    '../data/cache/test': lambda: load_real('../data/hidden-real-circuit-data/', mode='train', testcase=[]),
    '../data/cache/test_original': lambda: load_real_original_size('../data/hidden-real-circuit-data/', mode='train', testcase=[]),
}


def precache(cache_dir, dataset):
    os.makedirs(cache_dir, exist_ok=True)
    for i in range(len(dataset)):
        out_path = os.path.join(cache_dir, f'{i:04d}.pt')
        if os.path.exists(out_path):
            continue
        data = dataset[i]
        torch.save(data, out_path)
        print(f'  [{i+1}/{len(dataset)}] saved {out_path}')



if __name__ == '__main__':
    for cache_dir, make_dataset in DATASETS.items():
        print(f'Caching {cache_dir} ...')
        dataset = make_dataset()
        precache(cache_dir, dataset)
    print('Done.')
