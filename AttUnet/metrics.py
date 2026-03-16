# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 18:49:01 2023

@author: Lizi Zhang
"""

import warnings
import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


def F1_Score(x, y):
    """Compute per-sample F1 score for hotspot detection (top 10% threshold)."""
    scores = []
    for i in range(x.shape[0]):
        pred = x[i, 0].copy()
        gt = y[i, 0].copy()

        pred_bin = (pred > 0.9 * pred.max()).astype(np.uint8)
        gt_bin = (gt > 0.9 * gt.max()).astype(np.uint8)

        scores.append(
            f1_score(gt_bin.flatten(), pred_bin.flatten(), average='binary', zero_division=0)
        )
    return scores


def compute_masked_mae(output, ir, quantile=0.95):
    """Compute MAE only on the highest IR drop regions."""
    threshold = torch.quantile(ir, quantile)
    high_value_mask = ir > threshold

    if high_value_mask.sum() == 0:
        return 0.0

    masked_output = output[high_value_mask]
    masked_ir = ir[high_value_mask]

    return torch.mean(torch.abs(masked_output - masked_ir)).item()
