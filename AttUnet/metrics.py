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
    """Compute per-sample F1 score for hotspot detection.

    Contest spec: hotspot = pixel with IR drop > 90% of that testcase's
    ground-truth maximum. The SAME threshold (from ground truth) is applied
    to both prediction and ground truth.
    """
    scores = []
    for i in range(x.shape[0]):
        pred = x[i, 0].copy()
        gt = y[i, 0].copy()

        threshold = 0.9 * gt.max()
        pred_bin = (pred > threshold).astype(np.uint8)
        gt_bin = (gt > threshold).astype(np.uint8)

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
