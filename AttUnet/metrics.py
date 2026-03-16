# -*- coding: utf-8 -*-
"""
Metrics for evaluating IR drop prediction quality.
@author: Lizi Zhang

Two metrics are used:

1. F1 Score (hotspot detection):
   - Binarize both prediction and ground truth into "hotspot" vs "non-hotspot"
   - Hotspot = any pixel with IR drop > 90% of that sample's maximum
   - F1 = harmonic mean of precision and recall on these binary maps
   - F1 = 1.0 means perfect hotspot detection, 0.0 means no overlap

2. Masked MAE (optional):
   - Only compute MAE on the top 5% of pixels by IR drop value
   - Useful for checking accuracy specifically in high-drop regions
"""

import warnings
import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.exceptions import UndefinedMetricWarning

# Suppress warnings when F1 is undefined (e.g., no positive predictions)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


def F1_Score(x, y):
    """Compute per-sample F1 score for hotspot detection.

    The contest defines "hotspots" as pixels where IR drop exceeds 90% of
    that sample's maximum IR drop. This function binarizes both the prediction
    and ground truth using this threshold, then computes the F1 score
    (overlap between predicted and actual hotspot pixels).

    Args:
        x: predicted IR drop, numpy array of shape (B, 1, H, W)
        y: ground truth IR drop, numpy array of shape (B, 1, H, W)

    Returns:
        list of F1 scores, one per sample in the batch
    """
    scores = []
    for i in range(x.shape[0]):
        pred = x[i, 0].copy()  # (H, W) predicted IR drop map
        gt = y[i, 0].copy()    # (H, W) ground truth IR drop map

        # Binarize: 1 if pixel is in top 10% of values, 0 otherwise
        # Note: threshold is relative to each map's OWN maximum
        pred_bin = (pred > 0.9 * pred.max()).astype(np.uint8)
        gt_bin = (gt > 0.9 * gt.max()).astype(np.uint8)

        # Compute F1 between the two binary maps
        # Flatten 2D maps to 1D arrays for sklearn
        scores.append(
            f1_score(gt_bin.flatten(), pred_bin.flatten(), average='binary', zero_division=0)
        )
    return scores


def compute_masked_mae(output, ir, quantile=0.95):
    """Compute MAE only on the highest IR drop regions.

    Instead of averaging error across ALL pixels (most of which have low IR drop),
    this only looks at pixels in the top 5% (by default) of the ground truth.
    This tells you how accurate the model is specifically where IR drop is worst.

    Args:
        output: model prediction tensor
        ir: ground truth IR drop tensor
        quantile: fraction of pixels to exclude (0.95 = only top 5%)

    Returns:
        MAE value (float) computed only on high-drop pixels
    """
    # Find the IR drop value at the given quantile
    threshold = torch.quantile(ir, quantile)
    # Create boolean mask: True only for pixels above the threshold
    high_value_mask = ir > threshold

    if high_value_mask.sum() == 0:
        return 0.0

    # Select only the high-value pixels from both prediction and ground truth
    masked_output = output[high_value_mask]
    masked_ir = ir[high_value_mask]

    return torch.mean(torch.abs(masked_output - masked_ir)).item()
