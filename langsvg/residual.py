"""Candidate regions from the reconstruction residual (L1 loss map)."""
from typing import List

import cv2
import numpy as np

from .masks import mask_iou


def residual_candidates(pred: np.ndarray, target: np.ndarray, n: int, iteration: int,
                        deleted: List[np.ndarray]) -> List[np.ndarray]:
    """Quantize the residual |pred - target|^p into quantile bins and return the connected
    components of each bin, ranked by ``area * mean residual`` (at most ``n``, skipping
    regions that duplicate previously deleted proposals).

    ``pred`` and ``target`` are (3, H, W) arrays in [0, 1]. The threshold below which the
    residual is ignored decays and the number of bins grows with ``iteration``.
    """
    diff = np.abs(pred - target)
    score_map = (diff ** 1.5).sum(0)
    residual = diff.copy().sum(0)
    bins = 24 + min(64, 12 * max(0, iteration - 1))
    threshold = max(0.15 - 0.02 * max(0, iteration - 1), 0.06)
    score_map[score_map < threshold] = 0
    edges = np.unique(np.quantile(score_map, np.linspace(0., 1., bins)))
    levels = np.clip(np.digitize(score_map, sorted(edges[1:-1]), right=False), 0, 255).astype(np.uint8)

    entries, components = [], []
    for level in np.unique(levels):
        if level == 0:
            continue
        _, component, stats, _ = cv2.connectedComponentsWithStats((levels == level).astype(np.uint8), connectivity=4)
        components.append(component)
        for index, stat in enumerate(stats[1:], start=1):
            area = int(stat[-1])
            if area <= 0:
                continue
            entries.append((len(components) - 1, index, area * float(residual[component == index].mean())))
    masks = []
    for level_index, index, _ in sorted(entries, key=lambda e: e[2], reverse=True):
        mask = (components[level_index] == index).astype(np.uint8) * 255
        if any(mask_iou(mask, d) > 0.8 for d in deleted):
            continue
        masks.append(mask)
        if n >= 0 and len(masks) >= n:
            break
    return masks
