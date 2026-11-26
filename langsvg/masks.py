"""Binary masks and their metadata.

Masks are ``uint8`` arrays with values in {0, 255}. Every mask that enters the SVG
tree is registered in a :class:`MaskRegistry`, which stores its semantic tag, its
source (e.g. ``grounded_sam``, ``semantic_sam``, ``sam``, ``l1_proposal``) and its
parent in the tree (``layer_parent_mask_id``).
"""
from typing import Dict, Iterable, List, Optional

import cv2
import numpy as np
from pycocotools import mask as mask_utils


class MaskRegistry(dict):
    """Metadata of every mask, keyed by ``id(mask)``.

    Parent links are stored as ``id`` of the parent mask. The registry keeps a
    reference to every registered mask so that ids are never recycled while the
    registry is alive.
    """

    def __init__(self):
        super().__init__()
        self._alive: Dict[int, np.ndarray] = {}

    def register(self, masks: Iterable[np.ndarray], tags: Iterable[str], sources: Iterable[str],
                 extra_metas: Optional[List[Optional[dict]]] = None) -> None:
        for idx, (mask, tag, source) in enumerate(zip(masks, tags, sources)):
            meta = {"tag": tag, "source": source}
            if extra_metas is not None and idx < len(extra_metas) and extra_metas[idx] is not None:
                meta.update(extra_metas[idx])
            self[id(mask)] = meta
            self._alive[id(mask)] = mask

    def meta(self, mask: np.ndarray) -> dict:
        return self.get(id(mask), {})


def is_background_meta(meta: dict) -> bool:
    return meta.get("tag") == "background" or meta.get("source") == "background"


def is_protected_semantic_meta(meta: Optional[dict]) -> bool:
    """True for masks that carry a real semantic label (open-vocab prior, dataset or VLM)."""
    meta = meta or {}
    tag = str(meta.get("tag", "unknown")).strip().lower()
    source = str(meta.get("source", "unknown")).strip().lower()
    level = str(meta.get("level", "")).strip().lower()
    if tag in {"", "unknown", "background"}:
        return False
    if source in {"", "unknown", "background"}:
        return False
    if level == "background":
        return False
    # Model placeholders such as "semantic_level_2" are not semantic labels.
    if tag.startswith("semantic_level_") or tag.startswith("unsamv2_gra_"):
        return False
    return True


# ----------------------------------------------------------------------------- basic ops
def area(mask: np.ndarray) -> int:
    return int(cv2.countNonZero(mask))


def sort_by_area(masks: List[np.ndarray]) -> List[np.ndarray]:
    """Largest first (ties keep the reverse of the input order, as in the original code)."""
    areas = [cv2.countNonZero(mask) for mask in masks]
    order = sorted(range(len(areas)), key=lambda k: areas[k])
    order.reverse()
    return [masks[i] for i in order]


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = np.sum(mask_a.astype(np.float32) + mask_b.astype(np.float32) == 510)
    total = cv2.countNonZero(mask_a + mask_b)
    return intersection / total if total > 0 else 0.0


def bbox_xyxy(mask: np.ndarray) -> Optional[List[int]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def bbox_area(box: Optional[List[int]]) -> int:
    if box is None:
        return 0
    x0, y0, x1, y1 = box
    return max(0, x1 - x0) * max(0, y1 - y0)


def encode_rle(mask: np.ndarray) -> dict:
    rle = mask_utils.encode(np.asfortranarray((mask > 0).astype(np.uint8)))
    return {"size": [int(rle["size"][0]), int(rle["size"][1])], "counts": rle["counts"].decode("utf-8")}


def fill_holes(mask: np.ndarray) -> np.ndarray:
    flood = mask.copy()
    h, w = mask.shape[:2]
    cv2.floodFill(flood, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 255)
    return cv2.bitwise_or(mask, cv2.bitwise_not(flood))


def preprocess(masks: List[np.ndarray], apply_morphology: bool = False, kernel_size: int = 5,
               min_component_area: int = 50) -> List[np.ndarray]:
    """Fill holes and drop tiny connected components (keep the largest if all are tiny)."""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    def morph(region):
        if not apply_morphology:
            return region
        return cv2.erode(cv2.dilate(region, kernel, iterations=1), kernel, iterations=1)

    out = []
    for mask in masks:
        filled = fill_holes(mask)
        num_labels, labels = cv2.connectedComponents(filled)
        if num_labels <= 1:
            out.append(morph(filled))
            continue
        merged = np.zeros_like(mask)
        kept, max_area, max_label = 0, 0, 1
        for label in range(1, num_labels):
            component_area = np.count_nonzero(labels == label)
            if component_area >= min_component_area:
                merged[labels == label] = 255
                kept += 1
            if component_area > max_area:
                max_area, max_label = component_area, label
        if kept == 0:
            merged[labels == max_label] = 255
        out.append(morph(merged))
    return out
