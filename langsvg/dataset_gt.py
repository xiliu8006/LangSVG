"""Ground-truth object / part masks of COCO-style part datasets (ADE20K-Part234, PascalPart116 layout).

Used instead of Grounded-SAM when ``priors.dataset_gt.enabled`` is true:

- objects: the non-crowd annotations of ``instance_json`` (category names without ``'s``);
- parts: for each object with a ``part_category_id`` list, the pixels of the part label PNG
  (``part_dir/<image stem>.png``) that carry that part id *and* lie inside the object mask;
  part ids index the part categories of the joint json (its categories after the object ones).

Every GT mask is a labeled node (``tag`` = category name, e.g. ``"car"`` / ``"car's wheel"``), so it
is never pruned, merged or dropped by the budget. The part labels are exported as the scene's
``dataset_vlm_ontology`` (the candidate labels of closed-set tagging).
"""
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image

IGNORE_PART_ID = 65535


def _decode(segmentation, height: int, width: int) -> np.ndarray:
    from pycocotools import mask as mask_utils
    if isinstance(segmentation, list):
        mask = mask_utils.decode(mask_utils.merge(mask_utils.frPyObjects(segmentation, height, width)))
    elif isinstance(segmentation.get("counts"), list):
        mask = mask_utils.decode(mask_utils.frPyObjects(segmentation, height, width))
    else:
        mask = mask_utils.decode(segmentation)
    if mask.ndim == 3:
        mask = np.any(mask, axis=2)
    return np.where(mask, 255, 0).astype(np.uint8)


def _resize(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    if mask.shape[:2] == (height, width):
        return mask
    return cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)


def part_annotation_id(object_id, part_category_id: int) -> str:
    """Annotation id of the part ``part_category_id`` of object annotation ``object_id``."""
    return f"{object_id}:part{int(part_category_id)}"


class DatasetGT:
    """GT masks of one COCO-style instance json (+ joint json for part names, + part label PNGs)."""

    def __init__(self, instance_json: str, joint_json: Optional[str] = None, part_dir: Optional[str] = None):
        instance_json = Path(instance_json).expanduser().resolve()
        payload = json.load(open(instance_json, "r", encoding="utf-8"))
        categories = payload.get("categories", [])
        self.object_names: Dict[int, str] = {int(c["id"]): str(c.get("name", c["id"])) for c in categories}
        joint = Path(joint_json) if joint_json else instance_json.with_name(instance_json.name.replace("_instance_", "_joint_"))
        self.part_names: Dict[int, str] = {}
        if joint.exists():
            joint_categories = json.load(open(joint, "r", encoding="utf-8")).get("categories", [])
            for part_id, category in enumerate(joint_categories[len(categories):]):
                self.part_names[part_id] = str(category.get("name", part_id))
        self.part_dir = Path(part_dir) if part_dir else None
        self.json_dir = instance_json.parent

        self.images: Dict[str, dict] = {}
        for info in payload.get("images", []):
            name = Path(str(info["file_name"]))
            self.images[name.stem] = info
        self.annotations = defaultdict(list)
        for ann in payload.get("annotations", []):
            self.annotations[str(ann["image_id"])].append(ann)
        self.ontology = {
            "dataset_json_path": str(instance_json),
            "object_labels": sorted({n for i, n in self.object_names.items() if n}),
            "part_labels": sorted({n for n in self.part_names.values() if n}),
        }

    def _part_png(self, image_path: str) -> Optional[Path]:
        stem = Path(image_path).stem + ".png"
        candidates = []
        if self.part_dir is not None:
            candidates.append(self.part_dir / stem)
        split = Path(image_path).parent.name  # e.g. images/validation/<stem>.jpg
        candidates += [self.json_dir / "annotations_detectron2_part" / split / stem,
                       self.json_dir / "annotations_detectron2_part" / stem]
        return next((p for p in candidates if p.exists()), None)

    def masks_for(self, image_path: str, image_shape) -> List[dict]:
        """GT entries ``{"mask", "label", "level", "id", "parent_id"}`` of one image (objects by area,
        each followed by its parts). Empty if the image is not in the annotation file."""
        h, w = image_shape[:2]
        info = self.images.get(Path(image_path).stem)
        if info is None:
            return []
        src_h, src_w = int(info.get("height", h)), int(info.get("width", w))
        annotations = [a for a in self.annotations.get(str(info["id"]), []) if int(a.get("iscrowd", 0)) != 1]
        annotations.sort(key=lambda a: float(a.get("area", 0.0)), reverse=True)
        part_map = None
        png = self._part_png(image_path)
        if png is not None and any("part_category_id" in a for a in annotations):
            part_map = _resize(np.array(Image.open(png)), h, w)

        objects, parts, direct_parts = [], [], []
        for ann in annotations:
            if not ann.get("segmentation"):
                continue
            mask = _resize(_decode(ann["segmentation"], src_h, src_w), h, w)
            if cv2.countNonZero(mask) == 0:
                continue
            label = self.object_names.get(int(ann.get("category_id", -1)), f"category_{ann.get('category_id')}")
            if "'s" in label:  # a part annotated directly in the instance json
                direct_parts.append({"mask": mask, "label": label, "level": "part", "id": ann.get("id"), "parent_id": None})
                continue
            objects.append({"mask": mask, "label": label, "level": "object", "id": ann.get("id"), "parent_id": None})
            if part_map is None:
                continue
            inside = mask > 0
            for part_id in ann.get("part_category_id", []):
                part_id = int(part_id)
                if part_id == IGNORE_PART_ID:
                    continue
                part = np.where((part_map == part_id) & inside, 255, 0).astype(np.uint8)
                if cv2.countNonZero(part) == 0:
                    continue
                parts.append({"mask": part, "label": self.part_names.get(part_id, f"part_{part_id}"), "level": "part",
                              "id": part_annotation_id(ann.get("id"), part_id), "parent_id": ann.get("id")})
        for entry in direct_parts:  # parent: the object covering >= 50% of the part
            area = max(1, cv2.countNonZero(entry["mask"]))
            best, best_overlap = None, 0
            for obj in objects:
                overlap = int(np.count_nonzero((entry["mask"] > 0) & (obj["mask"] > 0)))
                if overlap > best_overlap:
                    best, best_overlap = obj, overlap
            if best is not None and best_overlap / area >= 0.5:
                entry["parent_id"] = best["id"]
        return objects + parts + direct_parts
