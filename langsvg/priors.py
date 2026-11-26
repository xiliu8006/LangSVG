"""Segmentation priors used to build the initial SVG tree.

* open-vocabulary objects: Grounded-SAM (RAM tags -> GroundingDINO boxes -> SAM masks), or
  dataset ground-truth object / part masks (``dataset_gt``, see :mod:`langsvg.dataset_gt`), or
  labeled masks loaded from a JSON file (e.g. a part segmenter);
* multi-granularity masks: Semantic-SAM (granularity levels) and SAM automatic masks.

All third-party models are vendored under ``langsvg/third_party`` and only need checkpoints.
"""
import json
import re
import types
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image

from .dataset_gt import DatasetGT
from .masks import MaskRegistry, preprocess, sort_by_area

THIRD_PARTY = Path(__file__).resolve().parent / "third_party"


def _full_mask(shape) -> np.ndarray:
    return np.full(shape[:2], 255, dtype=np.uint8)


# ----------------------------------------------------------------------------- SAM
def build_sam(checkpoint: str, model_type: str, device):
    from segment_anything import sam_model_registry
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    return sam.to(device=device)


def _patch_sam_device(generator) -> None:
    """Run SamAutomaticMaskGenerator's per-crop NMS on the boxes' device (GPU)."""
    from segment_anything import automatic_mask_generator as amg

    def _process_crop(self, image, crop_box, crop_layer_idx, orig_size):
        x0, y0, x1, y1 = crop_box
        cropped = image[y0:y1, x0:x1, :]
        cropped_size = cropped.shape[:2]
        self.predictor.set_image(cropped)
        points = self.point_grids[crop_layer_idx] * np.array(cropped_size)[None, ::-1]
        data = amg.MaskData()
        for (batch,) in amg.batch_iterator(self.points_per_batch, points):
            data.cat(self._process_batch(batch, cropped_size, crop_box, orig_size))
        self.predictor.reset_image()
        device = data["boxes"].device
        keep = amg.batched_nms(data["boxes"].float(), data["iou_preds"],
                               torch.zeros(len(data["boxes"]), dtype=torch.int64, device=device),
                               iou_threshold=self.box_nms_thresh)
        data.filter(keep)
        data["boxes"] = amg.uncrop_boxes_xyxy(data["boxes"], crop_box)
        data["points"] = amg.uncrop_points(data["points"], crop_box)
        data["crop_boxes"] = torch.tensor([crop_box for _ in range(len(data["rles"]))],
                                          dtype=data["boxes"].dtype, device=device)
        return data

    generator._process_crop = types.MethodType(_process_crop, generator)


class SAMAutomatic:
    """SAM automatic mask generation (unlabeled fine-grained masks)."""

    def __init__(self, sam, conf: dict):
        from segment_anything import SamAutomaticMaskGenerator
        self.generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=conf["points_per_side"],
            pred_iou_thresh=conf["pred_iou_thresh"],
            stability_score_thresh=conf["stability_score_thresh"],
            crop_n_layers=conf["crop_n_layers"],
            crop_n_points_downscale_factor=conf["crop_n_points_downscale_factor"],
            min_mask_region_area=conf["min_mask_region_area"],
            box_nms_thresh=conf["box_nms_thresh"],
        )
        _patch_sam_device(self.generator)

    def __call__(self, image: np.ndarray) -> List[np.ndarray]:
        return [np.where(ann["segmentation"], 255, 0).astype(np.uint8) for ann in self.generator.generate(image)]


# ----------------------------------------------------------------------------- Semantic-SAM
class SemanticSAM:
    """Semantic-SAM automatic masks at a given granularity level (1 = coarse ... 6 = fine)."""

    def __init__(self, checkpoint: str, device, model_type: str = "L"):
        from .third_party.semantic_sam.semantic_sam import SemanticSamAutomaticMaskGenerator, build_semantic_sam
        self.device = device
        self.model = build_semantic_sam(model_type=model_type, ckpt=checkpoint, device=device)
        self.generator_cls = SemanticSamAutomaticMaskGenerator

    def __call__(self, image: np.ndarray, level: int) -> List[np.ndarray]:
        h, w = image.shape[:2]
        resized = torchvision.transforms.Resize(640, interpolation=Image.BICUBIC)(Image.fromarray(image).convert("RGB"))
        tensor = torch.from_numpy(np.asarray(resized).copy()).permute(2, 0, 1).to(self.device)
        masks = []
        for ann in self.generator_cls(self.model, level=[level]).generate(tensor):
            mask = np.where(ann["segmentation"], 255, 0).astype(np.uint8)
            if mask.shape != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            masks.append(mask)
        return masks


# ----------------------------------------------------------------------------- Grounded-SAM
class GroundedSAM:
    """Open-vocabulary object masks: RAM image tags -> GroundingDINO boxes -> SAM box prompts."""

    def __init__(self, sam, groundingdino_ckpt: str, ram_ckpt: str, device, conf: dict):
        import torchvision.transforms as TS
        from segment_anything import SamPredictor
        from .third_party.groundingdino.datasets import transforms as GT
        from .third_party.groundingdino.models import build_model
        from .third_party.groundingdino.util.slconfig import SLConfig
        from .third_party.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
        from .third_party.ram import inference_ram
        from .third_party.ram.models import ram

        self.device, self.conf = device, conf
        args = SLConfig.fromfile(str(THIRD_PARTY / "groundingdino/config/GroundingDINO_SwinT_OGC.py"))
        args.device = str(device)
        model = build_model(args)
        model.load_state_dict(clean_state_dict(torch.load(groundingdino_ckpt, map_location="cpu")["model"]), strict=False)
        self.grounding = model.eval().to(device)
        self.grounding_transform = GT.Compose([GT.RandomResize([800], max_size=1333), GT.ToTensor(),
                                               GT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.get_phrases = get_phrases_from_posmap
        self.ram = ram(pretrained=ram_ckpt, image_size=384, vit="swin_l").eval().to(device)
        self.ram_transform = TS.Compose([TS.Resize((384, 384)), TS.ToTensor(),
                                         TS.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.inference_ram = inference_ram
        self.predictor = SamPredictor(sam)

    @torch.no_grad()
    def _boxes(self, image_pil: Image.Image, caption: str):
        caption = caption.lower().strip()
        if not caption.endswith("."):
            caption += "."
        image, _ = self.grounding_transform(image_pil, None)
        outputs = self.grounding(image[None].to(self.device), captions=[caption])
        logits = outputs["pred_logits"].cpu().sigmoid()[0]
        boxes = outputs["pred_boxes"].cpu()[0]
        keep = logits.max(dim=1)[0] > self.conf["box_threshold"]
        logits, boxes = logits[keep], boxes[keep]
        tokenizer = self.grounding.tokenizer
        tokenized = tokenizer(caption)
        phrases, scores, kept = [], [], []
        for idx, logit in enumerate(logits):
            phrase = self.get_phrases(logit > self.conf["text_threshold"], tokenized, tokenizer).strip()
            if not phrase:
                continue
            phrases.append(phrase)
            scores.append(logit.max().item())
            kept.append(idx)
        if not kept:
            return torch.empty((0, 4)), torch.empty((0,)), []
        return boxes[kept], torch.tensor(scores), phrases

    def __call__(self, image: np.ndarray) -> List[Tuple[np.ndarray, str, float]]:
        """Returns (mask, label, score) per detected object."""
        image_pil = Image.fromarray(image)
        ram_input = self.ram_transform(image_pil.resize((384, 384))).unsqueeze(0).to(self.device)
        tags = self.inference_ram(ram_input, self.ram)[0]
        if not tags or tags.lower() == "none":
            return []
        boxes, scores, phrases = self._boxes(image_pil, tags.replace(" |", ","))
        if boxes.numel() == 0:
            return []
        h, w = image.shape[:2]
        boxes = boxes * torch.tensor([w, h, w, h], dtype=boxes.dtype)
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        keep = torchvision.ops.nms(boxes, scores, self.conf["iou_threshold"]).cpu().tolist()
        boxes, phrases = boxes[keep], [phrases[i] for i in keep]
        self.predictor.set_image(image)
        boxes = self.predictor.transform.apply_boxes_torch(boxes, image.shape[:2]).to(self.device)
        masks, _, _ = self.predictor.predict_torch(point_coords=None, point_labels=None, boxes=boxes, multimask_output=False)
        return [(np.where(mask.squeeze(0).cpu().numpy(), 255, 0).astype(np.uint8), phrases[i], float(scores[keep[i]]))
                for i, mask in enumerate(masks)]


# ----------------------------------------------------------------------------- external masks
def load_external_masks(path: str, image_shape) -> List[dict]:
    """Labeled masks from JSON: a list of ``{"segmentation": RLE | polygons, "label": str,
    "level": "object" | "part", "id": any, "parent_id": any, "score": float}``.

    Parts can also be written as ``"<object>'s <part>"`` labels.
    """
    from pycocotools import mask as mask_utils
    h, w = image_shape[:2]
    entries = json.load(open(path, "r", encoding="utf-8"))
    if isinstance(entries, dict):
        entries = entries.get("annotations", entries.get("masks", []))
    out = []
    for entry in entries:
        seg = entry["segmentation"]
        if isinstance(seg, list):
            rle = mask_utils.merge(mask_utils.frPyObjects(seg, h, w))
        elif isinstance(seg.get("counts"), list):
            rle = mask_utils.frPyObjects(seg, h, w)
        else:
            rle = seg
        mask = np.where(mask_utils.decode(rle) > 0, 255, 0).astype(np.uint8)
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        if cv2.countNonZero(mask) > 0:
            out.append(dict(entry, mask=mask))
    return out


# ----------------------------------------------------------------------------- all priors
class MaskPriors:
    """Builds the prior models once and produces the ordered candidate masks for an image."""

    def __init__(self, conf: dict, checkpoints: dict, device):
        self.conf = conf
        sam = build_sam(checkpoints["sam"], conf["sam"].get("model_type", "vit_h"), device)
        self.sam = SAMAutomatic(sam, conf["sam"])
        self.semantic_sam = SemanticSAM(checkpoints["semantic_sam"], device)
        self.grounded_sam = None
        self.dataset_gt = None
        gt_conf = conf.get("dataset_gt") or {}
        if gt_conf.get("enabled", False):  # GT objects / parts replace Grounded-SAM
            if not gt_conf.get("instance_json"):
                raise ValueError("priors.dataset_gt.enabled needs priors.dataset_gt.instance_json")
            self.dataset_gt = DatasetGT(gt_conf["instance_json"], gt_conf.get("joint_json"), gt_conf.get("part_dir"))
        elif conf["grounded_sam"].get("enabled", True):
            self.grounded_sam = GroundedSAM(sam, checkpoints["groundingdino"], checkpoints["ram"], device, conf["grounded_sam"])

    def __call__(self, image: np.ndarray, registry: MaskRegistry, external_json: Optional[str] = None,
                 image_path: Optional[str] = None) -> List[np.ndarray]:
        """Returns ``[background] + labeled masks + Semantic-SAM masks (per level) + SAM masks``,
        each group sorted by area (largest first). All masks are registered in ``registry``.
        Labeled masks come from ``external_json`` if given, else from the dataset GT (``image_path`` is
        looked up by file stem), else from Grounded-SAM."""
        background = _full_mask(image.shape)
        registry.register([background], ["background"], ["background"])

        labeled = []
        if external_json is None and self.dataset_gt is not None:
            entries = self.dataset_gt.masks_for(image_path or "", image.shape)
            if not entries:
                print(f"[lang-svg] dataset_gt: no GT annotations for {image_path}")
            for entry in entries:
                meta = {"level": entry["level"], "annotation_id": entry["id"], "confidence": 1.0}
                if entry["parent_id"] is not None:
                    meta["parent_annotation_id"] = entry["parent_id"]
                registry.register([entry["mask"]], [entry["label"]], ["dataset_gt"], [meta])
                labeled.append(entry["mask"])
        elif external_json is not None:
            entries = load_external_masks(external_json, image.shape)
            for entry in entries:
                label = str(entry.get("label", entry.get("tag", "unknown")))
                level = entry.get("level") or ("part" if "'s " in label else "object")
                meta = {"level": level, "annotation_id": entry.get("id"), "confidence": entry.get("score")}
                if entry.get("parent_id") is not None:
                    meta["parent_annotation_id"] = entry["parent_id"]
                registry.register([entry["mask"]], [label], ["external_json"], [meta])
                labeled.append(entry["mask"])
        elif self.grounded_sam is not None:
            for mask, label, score in self.grounded_sam(image):
                registry.register([mask], [re.sub(r"\(\s*[\d.]+\s*\)$", "", label).strip()], ["grounded_sam"],
                                  [{"confidence": score}])
                labeled.append(mask)
        masks = [background] + sort_by_area(labeled)

        for level in self.conf["semantic_sam"]["levels"]:
            level_masks = preprocess(self.semantic_sam(image, level))
            tag = f"semantic_level_{level}"
            registry.register(level_masks, [tag] * len(level_masks), ["semantic_sam"] * len(level_masks))
            masks += sort_by_area(level_masks)

        sam_masks = preprocess(self.sam(image))
        registry.register(sam_masks, ["unknown"] * len(sam_masks), ["sam"] * len(sam_masks))
        masks += sort_by_area(sam_masks)
        return masks

    def l1_proposals(self, image: np.ndarray, count: int, level: int) -> List[np.ndarray]:
        """L1-SAM-Prompting refiner: the ``count`` largest fine-level Semantic-SAM masks of the image."""
        if count <= 0:
            return []
        masks = [m for m in preprocess(self.semantic_sam(image, level)) if cv2.countNonZero(m) > 0]
        return sort_by_area(masks)[:count] if masks else []
