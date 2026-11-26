"""Prompted object extraction: a point or a box -> the SVG tree of the object at that location.

Meant as a tool for agents (see :data:`TOOL_SPEC` / :func:`run_tool`) and ``scripts/extract_object.py``.
Full-image vectorization (:meth:`LangSVG.vectorize`) is untouched: this module only adds a subclass.

1. **Object mask** ``M``: SAM prompted with the point (the last of SAM's three multimask outputs,
   subpart -> part -> whole object) or the box (single output).
2. **Object tree**: the image is cropped to ``M``'s box (plus a margin). Semantic-SAM (automatic masks at
   ``semantic_sam_levels``) and SAM prompted with a ``sam_grid x sam_grid`` point grid over ``M`` (grid points
   inside ``M`` only) give the sub-object masks; every mask is clipped to ``M`` and kept if
   ``>= min_inside_ratio`` of it lies inside ``M``. ``M`` is the
   root (``layers[0]``, no square background), the rest is inserted with the full-image layout logic
   (:func:`tree.insert_masks`, :func:`tree.refine_by_occlusion`).
3. **Region-only optimization**: the canvas is the crop; the reconstruction loss covers ``M`` only and
   pushes the alpha outside ``M`` to 0 (:func:`optimize._masked_loss`). Refinement (L1-SAM-Prompting, then
   direct L1 masks) only looks at the residual inside ``M``. There is no point budget: refinement stops
   when no candidate mask has ``>= min_mask_pixels`` pixels, or when the SVG has ``max_paths`` paths.
"""
import copy
import json
import os
import xml.etree.ElementTree as ET
from typing import List, Optional, Sequence, Union

import cv2
import numpy as np
import pydiffvg
import torch
from PIL import Image

from . import tree as T
from .budget import Budget
from .export import annotate_svg, build_scene_graph, save_json
from .masks import MaskRegistry, bbox_xyxy, encode_rle, preprocess, sort_by_area
from .optimize import optimize_structure, optimize_visual
from .pipeline import LangSVG, seed_everything
from .priors import MaskPriors, _patch_sam_device
from .residual import residual_candidates
from .svg import (clamp_colors, cut_holes_open, dominant_color_fitting, init_svg_from_layers, mask_to_path, render,
                  to_uint8_image, total_points)

DEFAULTS = {
    "max_paths": 64,             # refinement stops once the SVG has this many paths
    "min_mask_pixels": 20,       # refinement only: smallest candidate mask (pixels) for a new path; none left -> stop
    "sam_grid": 10,              # SAM point prompts: sam_grid x sam_grid grid over the object (inside points only)
    "semantic_sam_levels": None,  # None: priors.semantic_sam.levels
    "min_inside_ratio": 0.5,     # a prior mask is kept if this share of it lies inside the object mask
    "crop_margin": 0.1,          # context around the object box, relative to its longer side
    "max_iterations": 30,        # safety cap of the refinement loop
}


def _paste_rle(rle: dict, crop_xyxy, width: int, height: int) -> dict:
    from pycocotools import mask as mask_utils
    x0, y0, x1, y1 = crop_xyxy
    full = np.zeros((height, width), dtype=np.uint8)
    full[y0:y1, x0:x1] = mask_utils.decode({"size": rle["size"], "counts": rle["counts"].encode("utf-8")})
    return encode_rle(full)


def _shift_node(node: dict, region: int, crop_xyxy, width: int, height: int) -> dict:
    """A node of a region's tree.json in original image coordinates, ids prefixed with ``r<region>_``."""
    node = dict(node)
    x0, y0 = crop_xyxy[:2]
    node["region"] = region
    node["node_id"] = node["svg_element_id"] = f"r{region}_{node['node_id']}"
    if node.get("parent_id") is not None:
        node["parent_id"] = f"r{region}_{node['parent_id']}"
    node["children_ids"] = [f"r{region}_{c}" for c in node.get("children_ids", [])]
    for key in ("bbox_xyxy", "visible_bbox_xyxy", "source_bbox_xyxy"):
        if node.get(key) is not None:
            node[key] = [node[key][0] + x0, node[key][1] + y0, node[key][2] + x0, node[key][3] + y0]
    for key in ("segmentation", "visible_segmentation", "source_segmentation"):
        if node.get(key) is not None:
            node[key] = _paste_rle(node[key], crop_xyxy, width, height)
    return node


class ObjectExtractor(LangSVG):
    """Point / box prompt -> SVG of one object.

    Example::

        cfg = load_config("configs/natural.yaml")
        tool = ObjectExtractor(cfg, device="cuda")
        result = tool.extract("examples/ADE_val_00000003.jpg", "outputs/obj", point=(320, 240))
    """

    def __init__(self, cfg: dict, device="cuda", priors: Optional[MaskPriors] = None):
        cfg = copy.deepcopy(cfg)
        if priors is None:  # the object prior is the prompt: no Grounded-SAM / dataset GT models needed
            cfg["priors"]["grounded_sam"] = dict(cfg["priors"].get("grounded_sam") or {}, enabled=False)
            cfg["priors"]["dataset_gt"] = dict(cfg["priors"].get("dataset_gt") or {}, enabled=False)
        self.object_conf = dict(DEFAULTS, **(cfg.get("object_tool") or {}))
        super().__init__(cfg, device=device, priors=priors)
        self._region = None

    # ================================================================== prompts -> masks
    def prompt_mask(self, image: np.ndarray, points: Optional[Sequence] = None, box: Optional[Sequence] = None):
        """SAM mask of the prompt: one point -> last multimask output (whole object); several points
        and/or a box -> single output. Returns (uint8 mask, SAM score)."""
        if points is None and box is None:
            raise ValueError("give a point or a box")
        predictor = self.priors.sam.generator.predictor
        coords = labels = None
        if points is not None:
            coords = np.asarray(points, dtype=np.float32).reshape(-1, 2)
            labels = np.ones(len(coords), dtype=np.int64)
        box_arr = None if box is None else np.asarray(box, dtype=np.float32).reshape(4)
        multimask = box_arr is None and len(coords) == 1
        predictor.set_image(image)
        masks, scores, _ = predictor.predict(point_coords=coords, point_labels=labels, box=box_arr,
                                             multimask_output=multimask)
        predictor.reset_image()
        index = len(masks) - 1 if multimask else 0
        mask = np.where(masks[index], 255, 0).astype(np.uint8)
        return mask, float(scores[index])

    def _grid_masks(self, crop: np.ndarray, region: np.ndarray) -> List[np.ndarray]:
        """SAM masks of a ``sam_grid x sam_grid`` point grid over the object (points inside it only)."""
        from segment_anything import SamAutomaticMaskGenerator
        n = int(self.object_conf["sam_grid"])
        box = bbox_xyxy(region)
        if n <= 0 or box is None:
            return []
        x0, y0, x1, y1 = box
        xs = x0 + (np.arange(n) + 0.5) * (x1 - x0) / n
        ys = y0 + (np.arange(n) + 0.5) * (y1 - y0) / n
        points = np.array([(x, y) for y in ys for x in xs])
        inside = region[np.clip(points[:, 1].astype(int), 0, region.shape[0] - 1),
                        np.clip(points[:, 0].astype(int), 0, region.shape[1] - 1)] > 0
        points = points[inside]
        if len(points) == 0:
            return []
        h, w = crop.shape[:2]
        sam_conf = self.cfg["priors"]["sam"]
        generator = SamAutomaticMaskGenerator(
            model=self.priors.sam.generator.predictor.model, points_per_side=None,
            point_grids=[points / np.array([w, h], dtype=np.float64)],
            pred_iou_thresh=sam_conf["pred_iou_thresh"], stability_score_thresh=sam_conf["stability_score_thresh"],
            box_nms_thresh=sam_conf["box_nms_thresh"], min_mask_region_area=sam_conf["min_mask_region_area"],
            crop_n_layers=0)
        _patch_sam_device(generator)
        return [np.where(ann["segmentation"], 255, 0).astype(np.uint8) for ann in generator.generate(crop)]

    def _clip(self, masks: List[np.ndarray], region: np.ndarray, min_pixels: int = 1) -> List[np.ndarray]:
        """Masks clipped to the object; kept if mostly (``min_inside_ratio``) inside it and not empty
        (refinement proposals: ``>= min_pixels`` pixels)."""
        ratio = float(self.object_conf["min_inside_ratio"])
        out = []
        for mask in masks:
            area = cv2.countNonZero(mask)
            clipped = cv2.bitwise_and(mask, region)
            inside = cv2.countNonZero(clipped)
            if area > 0 and inside >= min_pixels and inside >= ratio * area:
                out.append(clipped)
        return out

    # ================================================================== public API
    def extract(self, image: Union[str, np.ndarray], output_dir: Optional[str] = None,
                point: Optional[Sequence[float]] = None, points: Optional[Sequence] = None,
                box: Optional[Sequence[float]] = None, label: Optional[str] = None) -> dict:
        """Extract the object at ``point`` (x, y) / ``points`` [(x, y), ...] / ``box`` (x0, y0, x1, y1), in
        pixels of the original image. Writes ``object.svg`` (crop coordinates), ``object_fullframe.svg``
        (original image coordinates, transparent outside the object), ``object.png`` (RGBA crop),
        ``mask.png`` and ``tree.json`` to ``output_dir`` (if given). Returns a summary dict."""
        cfg, oc = copy.deepcopy(self.cfg), self.object_conf
        self.run_cfg, self.budget = cfg, Budget(cfg)  # the budget only sets path simplification
        self.reg, self.history, self._l1_cache = MaskRegistry(), [], None
        if cfg.get("seed") is not None:
            seed_everything(int(cfg["seed"]))
        image_path = image if isinstance(image, str) else "<array>"
        full = np.array(Image.open(image).convert("RGB")) if isinstance(image, str) else np.ascontiguousarray(image[:, :, :3])
        H, W = full.shape[:2]
        if point is not None:
            points = [point]

        # 1) object mask and crop
        object_mask, score = self.prompt_mask(full, points, box)
        obox = bbox_xyxy(object_mask)
        if obox is None:
            raise ValueError(f"SAM returned an empty mask for point={points} box={box}")
        margin = int(round(float(oc["crop_margin"]) * max(obox[2] - obox[0], obox[3] - obox[1])))
        x0, y0 = max(0, obox[0] - margin), max(0, obox[1] - margin)
        x1, y1 = min(W, obox[2] + margin), min(H, obox[3] + margin)
        crop = np.ascontiguousarray(full[y0:y1, x0:x1])
        region = np.ascontiguousarray(object_mask[y0:y1, x0:x1])
        self.image, self._region = crop, region
        h, w = crop.shape[:2]

        # 2) object-level tree: root = object mask, Semantic-SAM levels + SAM grid prompts inside it
        root = region.copy()
        self.reg.register([root], [label or "object"], ["prompt"],
                          [{"level": "object", "confidence": score, "annotation_id": "prompt_object"}])
        rest = []
        levels = oc["semantic_sam_levels"] or cfg["priors"]["semantic_sam"]["levels"]
        for level in levels:
            level_masks = self._clip(preprocess(self.priors.semantic_sam(crop, int(level))), region)
            self.reg.register(level_masks, [f"semantic_level_{level}"] * len(level_masks), ["semantic_sam"] * len(level_masks))
            rest += sort_by_area(level_masks)
        sam_masks = self._clip(preprocess(self._grid_masks(crop, region)), region)
        self.reg.register(sam_masks, ["unknown"] * len(sam_masks), ["sam"] * len(sam_masks),
                          [{"proposal_source": "sam_point_grid"} for _ in sam_masks])
        rest += sort_by_area(sam_masks)
        layers = T.insert_masks([[root]], rest, self.reg)
        layers = T.refine_by_occlusion(layers, self.reg, float(cfg["tree"]["occlusion_threshold"]))
        layers = T.rebuild_layers_from_tree(layers, self.reg)

        # 3) initialization + structure / color fitting, region-only loss
        opt = cfg["optimization"]
        loss_mask = torch.tensor(region > 0, dtype=torch.float32, device=self.device)
        shapes, groups = init_svg_from_layers(layers, crop, self.budget.epsilon(), self.budget.path_conf)
        shapes[0] = mask_to_path(cut_holes_open(root), self.budget.epsilon(), self.budget.path_conf(0, 0))
        groups, _ = dominant_color_fitting(groups, crop, layers, True, opt["kmeans_k"])
        shapes, groups = optimize_structure(shapes, groups, crop, layers, opt["struct_iters"], opt["lr"], self.device,
                                            cfg["opacity_loss"], loss_mask=loss_mask)
        groups, quantized = dominant_color_fitting(groups, crop, layers, True, opt["kmeans_k"])
        shapes, groups = optimize_visual(shapes, groups, crop, opt["visual_iters"], opt["lr"], self.device,
                                         point_scales=[0.0] * len(shapes), color_scales=[1.0] * len(shapes),
                                         loss_mask=loss_mask)
        shapes, groups, layers, _ = self._tree_prune(shapes, groups, layers)
        init_paths = len(shapes)

        # 4) refinement inside the object until no mask >= min_mask_pixels or max_paths paths
        shapes, groups, layers, stop = self._refine_object(shapes, groups, layers, crop, quantized, loss_mask)
        shapes, groups, _, layers = self._reorder_to_tree(shapes, groups, None, layers)

        # 5) export
        clamp_colors(groups)
        scene = build_scene_graph(layers, self.reg, shapes, groups, w, h, self.device, image_path, cfg["path"], self.history)
        with torch.no_grad():
            rgba = render(w, h, shapes, groups, self.device).detach().cpu().numpy()
        inside = region > 0
        composite = rgba[:, :, :3] * rgba[:, :, 3:4] + (1 - rgba[:, :, 3:4])
        mse = float(np.mean((composite[inside] - crop[inside].astype(np.float32) / 255.0) ** 2))
        scene["psnr_region"] = float("inf") if mse <= 1e-12 else float(10 * np.log10(1.0 / mse))
        scene["object_prompt"] = {"points": None if points is None else [list(map(float, p)) for p in points],
                                  "box": None if box is None else [float(v) for v in box], "label": label or "object",
                                  "sam_score": score, "crop_xyxy": [x0, y0, x1, y1], "image_size": [W, H],
                                  "object_mask_pixels": int(cv2.countNonZero(region)), "init_paths": init_paths,
                                  "stop_reason": stop}
        result = {"paths": len(shapes), "points": total_points(shapes), "psnr_region": scene["psnr_region"],
                  "crop_xyxy": [x0, y0, x1, y1], "init_paths": init_paths, "stop_reason": stop, "sam_score": score}
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            svg_path = os.path.join(output_dir, "object.svg")
            pydiffvg.save_svg(svg_path, w, h, shapes, groups)
            annotate_svg(svg_path, scene)
            shifted = copy.deepcopy(shapes)
            for shape in shifted:
                shape.points = shape.points.detach() + torch.tensor([float(x0), float(y0)])
            full_path = os.path.join(output_dir, "object_fullframe.svg")
            pydiffvg.save_svg(full_path, W, H, shifted, groups)
            annotate_svg(full_path, scene)
            Image.fromarray(np.clip(rgba * 255, 0, 255).astype(np.uint8), "RGBA").save(os.path.join(output_dir, "object.png"))
            Image.fromarray(object_mask).save(os.path.join(output_dir, "mask.png"))
            save_json(os.path.join(output_dir, "tree.json"), scene)
            result.update(svg=svg_path, fullframe_svg=full_path, png=os.path.join(output_dir, "object.png"),
                          mask=os.path.join(output_dir, "mask.png"), tree_json=os.path.join(output_dir, "tree.json"))
        print(f"[lang-svg object] {os.path.basename(image_path)}: crop={result['crop_xyxy']} init_paths={init_paths} "
              f"paths={result['paths']} points={result['points']} psnr_region={result['psnr_region']:.4f} stop={stop}")
        self._region = None
        return result

    def extract_regions(self, image: str, output_dir: str, prompts: Sequence[dict], label: Optional[str] = None) -> dict:
        """One object made of several regions (e.g. the connected components of an occluded object): one
        prompt (``{"point": (x, y)}`` / ``{"points": [...]}`` / ``{"box": (x0, y0, x1, y1)}``) per region, one
        tree build per region (``<output_dir>/region_<k>/``), joined under a virtual root without a path.
        Writes ``object_fullframe.svg`` (``<g data-node-id="root">`` holding one ``<g>`` per region) and
        ``tree.json`` (virtual root + the region nodes in original image coordinates)."""
        if not prompts:
            raise ValueError("no prompts")
        regions = []
        for k, prompt in enumerate(prompts):
            result = self.extract(image, os.path.join(output_dir, f"region_{k}"), label=label, **prompt)
            regions.append(dict(result, region=k, prompt={key: list(map(float, np.ravel(v))) for key, v in prompt.items()}))
        width, height = json.load(open(regions[0]["tree_json"], encoding="utf-8"))["object_prompt"]["image_size"]

        # SVG: virtual root group > region groups > paths (ids prefixed with the region)
        ns = "http://www.w3.org/2000/svg"
        ET.register_namespace("", ns)
        svg = ET.Element(f"{{{ns}}}svg", {"version": "1.1", "width": str(width), "height": str(height),
                                         "viewBox": f"0 0 {width} {height}"})
        root_group = ET.SubElement(svg, f"{{{ns}}}g", {"id": "root", "data-node-id": "root",
                                                        "data-tag": label or "object", "data-level": "object"})
        nodes, root_children = [], []
        for region in regions:
            k = region["region"]
            group = ET.SubElement(root_group, f"{{{ns}}}g", {"id": f"region_{k}", "data-region": str(k)})
            for element in ET.parse(region["fullframe_svg"]).getroot().iter():
                if element.tag.split("}")[-1] not in {"path", "circle", "ellipse", "rect", "polygon"}:
                    continue
                element = copy.deepcopy(element)
                for key in ("id", "data-node-id", "data-parent-id"):
                    if element.get(key) is not None:
                        element.set(key, f"r{k}_{element.get(key)}")
                if element.get("data-parent-id") is None:
                    element.set("data-parent-id", "root")
                group.append(element)
            scene = json.load(open(region["tree_json"], encoding="utf-8"))
            for node in scene["nodes"]:
                node = _shift_node(node, k, region["crop_xyxy"], width, height)
                if node["parent_id"] is None:
                    node["parent_id"] = "root"
                    root_children.append(node["node_id"])
                nodes.append(node)
        full_svg = os.path.join(output_dir, "object_fullframe.svg")
        ET.ElementTree(svg).write(full_svg, encoding="unicode", xml_declaration=True)
        virtual_root = {"node_id": "root", "tag": label or "object", "level": "object", "virtual": True,
                        "parent_id": None, "children_ids": root_children}
        save_json(os.path.join(output_dir, "tree.json"), {
            "schema_version": "langsvg_object_regions_v1", "image_path": str(image), "image_size": [width, height],
            "roots": ["root"], "nodes": [virtual_root] + nodes,
            "regions": [{k: r[k] for k in ("region", "prompt", "crop_xyxy", "paths", "points", "psnr_region",
                                           "init_paths", "stop_reason", "sam_score", "tree_json", "mask")} for r in regions]})
        return {"regions": regions, "paths": sum(r["paths"] for r in regions), "points": sum(r["points"] for r in regions),
                "fullframe_svg": full_svg, "tree_json": os.path.join(output_dir, "tree.json")}

    # ================================================================== refinement
    def _refine_object(self, shapes, groups, layers, crop, quantized, loss_mask):
        """L1-SAM-Prompting iterations, then direct L1 iterations; no point budget.
        Stops when no candidate mask has >= min_mask_pixels pixels, the SVG reaches max_paths paths, or two
        direct iterations in a row add no net path (new paths pruned / merged away again)."""
        rcfg, opt, oc = self.run_cfg["refinement"], self.run_cfg["optimization"], self.object_conf
        max_paths, min_pixels = int(oc["max_paths"]), int(oc["min_mask_pixels"])
        prompt_iters = max(0, int(rcfg["iterations"]) - int(rcfg["direct_iterations"]))
        per_iter = max(1, int(max_paths / max(float(rcfg["max_new_paths_divisor"]), 1.0)))
        level = int(rcfg["l1_semantic_sam_level"])
        self._l1_cache = self._clip(self.priors.l1_proposals(crop, 10 ** 9, level), self._region, min_pixels)
        self._l1_cache = sort_by_area(self._l1_cache)
        old_ids = {id(m) for m in T.flatten(layers)}
        flags, deleted, stop, stalls = [], [], "max_iterations", 0
        for i in range(int(oc["max_iterations"])):
            if len(shapes) >= max_paths:
                stop = "max_paths"
                break
            mode = "prompt" if i < prompt_iters else "direct"
            n_new = min(per_iter, max_paths - len(shapes))
            record = {"iteration": i + 1, "mode": mode, "max_new_paths": n_new, "paths_before": len(shapes)}
            if mode == "prompt":
                added, shapes, groups, layers, flags, deleted = self._add_prompt_paths(
                    shapes, groups, layers, crop, quantized, flags, n_new, i + 1, deleted, allow_merge=True)
            else:
                added, shapes, groups, layers, flags, deleted = self._add_direct_paths(
                    shapes, groups, layers, quantized, flags, n_new, i + 1, deleted, final=False)
            if not added:
                record["stopped"] = f"no_mask_>={min_pixels}px"
                self.history.append(record)
                if mode == "prompt":
                    continue
                stop = record["stopped"]
                break
            if mode == "prompt":
                shapes, groups = optimize_structure(shapes, groups, crop, layers, opt["struct_iters"], opt["lr"], self.device,
                                                    self.run_cfg["opacity_loss"], new_flags=flags, loss_mask=loss_mask)
            else:
                trainable = [0 if id(m) in old_ids else 1 for m in T.flatten(layers)]
                old_point, old_color = float(opt["old_point_lr_scale"]), float(opt["old_color_lr_scale"])
                shapes, groups = optimize_visual(shapes, groups, crop, opt["visual_iters"], opt["lr"], self.device,
                                                 point_scales=[1.0 if f == 1 else old_point for f in trainable],
                                                 color_scales=[1.0 if f == 1 else old_color for f in trainable],
                                                 loss_mask=loss_mask)
            shapes, groups, layers, flags = self._tree_prune(shapes, groups, layers, flags)
            if mode == "prompt":
                shapes, groups, layers, flags = self._remove_invisible(shapes, groups, layers, flags)
            record["paths_after"] = len(shapes)
            self.history.append(record)
            # direct masks that tree pruning / merging removes again: stop after 2 iterations without net growth
            stalls = stalls + 1 if mode == "direct" and len(shapes) <= record["paths_before"] else 0
            if stalls >= 2:
                stop = "no_progress"
                break
        else:
            stop = "max_iterations"
        if len(shapes) >= max_paths:
            stop = "max_paths"
        return shapes, groups, layers, stop

    def _candidates(self, shapes, groups, layers, quantized, n, iteration, deleted, color_fit: bool):
        """Residual regions inside the object with >= min_mask_pixels pixels (premultiplied render vs.
        the masked target; the residual outside the object is zero)."""
        if self._region is None:
            return super()._candidates(shapes, groups, layers, quantized, n, iteration, deleted, color_fit)
        if n <= 0:
            return []
        h, w = quantized.shape[:2]
        render_groups = copy.deepcopy(groups)
        if color_fit and layers:
            render_groups, _ = dominant_color_fitting(render_groups, quantized, layers, False, self.run_cfg["optimization"]["kmeans_k"],
                                                      self.run_cfg["refinement"]["max_color_change"])
        with torch.no_grad():
            rgba = render(w, h, shapes, render_groups, self.device).detach().cpu().numpy()
        inside = (self._region > 0)[None]
        pred = np.transpose(rgba[:, :, :3] * rgba[:, :, 3:4], (2, 0, 1))
        target = np.transpose(quantized.astype(np.float32) / 255, (2, 0, 1)) * inside
        pred = np.where(inside, pred, target)
        min_pixels = int(self.object_conf["min_mask_pixels"])
        masks = [m for m in residual_candidates(pred, target, -1, iteration, deleted)
                 if cv2.countNonZero(cv2.bitwise_and(m, self._region)) >= min_pixels]
        return [cv2.bitwise_and(m, self._region) for m in masks[:n]]


# ====================================================================== agent tool
TOOL_SPEC = {
    "name": "extract_object_svg",
    "description": ("Vectorize one object of an image into a layered SVG. The object is selected by a click point "
                    "(x, y) or a bounding box (x0, y0, x1, y1) in image pixels; SAM segments it, Semantic-SAM / SAM "
                    "split it into parts, and only the object region is fitted (transparent elsewhere). Returns the "
                    "paths of the SVG files (object crop and full-frame overlay), the tree JSON and fit quality."),
    "input_schema": {
        "type": "object",
        "properties": {
            "image_path": {"type": "string", "description": "path of the input image"},
            "point": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2,
                      "description": "click point [x, y] on the object"},
            "box": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4,
                    "description": "bounding box [x0, y0, x1, y1] around the object"},
            "label": {"type": "string", "description": "optional name of the object (tag of the root node)"},
            "output_dir": {"type": "string", "description": "directory for object.svg, object_fullframe.svg, tree.json"},
        },
        "required": ["image_path", "output_dir"],
    },
}


def run_tool(extractor: ObjectExtractor, arguments: dict) -> dict:
    """Execute :data:`TOOL_SPEC` with an already loaded extractor (models stay in memory between calls)."""
    if arguments.get("point") is None and arguments.get("box") is None:
        return {"error": "give a point or a box"}
    try:
        return extractor.extract(arguments["image_path"], arguments["output_dir"], point=arguments.get("point"),
                                 box=arguments.get("box"), label=arguments.get("label"))
    except ValueError as error:
        return {"error": str(error)}
