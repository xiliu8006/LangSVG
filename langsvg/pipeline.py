"""Lang-SVG vectorization pipeline (Algorithms 1-2 of the paper)."""
import copy
import os
import random
from typing import Optional

import cv2
import numpy as np
import pydiffvg
import torch
from PIL import Image

from . import tree as T
from .budget import Budget
from .export import annotate_svg, build_scene_graph, save_json
from .masks import MaskRegistry, is_background_meta, is_protected_semantic_meta, mask_iou, sort_by_area
from .optimize import optimize_structure, optimize_visual
from .priors import MaskPriors
from .residual import residual_candidates
from .svg import (clamp_colors, cut_holes_open, dominant_color_fitting, init_svg_from_layers, mask_to_path, masks_to_paths, mean_color, new_group,
                  point_count, psnr, reindex, render_rgb, to_uint8_image, total_points)


def never_drop(meta) -> bool:
    """The background and every labeled (open-vocabulary / dataset / VLM) mask are never removed."""
    meta = meta or {}
    return is_protected_semantic_meta(meta) or is_background_meta(meta) or str(meta.get("level", "")) == "background"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cv2.setRNGSeed(seed)


class LangSVG:
    """Image -> semantic SVG tree.

    Example::

        cfg = load_config("configs/natural.yaml")
        model = LangSVG(cfg, device="cuda")
        result = model.vectorize("examples/cat.jpg", "outputs/cat")
    """

    def __init__(self, cfg: dict, device="cuda", priors: Optional[MaskPriors] = None):
        self.cfg = cfg
        self.device = torch.device(device)
        pydiffvg.set_device(self.device)
        pydiffvg.set_use_gpu(self.device.type == "cuda")
        pydiffvg.set_print_timing(False)
        self.priors = priors or MaskPriors(cfg["priors"], cfg["checkpoints"], self.device)

    # ================================================================== public API
    def vectorize(self, image_path: str, output_dir: str, external_masks: Optional[str] = None,
                  point_budget: Optional[int] = None) -> dict:
        """Vectorize one image. Writes ``final.svg`` (with tree metadata), ``final.png``,
        ``tree.json`` (scene graph consumed by the tagger) and ``init.svg`` to ``output_dir``."""
        cfg = copy.deepcopy(self.cfg)
        if point_budget is not None:
            cfg["point_budget"], cfg["path_budget"] = int(point_budget), None
        self.run_cfg = cfg
        self.budget = Budget(cfg)
        self.reg = MaskRegistry()
        self.history = []
        self._l1_cache = None
        if cfg.get("seed") is not None:
            seed_everything(int(cfg["seed"]))
        os.makedirs(output_dir, exist_ok=True)

        image = np.array(Image.open(image_path).convert("RGB"))
        self.image = image
        h, w = image.shape[:2]

        # 1) segmentation priors -> SVG tree (Sec. 3.1)
        masks = self.priors(image, self.reg, external_json=external_masks, image_path=image_path)
        layers, rest = T.init_tree_with_semantic_masks([[masks[0]]], masks[1:], self.reg)
        layers = T.insert_masks(layers, rest, self.reg)
        layers = T.refine_by_occlusion(layers, self.reg, float(cfg["tree"]["occlusion_threshold"]))
        layers = T.rebuild_layers_from_tree(layers, self.reg)
        layers = T.rebuild_layers_from_tree(self._trim_to_init_budget(layers), self.reg)

        # 2) tree-guided initialization with depth-adaptive simplification, then structure fitting
        opt = cfg["optimization"]
        shapes, groups = init_svg_from_layers(layers, image, self.budget.epsilon(), self.budget.path_conf)
        groups, _ = dominant_color_fitting(groups, image, layers, True, opt["kmeans_k"])
        shapes, groups = optimize_structure(shapes, groups, image, layers, opt["struct_iters"], opt["lr"],
                                            self.device, cfg["opacity_loss"])
        groups, quantized = dominant_color_fitting(groups, image, layers, True, opt["kmeans_k"])
        shapes, groups = optimize_visual(shapes, groups, image, opt["visual_iters"], opt["lr"], self.device,
                                         point_scales=[0.0] * len(shapes), color_scales=[1.0] * len(shapes))
        shapes, groups, layers, _ = self._tree_prune(shapes, groups, layers)
        self._export_init_tree(shapes, groups, layers, image, image_path, output_dir)

        # 3) L1-SAM-Prompting + residual refinement under the point budget
        shapes, groups, layers, flags = self._refine(shapes, groups, layers, image, quantized)
        shapes, groups, flags, layers = self._reorder_to_tree(shapes, groups, flags, layers)

        # 4) export
        scene = build_scene_graph(layers, self.reg, shapes, groups, w, h, self.device, image_path,
                                  cfg["path"], self.history)
        self._add_ontology(scene)
        svg_path = os.path.join(output_dir, "final.svg")
        clamp_colors(groups)
        pydiffvg.save_svg(svg_path, w, h, shapes, groups)
        annotate_svg(svg_path, scene)
        with torch.no_grad():
            rendered = render_rgb(w, h, shapes, groups, self.device)
        Image.fromarray(to_uint8_image(rendered)).save(os.path.join(output_dir, "final.png"))
        scene["psnr"] = psnr(image, rendered.permute(1, 2, 0).detach().cpu().numpy())
        save_json(os.path.join(output_dir, "tree.json"), scene)
        print(f"[lang-svg] {os.path.basename(image_path)}: paths={len(shapes)} points={total_points(shapes)} "
              f"psnr={scene['psnr']:.4f}")
        return scene

    def _add_ontology(self, scene: dict) -> None:
        """With dataset GT masks: the dataset's object / part labels (closed-set tagging candidates)."""
        gt = getattr(self.priors, "dataset_gt", None)
        if gt is not None:
            scene["dataset_vlm_ontology"] = dict(gt.ontology)

    # ================================================================== initial tree export
    def _export_init_tree(self, shapes, groups, layers, image, image_path, output_dir):
        """The initial tree (before L1-SAM-Prompting / visual refinement) as ``init.svg`` with the same
        node attributes as ``final.svg``, its rendering ``init.png`` and scene graph ``init_tree.json``."""
        h, w = image.shape[:2]
        svg_path = os.path.join(output_dir, "init.svg")
        pydiffvg.save_svg(svg_path, w, h, shapes, groups)
        scene = build_scene_graph(layers, self.reg, shapes, groups, w, h, self.device, image_path,
                                  self.run_cfg["path"], [])
        self._add_ontology(scene)
        annotate_svg(svg_path, scene)
        with torch.no_grad():
            rendered = render_rgb(w, h, shapes, groups, self.device)
        Image.fromarray(to_uint8_image(rendered)).save(os.path.join(output_dir, "init.png"))
        scene["psnr"] = psnr(image, rendered.permute(1, 2, 0).detach().cpu().numpy())
        save_json(os.path.join(output_dir, "init_tree.json"), scene)
        print(f"[lang-svg] initial tree: paths={len(shapes)} points={total_points(shapes)} psnr={scene['psnr']:.4f}")

    # ================================================================== initial tree
    def _trim_to_init_budget(self, layers: list) -> list:
        """Drop the smallest unlabeled masks until the initial tree fits ``init_budget_ratio`` of the budget.
        Unlabeled masks whose path alone exceeds 10% of the point budget are dropped first.
        The background and labeled masks are never dropped (:func:`never_drop`), even over the limit."""
        limit = self.budget.init_limit()
        single_limit = max(1, int(self.budget.max_points * 0.10))
        entries, total = [], 0
        index = [(li, mi, mask) for li, layer in enumerate(layers) for mi, mask in enumerate(layer)]
        try:
            costs = [point_count(p) for p in masks_to_paths([m for _, _, m in index], self.budget.epsilon(),
                                                            [self.budget.path_conf(li, mi) for li, mi, _ in index])]
        except Exception:  # some mask failed: per-mask costs, failures exceed the limit
            costs = []
            for li, mi, mask in index:
                try:
                    costs.append(point_count(mask_to_path(mask, self.budget.epsilon(), self.budget.path_conf(li, mi))))
                except Exception:
                    costs.append(limit + 1)
        for (li, mi, mask), cost in zip(index, costs):
            pixels = cv2.countNonZero(np.where(mask > 0, 255, 0).astype(np.uint8))
            cost = max(1, cost)
            entries.append((pixels, li, mi, cost, never_drop(self.reg.meta(mask))))
            total += cost
        if not entries:
            return []
        selected = {(li, mi) for _, li, mi, _, _ in entries}
        for _, li, mi, cost, protected in entries:
            if cost > single_limit and not protected:
                selected.discard((li, mi))
        current = sum(cost for _, li, mi, cost, _ in entries if (li, mi) in selected)
        if current > limit and total > limit:
            for _, li, mi, cost, protected in sorted(entries, key=lambda e: (e[0], e[1], e[2])):
                if current <= limit:
                    break
                if (li, mi) not in selected or protected:
                    continue
                if len(selected) <= 1:
                    break
                selected.discard((li, mi))
                current -= cost
        elif current > limit:  # total <= limit can only happen without oversized masks
            return layers
        return [kept for kept in ([m for mi, m in enumerate(layer) if (li, mi) in selected]
                                  for li, layer in enumerate(layers)) if kept]

    # ================================================================== refinement loop
    def _refine(self, shapes, groups, layers, image, quantized):
        rcfg, opt = self.run_cfg["refinement"], self.run_cfg["optimization"]
        total_iters = int(rcfg["iterations"])
        direct_iters = max(0, min(int(rcfg["direct_iterations"]), total_iters))
        prompt_iters = total_iters - direct_iters
        used = total_points(shapes)
        max_points = self.budget.max_points
        prompt_phase_max = used + int(max(0, max_points - used) * float(rcfg["prompt_phase_budget_ratio"]))
        max_new = max(1, int(self.budget.path_count / max(float(rcfg["max_new_paths_divisor"]), 1.0)))
        old_ids = {id(m) for m in T.flatten(layers)}  # initial tree: points frozen during direct refinement
        flags, deleted = [], []

        for i in range(total_iters):
            final = i == total_iters - 1
            used = total_points(shapes)
            if i < prompt_iters:
                mode, remaining = "prompt", int((prompt_phase_max - used) * float(rcfg["nonfinal_ratio"]))
            elif final:
                mode, remaining = "direct", max_points - used
            else:
                mode, remaining = "direct", int((max_points - used) * float(rcfg["nonfinal_ratio"]))
            n_new = self.budget.candidates_for(remaining)
            if not final:
                n_new = min(max_new, n_new)
            record = {"iteration": i + 1, "mode": mode, "final": final, "max_new_paths": n_new, "paths_before": len(shapes)}

            if mode == "prompt":
                added, shapes, groups, layers, flags, deleted = self._add_prompt_paths(
                    shapes, groups, layers, image, quantized, flags, n_new, i + 1, deleted, allow_merge=not final)
            else:
                added, shapes, groups, layers, flags, deleted = self._add_direct_paths(
                    shapes, groups, layers, quantized, flags, n_new, i + 1, deleted, final)
            shapes, groups, layers, flags, removed = self._enforce_budget(shapes, groups, layers, flags)
            if not added or (removed > 0 and sum(1 for f in flags if f == 1) == 0):
                record["stopped"] = "no_new_paths"
                self.history.append(record)
                if mode == "prompt" and direct_iters > 0:
                    continue
                break

            if mode == "prompt":  # incremental structure fitting of the new proposals
                shapes, groups = optimize_structure(shapes, groups, image, layers, opt["struct_iters"], opt["lr"],
                                                    self.device, self.run_cfg["opacity_loss"], new_flags=flags)
            else:
                trainable = [0 if id(m) in old_ids else 1 for m in T.flatten(layers)]
                if len(trainable) != len(shapes):
                    trainable = flags
                old_point, old_color = float(opt["old_point_lr_scale"]), float(opt["old_color_lr_scale"])
                shapes, groups = optimize_visual(shapes, groups, image, opt["visual_iters"], opt["lr"], self.device,
                                                 point_scales=[1.0 if f == 1 else old_point for f in trainable],
                                                 color_scales=[1.0 if f == 1 else old_color for f in trainable])
            record["paths_after_optimize"] = len(shapes)
            if not final:
                shapes, groups, layers, flags = self._tree_prune(shapes, groups, layers, flags)
            if final:
                self.history.append(record)
                break

            if mode == "prompt":
                shapes, groups, layers, flags = self._remove_invisible(shapes, groups, layers, flags)
            record["paths_after"] = len(shapes)
            self.history.append(record)
        return shapes, groups, layers, flags

    # ------------------------------------------------------------------ adding paths
    def _new_paths(self, shapes, groups, flags, layers_old, layers_new, color_image, conf_fn):
        """Rebuild shape lists in the order of ``layers_new``; new masks get freshly initialized paths."""
        old = {id(m): (s, g) for m, s, g in zip(T.flatten(layers_old), shapes, groups)}
        new_shapes, new_groups, new_flags, fresh = [], [], [], []
        for li, layer in enumerate(layers_new):
            for mask in layer:
                if id(mask) in old:
                    s, g = old[id(mask)]
                    new_shapes.append(s), new_groups.append(g), new_flags.append(0)
                else:
                    fresh.append((len(new_shapes), li, mask))
                    new_shapes.append(None)
                    new_groups.append(new_group(mean_color(color_image, mask)))
                    new_flags.append(1)
        fitted = masks_to_paths([cut_holes_open(m) for _, _, m in fresh], self.budget.epsilon(),
                                [conf_fn(li) for _, li, _ in fresh])
        for (pos, _, _), path in zip(fresh, fitted):
            new_shapes[pos] = path
        reindex(new_groups)
        return new_shapes, new_groups, new_flags

    def _candidates(self, shapes, groups, layers, quantized, n, iteration, deleted, color_fit: bool):
        """Connected regions of the quantized reconstruction residual, ranked by area x residual."""
        h, w = quantized.shape[:2]
        render_groups = copy.deepcopy(groups)
        if color_fit and layers:
            render_groups, _ = dominant_color_fitting(render_groups, quantized, layers, False, self.run_cfg["optimization"]["kmeans_k"],
                                                      self.run_cfg["refinement"]["max_color_change"])
        with torch.no_grad():
            rendered = render_rgb(w, h, shapes, render_groups, self.device).detach().cpu().numpy()
        target = np.transpose((quantized / 255).astype(np.float16), (2, 0, 1))
        return residual_candidates(rendered, target, n, iteration, deleted)

    def _add_prompt_paths(self, shapes, groups, layers, image, quantized, flags, n, iteration, deleted, allow_merge):
        """L1-SAM-Prompting: residual regions decide how many new structural masks are added; the masks
        themselves come from the fine-granularity Semantic-SAM prior."""
        if len(flags) == 0:
            flags = [0] * len(shapes)
        candidates = self._candidates(shapes, groups, layers, quantized, n, iteration, deleted, color_fit=True)
        if not candidates:
            return False, shapes, groups, layers, flags, deleted
        level = int(self.run_cfg["refinement"]["l1_semantic_sam_level"])
        if self._l1_cache is None:  # the image never changes, so the fine-level masks are computed once
            self._l1_cache = self.priors.l1_proposals(image, 10 ** 9, level)
        proposals = [m.copy() for m in self._l1_cache[:len(candidates)]]
        if not proposals:
            return False, shapes, groups, layers, flags, deleted
        proposals = sort_by_area(proposals)
        self.reg.register(proposals, ["unknown"] * len(proposals), ["sam_proposal"] * len(proposals),
                          [{"proposal_source": f"semantic_sam_level_{level}", "proposal_refiner": "semantic_sam",
                            "candidate_source": "l1_residual"} for _ in proposals])
        new_layers, newly_deleted = T.insert_proposals([l.copy() for l in layers], proposals.copy(), self.reg)
        deleted = deleted + newly_deleted
        new_layers = T.rebuild_layers_from_tree(new_layers, self.reg)
        old_ids = {id(m) for m in T.flatten(layers)}
        if not any(id(m) not in old_ids for m in T.flatten(new_layers)):
            return False, shapes, groups, layers, flags, deleted
        shapes, groups, flags = self._new_paths(shapes, groups, flags, layers, new_layers, quantized,
                                                lambda li: self.budget.path_conf(li, refine=True))
        layers = new_layers
        if allow_merge:
            shapes, groups, layers, flags = self._local_merge(shapes, groups, layers, flags, quantized)
        return True, shapes, groups, layers, flags, deleted

    def _add_direct_paths(self, shapes, groups, layers, quantized, flags, n, iteration, deleted, final):
        """Residual regions are inserted directly as new nodes (fine appearance details)."""
        if len(flags) == 0:
            flags = [0] * len(shapes)
        pool = max(n, min(n * 4, n + 128)) if (final and n > 0) else n
        candidates = sort_by_area(self._candidates(shapes, groups, layers, quantized, pool, iteration, deleted,
                                                   color_fit=False))
        old_ids = {id(m) for m in T.flatten(layers)}
        new_layers, added = [l.copy() for l in layers], 0
        for candidate in candidates:
            if added >= n:
                break
            if any(mask_iou(candidate, d) > 0.8 for d in deleted):
                continue
            self.reg.register([candidate], ["unknown"], ["l1_proposal"],
                              [{"proposal_source": "l1_residual", "proposal_refiner": "none"}])
            before = sum(1 for m in T.flatten(new_layers) if id(m) not in old_ids)
            trial, newly_deleted = T.insert_proposals([l.copy() for l in new_layers], [candidate], self.reg)
            deleted = deleted + newly_deleted
            if sum(1 for m in T.flatten(trial) if id(m) not in old_ids) > before:
                new_layers, added = trial, added + 1
        if added == 0:
            return False, shapes, groups, layers, flags, deleted
        new_layers = T.rebuild_layers_from_tree(new_layers, self.reg)
        conf_fn = lambda li: self.budget.path_conf(li, force_fallback=True, refine=True)
        shapes, groups, flags = self._new_paths(shapes, groups, flags, layers, new_layers, quantized, conf_fn)
        layers = new_layers
        if not final:
            shapes, groups, layers, flags = self._local_merge(shapes, groups, layers, flags, quantized,
                                                              force_fallback=True)
        return True, shapes, groups, layers, flags, deleted

    # ------------------------------------------------------------------ tree-guided pruning & merging
    def _tree_prune(self, shapes, groups, layers, flags=None):
        """Remove unlabeled children whose fill color is close to their parent's (mean |dRGB| < threshold)."""
        pcfg = self.run_cfg["tree_prune"]
        flat = T.flatten(layers)
        if not pcfg.get("enabled", True) or len(flat) != len(shapes):
            return shapes, groups, layers, flags
        threshold = float(pcfg["color_threshold"])
        reference = int(pcfg["reference_path_count"])
        if reference > 0 and self.budget.path_count > 0 and float(pcfg["threshold_scale_per_path_halving"]) > 0:
            threshold *= float(pcfg["threshold_scale_per_path_halving"]) ** float(np.log2(reference / self.budget.path_count))
        min_depth = int(pcfg["min_layer_init_index"])
        layer_of = {id(m): li for li, layer in enumerate(layers) for m in layer}
        index_of = {id(m): i for i, m in enumerate(flat)}

        prune = set()
        for child, mask in enumerate(flat):
            parent = index_of.get(self.reg.meta(mask).get("layer_parent_mask_id"))
            if parent is None or never_drop(self.reg.meta(mask)):
                continue
            if layer_of.get(id(mask), 0) + 1 < min_depth:
                continue
            c, p = groups[child].fill_color, groups[parent].fill_color
            if c is None or p is None or c.numel() < 3 or p.numel() < 3:
                continue
            distance = float(np.mean(np.abs(c[:3].detach().float().cpu().numpy() - p[:3].detach().float().cpu().numpy())))
            if distance < threshold:
                prune.add(child)
        if not prune:
            return shapes, groups, layers, flags

        removed_parent = {}
        for idx in prune:
            meta = self.reg.meta(flat[idx])
            replacement = meta.get("layer_parent_mask_id")
            removed_parent[id(flat[idx])] = replacement
            if meta.get("annotation_id") is not None:
                removed_parent[meta["annotation_id"]] = replacement
        keep = [i for i in range(len(flat)) if i not in prune]
        layers = [[m for m in layer if index_of[id(m)] not in prune] for layer in layers]
        layers = [layer for layer in layers if layer]
        T.reconnect_children(layers, self.reg, removed_parent)
        shapes, groups = [shapes[i] for i in keep], [groups[i] for i in keep]
        reindex(groups)
        if flags is not None and len(flags) == len(flat):
            flags = [flags[i] for i in keep]
        shapes, groups, flags, layers = self._reorder_to_tree(shapes, groups, flags, layers)
        return shapes, groups, layers, flags

    def _local_merge(self, shapes, groups, layers, flags, color_image, force_fallback=False):
        """Merge same-depth siblings (same parent and tag, at least one new) with similar color that overlap."""
        mcfg = self.run_cfg["local_merge"]
        if not mcfg.get("enabled", False):
            return shapes, groups, layers, flags
        color_t, overlap_t = float(mcfg["color_threshold"]), int(mcfg["overlap_pixels_threshold"])
        entries, k = [], 0
        for li, layer in enumerate(layers):
            for mask in layer:
                entries.append({"layer": li, "mask": mask, "shape": shapes[k], "group": groups[k],
                                "new": bool(flags[k]) if k < len(flags) else False, "meta": dict(self.reg.meta(mask))})
                k += 1

        def parent_ref(meta):
            return meta.get("layer_parent_mask_id", meta.get("parent_id", meta.get("parent_annotation_id")))

        def can_merge(a, b):
            if not (a["new"] or b["new"]) or a["layer"] != b["layer"]:
                return False
            if is_protected_semantic_meta(a["meta"]) or is_protected_semantic_meta(b["meta"]):
                return False
            if str(a["meta"].get("tag", "unknown")) != str(b["meta"].get("tag", "unknown")):
                return False
            if str(a["meta"].get("level", "unknown")) != str(b["meta"].get("level", "unknown")):
                return False
            if parent_ref(a["meta"]) != parent_ref(b["meta"]):
                return False
            if int(np.count_nonzero((a["mask"] > 0) & (b["mask"] > 0))) < overlap_t:
                return False
            ca, cb = a["group"].fill_color[:3].detach().cpu(), b["group"].fill_color[:3].detach().cpu()
            return float(torch.sum(torch.abs(ca - cb)).item()) <= color_t

        merged_pairs, changed = 0, True
        while changed:
            changed = False
            for i in range(len(entries)):
                j = next((j for j in range(i + 1, len(entries)) if can_merge(entries[i], entries[j])), None)
                if j is None:
                    continue
                a, b = entries[i], entries[j]
                mask = np.where((a["mask"] > 0) | (b["mask"] > 0), 255, 0).astype(np.uint8)
                conf = self.budget.path_conf(a["layer"], None, force_fallback=force_fallback, refine=True)
                meta = self._merged_meta(a["meta"], b["meta"])
                entries[i] = {"layer": a["layer"], "mask": mask, "group": new_group(mean_color(color_image, mask)),
                              "shape": mask_to_path(cut_holes_open(mask), self.budget.epsilon(), conf),
                              "new": a["new"] or b["new"], "meta": meta}
                entries.pop(j)
                merged_pairs, changed = merged_pairs + 1, True
                break
        if merged_pairs == 0:
            return shapes, groups, layers, flags

        shapes, groups, flags, layers, metas = [], [], [], [], []
        for li in sorted({e["layer"] for e in entries}):
            layer_entries = [e for e in entries if e["layer"] == li]
            layers.append([e["mask"] for e in layer_entries])
            for e in layer_entries:
                shapes.append(e["shape"]), groups.append(e["group"]), flags.append(1 if e["new"] else 0), metas.append(e["meta"])
        reindex(groups)
        flat = T.flatten(layers)
        self.reg.register(flat, [str(m.get("tag", "unknown")) for m in metas],
                          [str(m.get("source", "local_merge")) for m in metas], metas)
        shapes, groups, flags, layers = self._reorder_to_tree(shapes, groups, flags, layers)
        return shapes, groups, layers, flags

    @staticmethod
    def _merged_meta(a: dict, b: dict) -> dict:
        sources = {str(a.get("source", "unknown")), str(b.get("source", "unknown"))}
        merged = dict(a)
        if "l1_proposal" in sources:
            merged["source"] = "l1_proposal"
        elif "sam_proposal" in sources:
            merged["source"] = "sam_proposal"
        else:
            merged.setdefault("source", "local_merge")
        merged["merged_from_sources"] = [str(a.get("source", "unknown")), str(b.get("source", "unknown"))]
        merged["merged_from_tags"] = [str(a.get("tag", "unknown")), str(b.get("tag", "unknown"))]
        merged["merged_from_levels"] = [str(a.get("level", "unknown")), str(b.get("level", "unknown"))]
        parent = a.get("layer_parent_mask_id", a.get("parent_id", a.get("parent_annotation_id")))
        if parent is not None:
            merged["layer_parent_mask_id"] = parent
        for key in ("proposal_source", "proposal_refiner", "candidate_source"):
            if key in a or key in b:
                merged[key] = a.get(key, b.get(key))
        return merged

    # ------------------------------------------------------------------ budget & cleanup
    def _enforce_budget(self, shapes, groups, layers, flags):
        """Drop the smallest new (unlabeled) paths until the point budget is met."""
        current, limit = total_points(shapes), self.budget.max_points
        flat = T.flatten(layers)
        if current <= limit or len(flat) != len(shapes) or len(flags) != len(shapes):
            return shapes, groups, layers, flags, 0
        removable = sorted(((int(cv2.countNonZero(m)), i, point_count(shapes[i])) for i, m in enumerate(flat)
                            if flags[i] == 1 and not never_drop(self.reg.meta(m))),
                           key=lambda x: (x[0], x[1]))
        remove = set()
        for _, i, cost in removable:
            if current <= limit:
                break
            remove.add(i)
            current -= cost
        if not remove:
            return shapes, groups, layers, flags, 0
        keep = [i for i in range(len(shapes)) if i not in remove]
        shapes, groups, flags = [shapes[i] for i in keep], [groups[i] for i in keep], [flags[i] for i in keep]
        reindex(groups)
        return shapes, groups, T.remove_nodes(layers, remove), flags, len(remove)

    def _remove_invisible(self, shapes, groups, layers, flags):
        """Remove new unlabeled paths whose deletion changes the rendering by <= threshold (sum |dRGB|)."""
        flat = T.flatten(layers)
        if len(shapes) == 0 or len(flat) != len(shapes):
            return shapes, groups, layers, flags
        h, w = self.image.shape[:2]
        threshold = float(self.run_cfg["refinement"]["invisible_threshold"])
        full = render_rgb(w, h, shapes, groups, self.device)
        remove = []
        for i in range(len(shapes)):
            if i >= len(flags) or flags[i] != 1 or never_drop(self.reg.meta(flat[i])):
                continue
            others = copy.deepcopy([g for j, g in enumerate(groups) if j != i])
            for j, g in enumerate(others):
                g.shape_ids = torch.tensor([j])
            without = render_rgb(w, h, [s for j, s in enumerate(shapes) if j != i], others, self.device)
            if torch.sum(torch.abs(without - full)).item() <= threshold:
                remove.append(i)
        if not remove:
            return shapes, groups, layers, flags
        remove = set(remove)
        shapes = [s for i, s in enumerate(shapes) if i not in remove]
        groups = [g for i, g in enumerate(groups) if i not in remove]
        flags = [f for i, f in enumerate(flags) if i not in remove]
        for i, g in enumerate(groups):
            g.shape_ids = torch.tensor([i])
        layers = T.rebuild_layers_from_tree(T.remove_nodes(layers, remove), self.reg)
        shapes, groups, flags, layers = self._reorder_to_tree(shapes, groups, flags, layers)
        return shapes, groups, layers, flags

    def _reorder_to_tree(self, shapes, groups, flags, layers):
        """Re-derive depth layers from the tree and sort paths into painter's order."""
        flat = T.flatten(layers)
        if len(flat) != len(shapes) or len(flat) != len(groups):
            return shapes, groups, flags, layers
        ordered_layers = T.rebuild_layers_from_tree(layers, self.reg)
        entries = {id(m): (shapes[i], groups[i], flags[i] if flags is not None and i < len(flags) else None)
                   for i, m in enumerate(flat)}
        ordered = T.flatten(ordered_layers)
        if any(id(m) not in entries for m in ordered):
            return shapes, groups, flags, layers
        shapes = [entries[id(m)][0] for m in ordered]
        groups = [entries[id(m)][1] for m in ordered]
        flags = None if flags is None else [entries[id(m)][2] for m in ordered]
        reindex(groups)
        return shapes, groups, flags, ordered_layers
