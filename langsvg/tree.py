"""SVG tree construction from multi-granularity masks (Sec. 3.1 of the paper).

The tree is stored as ``layers``: a list of lists of masks where ``layers[d]`` holds the
nodes at depth ``d`` (``layers[0] == [background]``). Parent links live in the
:class:`~langsvg.masks.MaskRegistry` (``meta["layer_parent_mask_id"] = id(parent)``).
Flattening ``layers`` gives the painter's rendering order of the SVG paths.
"""
from typing import Optional, Tuple

import cv2
import numpy as np

from .masks import MaskRegistry, is_background_meta, is_protected_semantic_meta

UNKNOWN_META = {"tag": "unknown", "source": "unknown"}


def _meta(lookup: Optional[MaskRegistry], mask) -> dict:
    return lookup.get(id(mask), UNKNOWN_META) if lookup is not None else UNKNOWN_META


def set_parent(lookup: Optional[MaskRegistry], child, parent) -> None:
    if lookup is None:
        return
    meta = lookup.setdefault(id(child), {"tag": "unknown", "source": "unknown"})
    if parent is None:
        meta.pop("layer_parent_mask_id", None)
    else:
        meta["layer_parent_mask_id"] = id(parent)


def flatten(layers: list) -> list:
    return [mask for layer in layers for mask in layer]


def background_of(layers: list, lookup: Optional[MaskRegistry] = None):
    for layer in layers:
        for mask in layer:
            if is_background_meta(lookup.get(id(mask), {}) if lookup is not None else {}):
                return mask
    if layers and layers[0]:
        return layers[0][0]
    return None


def _render_order_key(mask, lookup):
    meta = lookup.get(id(mask), {}) if lookup is not None else {}
    semantic_rank = 1 if is_protected_semantic_meta(meta) else 0
    pixels = int(cv2.countNonZero(np.where(mask > 0, 255, 0).astype(np.uint8)))
    return semantic_rank, -pixels, str(meta.get("tag", "unknown"))


def order_layer_for_render(layer: list, lookup: Optional[MaskRegistry] = None) -> list:
    """Within a depth: unknown nodes before labeled ones, larger before smaller."""
    return sorted(layer, key=lambda mask: _render_order_key(mask, lookup))


def rebuild_layers_from_tree(layers: list, lookup: Optional[MaskRegistry] = None) -> list:
    """Re-derive depth layers from parent links (repairing dangling links to the root)."""
    if len(layers) == 0:
        return layers
    flat = flatten(layers)
    if len(flat) == 0:
        return layers
    background = background_of(layers, lookup)
    if background is None:
        return [order_layer_for_render(layer, lookup) for layer in layers if len(layer) > 0]

    background_id = id(background)
    present = {id(mask) for mask in flat}
    if lookup is not None:
        for mask in flat:
            meta = lookup.setdefault(id(mask), {"tag": "unknown", "source": "unknown"})
            if id(mask) == background_id or is_background_meta(meta):
                meta.pop("layer_parent_mask_id", None)
                continue
            parent_ref = meta.get("layer_parent_mask_id")
            seen = {id(mask)}
            while parent_ref is not None and parent_ref not in present and parent_ref not in seen:
                seen.add(parent_ref)
                parent_ref = lookup.get(parent_ref, {}).get("layer_parent_mask_id")
            if parent_ref is None or parent_ref == id(mask) or parent_ref not in present:
                parent_ref = background_id
            meta["layer_parent_mask_id"] = parent_ref

    id_to_mask = {id(mask): mask for mask in flat}
    depth_cache = {background_id: 0}

    def depth(mask, visiting=None) -> int:
        visiting = set() if visiting is None else visiting
        mask_id = id(mask)
        if mask_id in depth_cache:
            return depth_cache[mask_id]
        if mask_id in visiting:  # cycle: re-attach to the root
            if lookup is not None:
                lookup.setdefault(mask_id, {"tag": "unknown", "source": "unknown"})["layer_parent_mask_id"] = background_id
            depth_cache[mask_id] = 1
            return 1
        visiting.add(mask_id)
        if lookup is None:
            depth_cache[mask_id] = 1
            return 1
        parent_ref = lookup.get(mask_id, {}).get("layer_parent_mask_id")
        if parent_ref not in id_to_mask or parent_ref == mask_id:
            parent_ref = background_id
        depth_cache[mask_id] = depth(id_to_mask[parent_ref], visiting) + 1
        return depth_cache[mask_id]

    by_depth = {}
    for mask in flat:
        by_depth.setdefault(depth(mask), []).append(mask)
    rebuilt = []
    for d in sorted(by_depth):
        rebuilt.append([background] if d == 0 else order_layer_for_render(by_depth[d], lookup))
    return rebuilt


def _is_prior_object(meta: dict) -> bool:
    """An object of the open-vocabulary prior: a Grounded-SAM mask or a dataset GT object."""
    return meta.get("source") == "grounded_sam" or (meta.get("source") == "dataset_gt" and meta.get("level") == "object")


def init_tree_with_semantic_masks(layers: list, masks: list, lookup: MaskRegistry) -> Tuple[list, list]:
    """Attach labeled (open-vocabulary / dataset) masks first, keeping their own hierarchy.

    Returns the initial layers and the remaining unlabeled masks.
    """
    background = background_of(layers, lookup)
    if background is None:
        return layers, masks
    semantic, unknown = [], []
    for mask in masks:
        (semantic if is_protected_semantic_meta(lookup.get(id(mask), {})) else unknown).append(mask)
    if len(semantic) == 0:
        return layers, unknown

    annotation_to_mask = {}
    for mask in semantic:
        annotation_id = lookup.get(id(mask), {}).get("annotation_id")
        if annotation_id is not None:
            annotation_to_mask[annotation_id] = mask
    for mask in semantic:
        meta = lookup.setdefault(id(mask), {"tag": "unknown", "source": "unknown"})
        parent_ref = meta.get("parent_id", meta.get("parent_annotation_id"))
        parent = None
        if parent_ref in annotation_to_mask:
            parent = annotation_to_mask[parent_ref]
        elif isinstance(parent_ref, int) and parent_ref in lookup:  # parent given as a mask id
            parent = next((candidate for candidate in semantic if id(candidate) == parent_ref), None)
        if parent is mask:
            parent = None
        set_parent(lookup, mask, parent if parent is not None else background)
    return rebuild_layers_from_tree([[background]] + [semantic], lookup), unknown


def insert_masks(layers: list, masks: list, lookup: Optional[MaskRegistry] = None) -> list:
    """Insert multi-granularity masks coarse-to-fine by spatial containment.

    A mask contained (>= 50%) in a larger node becomes the child of its deepest container
    (the layers are searched top-down for a deeper container); a mask that duplicates an
    existing node (IoU > 0.5) or overlaps a Grounded-SAM / GT object while being a Semantic-SAM
    level-2/3 mask is dropped; a mask no node contains is attached to the root.

    Fix (2026-11-13): the original Layered Vectorization logic only made a contained mask a
    child when its container was in the deepest layer; otherwise it moved to the next layer
    and, finding no container there, was attached to the root. After the parent-based
    re-layering it became a same-layer sibling of its container and the structure
    optimization's exclusion loss squeezed it (ADE_val_00001699: a SAM curtain 98% inside a
    Semantic-SAM blob collapsed to a 10 px strip). The container is now remembered.
    """
    while len(masks) > 0:
        found = False
        root = background_of(layers, lookup)
        mask_new = masks.pop(0)
        meta_new = _meta(lookup, mask_new)
        protected = is_protected_semantic_meta(meta_new)

        if meta_new.get("level") == "object":
            while len(layers) < 2:
                layers.append([])
            layers[1].append(mask_new)
            set_parent(lookup, mask_new, root)
            continue

        container, container_layer = None, -1  # deepest node found so far that contains mask_new
        for layer_i, layer in enumerate(layers):
            if len(layer) == 0:
                continue
            for mask_i, mask in enumerate(layer):
                meta = _meta(lookup, mask)
                area0 = cv2.countNonZero(mask)
                area1 = cv2.countNonZero(mask_new)
                inter = np.sum(mask.astype(np.float32) + mask_new.astype(np.float32) == 510)
                union = cv2.countNonZero(mask + mask_new)
                if (meta_new.get("tag") in {"semantic_level_2", "semantic_level_3"}
                        and _is_prior_object(meta) and inter > 0):
                    found = True
                    break
                if (not protected) and union > 0 and inter / union > 0.5:
                    found = True
                    break
                if area1 > 0 and inter / area1 >= 0.5 and area1 < area0:
                    container, container_layer = mask, layer_i
                    if layer_i == len(layers) - 1:
                        _attach_child(layers, mask_new, container, container_layer, lookup)
                        found = True
                    break  # look for a deeper container in the next layer
                if mask_i == len(layer) - 1:
                    if container is not None:  # no deeper container: child of the deepest one found
                        _attach_child(layers, mask_new, container, container_layer, lookup)
                    else:
                        layers[layer_i].append(mask_new)
                        set_parent(lookup, mask_new, root)
                    found = True
                    break
            if found:
                break
        if not found and container is not None:
            _attach_child(layers, mask_new, container, container_layer, lookup)
    return rebuild_layers_from_tree(layers, lookup)


def _attach_child(layers: list, mask_new, parent, parent_layer: int, lookup: Optional[MaskRegistry]) -> None:
    while len(layers) <= parent_layer + 1:
        layers.append([])
    layers[parent_layer + 1].append(mask_new)
    set_parent(lookup, mask_new, parent)


def refine_by_occlusion(layers: list, lookup: Optional[MaskRegistry] = None, threshold: float = 0.9) -> list:
    """Re-parent unknown nodes that are (>= threshold) hidden by later nodes."""
    entries = [(layer_i, mask) for layer_i, layer in enumerate(layers) for mask in layer]
    if len(entries) <= 1:
        return layers

    occluded_ids, occluded, removed_parent = set(), [], {}
    for flat_i, (layer_i, mask) in enumerate(entries):
        meta = _meta(lookup, mask)
        if layer_i == 0 or is_background_meta(meta):
            continue
        if str(meta.get("tag", "unknown")).strip().lower() != "unknown":
            continue
        mask_area = cv2.countNonZero(mask)
        later = [entry_mask for _, entry_mask in entries[flat_i + 1:]]
        if mask_area <= 0 or len(later) == 0:
            continue
        union = np.zeros_like(mask, dtype=np.uint8)
        for later_mask in later:
            union = cv2.bitwise_or(union, later_mask)
        overlap = np.sum(mask.astype(np.uint16) + union.astype(np.uint16) == 510)
        if float(overlap) / float(mask_area) >= threshold:
            occluded_ids.add(id(mask))
            occluded.append(mask)
            if lookup is not None and id(mask) in lookup:
                meta = lookup[id(mask)]
                replacement = meta.get("layer_parent_mask_id", meta.get("parent_id", meta.get("parent_annotation_id")))
                removed_parent[id(mask)] = replacement
                if meta.get("annotation_id") is not None:
                    removed_parent[meta["annotation_id"]] = replacement

    if len(occluded_ids) == 0:
        return layers
    refined = [[mask for mask in layer if id(mask) not in occluded_ids] for layer in layers]
    if lookup is not None and removed_parent:
        reconnect_children(refined, lookup, removed_parent)

    for mask in occluded:  # re-insert under the smallest node that contains it
        if lookup is not None and id(mask) in lookup:
            for key in ("parent_id", "parent_annotation_id", "layer_parent_mask_id"):
                lookup[id(mask)].pop(key, None)
        mask_area = cv2.countNonZero(mask)
        best_layer, best_parent, best_area = None, None, None
        if mask_area > 0:
            for layer_i, layer in enumerate(refined):
                for candidate in layer:
                    candidate_area = cv2.countNonZero(candidate)
                    if candidate_area <= mask_area:
                        continue
                    inter = np.sum(mask.astype(np.uint16) + candidate.astype(np.uint16) == 510)
                    if float(inter) / float(mask_area) < 0.5:
                        continue
                    if best_area is None or candidate_area < best_area:
                        best_area, best_layer, best_parent = candidate_area, layer_i, candidate
        if best_layer is None:
            continue
        while len(refined) <= best_layer + 1:
            refined.append([])
        refined[best_layer + 1].append(mask)
        set_parent(lookup, mask, best_parent)
    return refined


def reconnect_children(layers: list, lookup: MaskRegistry, removed_parent: dict) -> None:
    """Point children of removed nodes to the closest surviving ancestor."""
    def resolve(ref):
        seen = set()
        while ref in removed_parent and ref not in seen:
            seen.add(ref)
            ref = removed_parent[ref]
        return ref

    for layer in layers:
        for mask in layer:
            meta = lookup.get(id(mask))
            if meta is None:
                continue
            key = next((k for k in ("layer_parent_mask_id", "parent_id", "parent_annotation_id") if k in meta), None)
            if key is None or meta.get(key) not in removed_parent:
                continue
            replacement = resolve(meta.get(key))
            if replacement is None:
                for k in ("layer_parent_mask_id", "parent_id", "parent_annotation_id"):
                    meta.pop(k, None)
            else:
                meta[key] = replacement


def insert_proposals(layers: list, masks: list, lookup: Optional[MaskRegistry] = None,
                     delete_overlapping: bool = True) -> Tuple[list, list]:
    """Insert refinement proposals: deepest containing node (>= 50%) becomes the parent.

    Proposals that duplicate an existing node (IoU > 0.7) are returned as deleted.
    """
    deleted = []
    while len(masks) > 0:
        root = background_of(layers, lookup)
        mask_new = masks.pop(0)
        protected = is_protected_semantic_meta(_meta(lookup, mask_new))
        area1 = cv2.countNonZero(mask_new)
        duplicate = False
        best_parent, best_parent_layer = root, 0

        for layer_i in range(len(layers) - 1, -1, -1):
            best_layer_parent, best_containment, best_area = None, -1.0, None
            for mask in layers[layer_i]:
                area0 = cv2.countNonZero(mask)
                inter = np.sum(mask.astype(np.float32) + mask_new.astype(np.float32) == 510)
                union = cv2.countNonZero(np.where((mask > 0) | (mask_new > 0), 255, 0).astype(np.uint8))
                iou = inter / union if union > 0 else 0.0
                if delete_overlapping and (not protected) and iou > 0.7:
                    deleted.append(mask_new)
                    duplicate = True
                    break
                containment = inter / area1 if area1 > 0 else 0.0
                if area1 > 0 and area1 < area0 and containment >= 0.5:
                    if (containment > best_containment
                            or (containment == best_containment and (best_area is None or area0 < best_area))):
                        best_layer_parent, best_containment, best_area = mask, containment, area0
            if duplicate:
                break
            if best_layer_parent is not None:
                best_parent, best_parent_layer = best_layer_parent, layer_i
                break
        if duplicate:
            continue
        target = best_parent_layer + 1 if best_parent is not None else 1
        while len(layers) <= target:
            layers.append([])
        layers[target].append(mask_new)
        set_parent(lookup, mask_new, best_parent)
    return rebuild_layers_from_tree(layers, lookup), deleted


def remove_nodes(layers: list, remove_flat_indices: set) -> list:
    """Drop nodes by flat index and remove empty layers."""
    rebuilt, index = [], 0
    for layer in layers:
        kept = []
        for mask in layer:
            if index not in remove_flat_indices:
                kept.append(mask)
            index += 1
        if kept:
            rebuilt.append(kept)
    return rebuilt
