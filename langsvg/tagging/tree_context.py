"""Scene-graph helpers: node index, depths, hierarchy tags, bbox layout, average color and crops.

A scene graph is the ``tree.json`` payload written by ``scripts/vectorize.py``: ``{"scene": {...}, "nodes": [...]}``
where each node has ``node_id``, ``parent_id``, ``children_ids``, ``tag``, ``level`` and
``visible_bbox_xyxy`` (full-image pixel coordinates).
"""

from __future__ import annotations

import collections
import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from PIL import Image, ImageStat

Node = Dict[str, Any]
NodeIndex = Dict[str, Node]

UNLABELED_TAGS = {"", "unknown", "background"}


# --------------------------------------------------------------------------- basic utilities


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def normalize_bbox(bbox: Any) -> Optional[List[int]]:
    """Round an ``[x0, y0, x1, y1]`` box to ints; ``None`` if malformed or empty."""
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    try:
        x0, y0, x1, y1 = [int(round(float(v))) for v in bbox]
    except (TypeError, ValueError):
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def crop_with_margin(image: Image.Image, bbox_xyxy: List[int], margin_ratio: float) -> Image.Image:
    """Crop ``bbox_xyxy`` enlarged by ``margin_ratio`` of its size on each side (clipped to the image)."""
    width, height = image.size
    x0, y0, x1, y1 = bbox_xyxy
    x0, x1 = max(0, min(width, x0)), max(0, min(width, x1))
    y0, y1 = max(0, min(height, y0)), max(0, min(height, y1))
    if x1 <= x0 or y1 <= y0:
        return image.copy()
    margin_x = int(round((x1 - x0) * margin_ratio))
    margin_y = int(round((y1 - y0) * margin_ratio))
    box = (max(0, x0 - margin_x), max(0, y0 - margin_y), min(width, x1 + margin_x), min(height, y1 + margin_y))
    if box[2] <= box[0] or box[3] <= box[1]:
        return image.copy()
    return image.crop(box)


# --------------------------------------------------------------------------- tag strings


def normalize_label(value: Any) -> str:
    """Lowercase, ``_`` -> space, curly apostrophe -> ``'``, collapse double spaces."""
    text = str(value or "").strip().lower().replace("_", " ").replace("’", "'")
    while "  " in text:
        text = text.replace("  ", " ")
    return text


def normalize_tag(value: Any) -> str:
    """``normalize_label`` plus removal of a trailing confidence such as ``"(0.83)"``."""
    return re.sub(r"\(\s*\d+(?:\.\d+)?\s*\)$", "", normalize_label(value)).strip()


def label_leaf(label: str) -> str:
    """Last component of a possessive hierarchical tag: ``"car's wheel's tire"`` -> ``"tire"``."""
    parts = normalize_label(label).split("'s ")
    return parts[-1].strip()


def label_parent(label: str) -> str:
    """First component of a hierarchical tag, or ``""`` for a single-level tag."""
    parts = normalize_label(label).split("'s ")
    return parts[0].strip() if len(parts) > 1 else ""


def is_internal_model_tag(value: Any) -> bool:
    """True for empty tags and segmentation-model artifacts (``"sam level 2"``, ``"mask"``, ...)."""
    tag = re.sub(r"\s+", " ", normalize_tag(value).replace("_", " ")).strip()
    if not tag:
        return True
    if tag in {"unknown", "background"}:
        return False
    return bool(
        re.search(r"\b(?:unsamv2|unsam|semantic sam|semanticsam|grounded sam|sam2?|sam)\b", tag)
        or re.search(r"\b(?:gra|granularity|level)\s*\d+(?:\.\d+)?\b", tag)
        or re.search(r"\b(?:proposal|mask|l1 residual|l1 proposal)\b", tag)
    )


def semantic_context_tag(value: Any) -> str:
    """The tag if it carries semantics usable as context, else ``""``."""
    tag = normalize_tag(value)
    if tag in UNLABELED_TAGS or is_internal_model_tag(tag):
        return ""
    return tag


# --------------------------------------------------------------------------- tree structure


def node_index(payload: Dict[str, Any]) -> NodeIndex:
    return {str(node.get("node_id")): node for node in payload.get("nodes", [])}


def parent_of(node: Node, index: NodeIndex) -> Optional[Node]:
    parent_id = node.get("parent_id")
    return index.get(str(parent_id)) if parent_id is not None else None


def node_depths(payload: Dict[str, Any]) -> Dict[str, int]:
    """Depth of every node (roots and nodes with a missing parent have depth 0)."""
    index = node_index(payload)
    depths: Dict[str, int] = {}

    def depth(node_id: str) -> int:
        if node_id not in depths:
            node = index.get(node_id)
            parent = parent_of(node, index) if node else None
            depths[node_id] = depth(str(parent["node_id"])) + 1 if parent else 0
        return depths[node_id]

    for node in payload.get("nodes", []):
        depth(str(node.get("node_id")))
    return depths


def ancestor_tags(node: Node, index: NodeIndex) -> List[str]:
    """Semantic tags of the ancestors, root first (unlabeled/internal ancestors skipped)."""
    chain: List[str] = []
    seen: Set[str] = set()
    current = node
    while current.get("parent_id") is not None:
        parent_id = str(current["parent_id"])
        if parent_id in seen or parent_id not in index:
            break
        seen.add(parent_id)
        current = index[parent_id]
        tag = semantic_context_tag(current.get("tag"))
        if tag:
            chain.append(tag)
    chain.reverse()
    return chain


def sibling_and_child_tags(node: Node, index: NodeIndex, max_tags: int = 16) -> Tuple[List[str], List[str]]:
    """Semantic tags of labeled siblings and labeled children (at most ``max_tags`` each)."""
    sibling_tags: List[str] = []
    parent = parent_of(node, index)
    if parent is not None:
        for sibling_id in parent.get("children_ids") or []:
            sibling_id = str(sibling_id)
            if sibling_id == str(node.get("node_id")) or sibling_id not in index:
                continue
            tag = semantic_context_tag(index[sibling_id].get("tag"))
            if tag:
                sibling_tags.append(tag)
            if len(sibling_tags) >= max_tags:
                break
    child_tags: List[str] = []
    for child_id in node.get("children_ids") or []:
        child = index.get(str(child_id))
        if child is None:
            continue
        tag = semantic_context_tag(child.get("tag"))
        if tag:
            child_tags.append(tag)
        if len(child_tags) >= max_tags:
            break
    return sibling_tags, child_tags


def semantic_object_ancestor(node: Node, index: NodeIndex) -> Optional[Node]:
    """Nearest ancestor with ``level == "object"`` and a semantic tag (the labeled subtree root)."""
    current = node
    while current.get("parent_id") is not None:
        parent_id = str(current["parent_id"])
        if parent_id not in index:
            break
        current = index[parent_id]
        tag = semantic_context_tag(current.get("tag"))
        if str(current.get("level", "")).strip().lower() == "object" and tag not in UNLABELED_TAGS | {"texture"}:
            return current
    return None


def select_target_nodes(
    payload: Dict[str, Any],
    target_tags: Iterable[str] = ("unknown", "texture"),
    min_area: int = 64,
    max_nodes: int = 256,
    require_object_ancestor: bool = True,
) -> List[Node]:
    """Nodes to tag: tag in ``target_tags``, inside a labeled object subtree, bbox area >= ``min_area``.

    Sorted by (depth, shape_index) so that parents are tagged before their descendants.
    """
    target_tags = {normalize_tag(tag) for tag in target_tags}
    index = node_index(payload)
    depths = node_depths(payload)
    selected = []
    for node in payload.get("nodes", []):
        if normalize_tag(node.get("tag")) not in target_tags:
            continue
        if require_object_ancestor and semantic_object_ancestor(node, index) is None:
            continue
        if normalize_bbox(node.get("visible_bbox_xyxy")) is None or safe_int(node.get("visible_bbox_area")) < min_area:
            continue
        selected.append(node)
    selected.sort(key=lambda n: (depths.get(str(n.get("node_id")), 0), safe_int(n.get("shape_index"))))
    return selected[:max_nodes] if max_nodes > 0 else selected


# --------------------------------------------------------------------------- closed-set context (with hidden tags)


def visible_tag(node: Optional[Node], hidden_ids: Set[str]) -> str:
    """Normalized tag, or ``"unknown"`` for a missing node or a node whose tag is hidden."""
    if not node or str(node.get("node_id")) in hidden_ids:
        return "unknown"
    return normalize_label(node.get("tag")) or "unknown"


def hidden_tag_context(node: Node, index: NodeIndex, hidden_ids: Set[str]) -> Dict[str, Any]:
    """Ancestor / parent / sibling / child tags with the tags of ``hidden_ids`` treated as unknown."""
    ancestors = []
    current = node
    while current.get("parent_id") and str(current["parent_id"]) in index:
        current = index[str(current["parent_id"])]
        tag = visible_tag(current, hidden_ids)
        if tag not in UNLABELED_TAGS:
            ancestors.append(tag)
    ancestors.reverse()

    parent = parent_of(node, index)
    sibling_tags = []
    if parent:
        for sibling_id in parent.get("children_ids", []):
            tag = visible_tag(index.get(str(sibling_id)), hidden_ids)
            if str(sibling_id) not in hidden_ids and tag not in UNLABELED_TAGS:
                sibling_tags.append(tag)
    child_tags = []
    for child_id in node.get("children_ids", []):
        tag = visible_tag(index.get(str(child_id)), hidden_ids)
        if tag not in UNLABELED_TAGS:
            child_tags.append(tag)
    return {
        "ancestor_tags": ancestors,
        "parent_tag": visible_tag(parent, hidden_ids),
        "sibling_tags": sibling_tags[:12],
        "child_tags": child_tags[:12],
    }


def object_context_node(node: Node, index: NodeIndex) -> Optional[Node]:
    """Nearest ancestor with ``level == "object"`` (else the nearest ancestor with a bbox)."""
    current = node
    fallback = None
    while current.get("parent_id") and str(current["parent_id"]) in index:
        current = index[str(current["parent_id"])]
        bbox = current.get("visible_bbox_xyxy")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        if fallback is None:
            fallback = current
        if str(current.get("level", "")).strip().lower() == "object":
            return current
    return fallback


# --------------------------------------------------------------------------- bbox layout and color


def _center(bbox: List[int]) -> Tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _size(bbox: List[int]) -> Tuple[float, float]:
    return (max(float(bbox[2] - bbox[0]), 1.0), max(float(bbox[3] - bbox[1]), 1.0))


def relative_position(target_bbox: List[int], other_bbox: List[int]) -> str:
    """Where ``other_bbox`` lies relative to ``target_bbox``: ``"above-left"``, ``"right"``, ..."""
    (tx, ty), (ox, oy) = _center(target_bbox), _center(other_bbox)
    (tw, th), (ow, oh) = _size(target_bbox), _size(other_bbox)
    dx, dy = ox - tx, oy - ty
    x_threshold, y_threshold = 0.25 * max(tw, ow), 0.25 * max(th, oh)
    horizontal = "left" if dx < -x_threshold else "right" if dx > x_threshold else ""
    vertical = "above" if dy < -y_threshold else "below" if dy > y_threshold else ""
    if horizontal and vertical:
        return f"{vertical}-{horizontal}"
    return horizontal or vertical or "overlapping_or_centered"


def bbox_in_container(bbox: List[int], container: Optional[List[int]]) -> Optional[List[float]]:
    """``bbox`` in the container's normalized [0, 1] coordinates (4 decimals)."""
    if container is None:
        return None
    width, height = _size(container)
    return [
        round((bbox[0] - container[0]) / width, 4),
        round((bbox[1] - container[1]) / height, 4),
        round((bbox[2] - container[0]) / width, 4),
        round((bbox[3] - container[1]) / height, 4),
    ]


def average_rgb(image: Image.Image, bbox: List[int]) -> Optional[List[int]]:
    """Mean RGB of the bbox region (computed on a <=32px thumbnail)."""
    clipped = [
        max(0, min(image.width, int(round(bbox[0])))),
        max(0, min(image.height, int(round(bbox[1])))),
        max(0, min(image.width, int(round(bbox[2])))),
        max(0, min(image.height, int(round(bbox[3])))),
    ]
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        return None
    crop = image.crop(tuple(clipped))
    if crop.width > 32 or crop.height > 32:
        crop.thumbnail((32, 32))
    return [int(round(value)) for value in ImageStat.Stat(crop.convert("RGB")).mean[:3]]


def _layout_summary(
    item: Node,
    target_bbox: List[int],
    parent_bbox: Optional[List[int]],
    hidden_ids: Set[str],
    image: Image.Image,
    is_target: bool,
) -> Optional[Dict[str, Any]]:
    bbox = normalize_bbox(item.get("visible_bbox_xyxy"))
    if bbox is None:
        return None
    tag = visible_tag(item, hidden_ids)
    return {
        "node_id": item.get("node_id"),
        "is_target": is_target,
        "tag": tag,
        "label_leaf": label_leaf(tag),
        "tag_hidden": str(item.get("node_id")) in hidden_ids,
        "level": item.get("level"),
        "status": item.get("status"),
        "shape_index": item.get("shape_index"),
        "layer_index": item.get("layer_index"),
        "bbox_xyxy": bbox,
        "bbox_in_parent_xyxy_0to1": bbox_in_container(bbox, parent_bbox),
        "relative_to_target": "target" if is_target else relative_position(target_bbox, bbox),
        "bbox_average_rgb": average_rgb(image, bbox),
        "visible_bbox_area": item.get("visible_bbox_area"),
        "visible_mask_pixels": item.get("visible_mask_pixels"),
    }


def _container_layout(item: Optional[Node], hidden_ids: Set[str], image: Image.Image) -> Dict[str, Any]:
    bbox = normalize_bbox(item.get("visible_bbox_xyxy")) if item else None
    tag = visible_tag(item, hidden_ids)
    return {
        "node_id": item.get("node_id") if item else None,
        "tag": tag,
        "label_leaf": label_leaf(tag),
        "level": item.get("level") if item else None,
        "bbox_xyxy": bbox,
        "bbox_average_rgb": average_rgb(image, bbox) if bbox else None,
    }


def build_layout_context(
    node: Node,
    index: NodeIndex,
    hidden_ids: Set[str],
    image: Image.Image,
    object_node: Optional[Node],
) -> Dict[str, Any]:
    """Target / parent / object / sibling bboxes with relative positions and average RGB."""
    target_bbox = normalize_bbox(node.get("visible_bbox_xyxy"))
    if target_bbox is None:
        return {}
    parent = parent_of(node, index)
    parent_bbox = normalize_bbox(parent.get("visible_bbox_xyxy")) if parent else None

    parent_child_layout = []
    sibling_bboxes = []
    known_sibling_leaf_counts: collections.Counter = collections.Counter()
    if parent:
        for child_id in parent.get("children_ids", []):
            child = index.get(str(child_id))
            if child is None:
                continue
            is_target = str(child_id) == str(node.get("node_id"))
            summary = _layout_summary(child, target_bbox, parent_bbox, hidden_ids, image, is_target)
            if summary is None:
                continue
            parent_child_layout.append(summary)
            if not is_target:
                sibling_bboxes.append(summary)
                if not summary["tag_hidden"] and summary["label_leaf"] not in UNLABELED_TAGS:
                    known_sibling_leaf_counts[summary["label_leaf"]] += 1

    return {
        "image_size_wh": [image.width, image.height],
        "target_bbox_xyxy": target_bbox,
        "target_bbox_in_parent_xyxy_0to1": bbox_in_container(target_bbox, parent_bbox),
        "target_bbox_average_rgb": average_rgb(image, target_bbox),
        "parent_layout": _container_layout(parent, hidden_ids, image),
        "object_layout": _container_layout(object_node, hidden_ids, image),
        "parent_child_layout": parent_child_layout,
        "sibling_bboxes": sibling_bboxes,
        "known_sibling_leaf_counts": dict(sorted(known_sibling_leaf_counts.items())),
    }
