"""Closed-set tagging: pick one label from a candidate list for a node of a scene graph.

The request layout is that of the paper's ADE20KPart234 hidden-tag recovery benchmark. The
default setting ``a4_full_hierarchy_color`` gives the VLM the full image, a crop of the containing
object, a crop of the target bbox, ancestor/parent/sibling/child tags, parent/sibling bbox layout,
average RGB colors and the candidate labels. Settings ``a0``-``a6`` are the paper's ablations.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from PIL import Image

from . import prompts
from .open_vocab import run_depth_ordered
from .qwen import QwenVLGenerator, run_with_batch_splitting
from .tree_context import (
    Node,
    NodeIndex,
    build_layout_context,
    crop_with_margin,
    hidden_tag_context,
    node_depths,
    node_index,
    normalize_label,
    object_context_node,
    select_target_nodes,
)

DEFAULT_SETTING = "a4_full_hierarchy_color"

_FULL_CONTEXT = {
    "include_object_tag": True,
    "include_parent_tags": True,
    "include_sibling_tags": True,
    "include_child_tags": True,
    "include_layout_geometry": True,
    "include_color": True,
}

# Descriptions are part of the prompt; they are the exact strings used for the paper's runs.
SETTINGS: Dict[str, Dict[str, Any]] = {
    "a0_visual_only": {
        "description": (
            "Best visual inputs only: full image, object crop, target crop, target bbox, "
            "and the same candidate label list. No scene-graph hierarchy tags or sibling layout."
        ),
        **{key: False for key in _FULL_CONTEXT},
    },
    "a1_parent_identity": {
        "description": (
            "Visual inputs plus top-down hierarchy identity: object tag, ancestor_tags, "
            "and parent_tag. No sibling tags, sibling bboxes, or color/layout."
        ),
        **_FULL_CONTEXT,
        "include_sibling_tags": False,
        "include_child_tags": False,
        "include_layout_geometry": False,
        "include_color": False,
    },
    "a2_parent_sibling_tags": {
        "description": (
            "Visual inputs plus parent/ancestor identity and same-parent sibling/child tags. "
            "No sibling bboxes, relative layout, or color."
        ),
        **_FULL_CONTEXT,
        "include_layout_geometry": False,
        "include_color": False,
    },
    "a3_geometry_no_color": {
        "description": (
            "Visual inputs plus hierarchy tags and parent/sibling/child bbox geometry. "
            "Average RGB/color fields are removed."
        ),
        **_FULL_CONTEXT,
        "include_color": False,
    },
    "a4_full_hierarchy_color": {
        "description": (
            "Visual inputs plus the full hierarchy/layout/color context used by masking.py. "
            "This is the fair full-context endpoint."
        ),
        **_FULL_CONTEXT,
    },
    "a5_full_context_no_crops": {
        "description": (
            "Full image plus full hierarchy/layout/color context, but no object crop and no target crop. "
            "This tests whether crop images add value beyond the full image and metadata."
        ),
        **_FULL_CONTEXT,
        "include_full_image": True,
        "include_object_crop": False,
        "include_target_crop": False,
        "instruction_addition": (
            "No crop images are provided for this setting. Use the full image, bbox_xyxy, "
            "and all enabled scene-graph/layout/color metadata to infer the target label."
        ),
    },
    "a6_metadata_only_no_images": {
        "description": (
            "No full image and no crop images; use only bbox coordinates, hierarchy/layout/color metadata, "
            "and candidate labels. This measures text/metadata-only prior reasoning."
        ),
        **_FULL_CONTEXT,
        "include_full_image": False,
        "include_object_crop": False,
        "include_target_crop": False,
        "instruction_addition": (
            "No visual images are provided in this setting by design. Do not claim to inspect image pixels. "
            "Use only the provided bbox coordinates, hierarchy, sibling layout, color statistics, and "
            "candidate_labels. Treat this as metadata-only VLM prompting rather than visual recognition."
        ),
    },
}


def normalize_candidate_labels(labels: Iterable[str]) -> List[str]:
    """Sorted, de-duplicated, normalized candidate labels."""
    return sorted({normalize_label(label) for label in labels if normalize_label(label)})


def ontology_part_labels(payload: Dict[str, Any]) -> List[str]:
    """Part labels stored by the vectorizer in ``payload["dataset_vlm_ontology"]`` (may be empty)."""
    labels = [normalize_label(item) for item in (payload.get("dataset_vlm_ontology") or {}).get("part_labels", [])]
    return [label for label in labels if label]


def _strip_color(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _strip_color(v) for k, v in value.items() if k not in {"bbox_average_rgb", "target_bbox_average_rgb"}}
    if isinstance(value, list):
        return [_strip_color(v) for v in value]
    return value


def _strip_layout_tags(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            if key in {"tag", "label_leaf"}:
                cleaned[key] = "unknown"
            elif key == "tag_hidden":
                cleaned[key] = True
            else:
                cleaned[key] = _strip_layout_tags(item)
        return cleaned
    if isinstance(value, list):
        return [_strip_layout_tags(v) for v in value]
    return value


def _filtered_layout(layout: Dict[str, Any], setting: Dict[str, Any]) -> Dict[str, Any]:
    if not setting["include_layout_geometry"]:
        return {}
    filtered = copy.deepcopy(layout)
    if not setting["include_color"]:
        filtered = _strip_color(filtered)
    if not setting["include_sibling_tags"]:
        filtered = _strip_layout_tags(filtered)
        filtered["known_sibling_leaf_counts"] = {}
    return filtered


def build_closed_set_request(
    node: Node,
    index: NodeIndex,
    image: Image.Image,
    candidate_labels: List[str],
    hidden_ids: Iterable[str] = (),
    setting: str = DEFAULT_SETTING,
    image_id: str = "",
    scene_json_path: str = "",
    dataset_name: str = "ADE20KPart234",
    crop_margin_ratio: float = 0.15,
    include_object_crop: bool = True,
    include_target_crop: bool = True,
) -> Dict[str, Any]:
    """VLM request asking for one label of ``candidate_labels`` for ``node``.

    ``hidden_ids`` are nodes whose tags the model must not see (the target is always hidden).
    ``image_id``, ``scene_json_path`` and ``dataset_name`` are echoed in the prompt metadata.
    Returns ``{"system_prompt", "user_payload", "full_image", "crop_image", "extra_images"}``.
    """
    config = SETTINGS[setting]
    bbox = node.get("visible_bbox_xyxy")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError(f"Invalid bbox: {bbox}")
    hidden = {str(item) for item in hidden_ids} | {str(node.get("node_id"))}
    object_node = object_context_node(node, index)
    object_bbox = object_node.get("visible_bbox_xyxy") if object_node else None
    object_crop = (
        crop_with_margin(image, object_bbox, crop_margin_ratio)
        if isinstance(object_bbox, list) and len(object_bbox) == 4
        else None
    )
    target_crop = crop_with_margin(image, bbox, crop_margin_ratio)
    use_full = bool(config.get("include_full_image", True))
    use_object_crop = bool(config.get("include_object_crop", include_object_crop) and object_crop is not None)
    use_target_crop = bool(config.get("include_target_crop", include_target_crop))

    context = hidden_tag_context(node, index, hidden)
    layout = build_layout_context(node, index, hidden, image, object_node)
    filtered = _filtered_layout(layout, config)
    object_tag = normalize_label(object_node.get("tag")) if object_node and config["include_object_tag"] else "unknown"
    payload = {
        "task": "masked_scene_graph_subpart_classification_fair_ablation",
        "ablation_setting": setting,
        "ablation_description": config["description"],
        "dataset_name": dataset_name,
        "image_id": image_id,
        "scene_json_path": scene_json_path,
        "target_node_id": node.get("node_id"),
        "target_node_level": node.get("level"),
        "target_node_status": node.get("status"),
        "masked_node": {"node_id": node.get("node_id"), "tag": "unknown", "original_tag_hidden": True},
        "target_shape_index": node.get("shape_index"),
        "target_layer_index": node.get("layer_index"),
        "bbox_xyxy": bbox,
        "bbox_reference": "target bbox in the original full-image coordinate system",
        "visible_bbox_area": node.get("visible_bbox_area"),
        "visible_mask_pixels": node.get("visible_mask_pixels"),
        "object_context_node": {
            "node_id": object_node.get("node_id") if object_node else None,
            "tag": object_tag,
            "tag_available": bool(config["include_object_tag"]),
            "level": object_node.get("level") if object_node else None,
            "bbox_xyxy": object_bbox,
            "object_crop_included": use_object_crop,
        },
        "image_3_target_bbox_crop": {"available": use_target_crop, "source_bbox_xyxy": bbox},
        "parent_id": node.get("parent_id"),
        "ancestor_tags": context["ancestor_tags"] if config["include_parent_tags"] else [],
        "parent_tag": context["parent_tag"] if config["include_parent_tags"] else "unknown",
        "sibling_tags": context["sibling_tags"] if config["include_sibling_tags"] else [],
        "child_tags": context["child_tags"] if config["include_child_tags"] else [],
        "layout_context": filtered,
        "parent_layout": filtered.get("parent_layout", {}),
        "object_layout": filtered.get("object_layout", {}),
        "sibling_bboxes": filtered.get("sibling_bboxes", []),
        "parent_child_layout": filtered.get("parent_child_layout", []),
        "known_sibling_leaf_counts": filtered.get("known_sibling_leaf_counts", {}) if config["include_sibling_tags"] else {},
        "target_bbox_average_rgb": layout.get("target_bbox_average_rgb") if config["include_color"] else None,
        "ground_truth_hidden": True,
        "all_masked_node_ids_in_this_scene": sorted(hidden),
        "candidate_labels": candidate_labels,
        "enabled_information": {
            "full_image": use_full,
            "object_crop": use_object_crop,
            "target_crop": use_target_crop,
            "object_tag": bool(config["include_object_tag"]),
            "parent_or_ancestor_tags": bool(config["include_parent_tags"]),
            "sibling_tags": bool(config["include_sibling_tags"]),
            "child_tags": bool(config["include_child_tags"]),
            "layout_geometry": bool(config["include_layout_geometry"]),
            "color": bool(config["include_color"]),
        },
        "image_2_description": (
            "This is a crop around the containing object, not the target bbox. "
            "Use bbox_xyxy to locate the masked target within this object context."
            if use_object_crop
            else "No object crop is included."
        ),
        "image_3_description": (
            "This is a crop around the target bbox for local visual details."
            if use_target_crop
            else "No target crop is included."
        ),
        "instruction": (
            "Predict the hidden part/subpart label for bbox_xyxy. Keep the visual evidence fixed across "
            "ablation settings and use only fields marked available in enabled_information. "
            "Select exactly one label from candidate_labels and return exactly one JSON object."
        ),
    }
    addition = str(config.get("instruction_addition", "")).strip()
    if addition:
        payload["instruction"] += " " + addition
    return {
        "system_prompt": prompts.CLOSED_SET_SYSTEM_PROMPT,
        "user_payload": payload,
        "full_image": image if use_full else None,
        "crop_image": object_crop if use_object_crop else None,
        "extra_images": (
            [{"image": target_crop, "description": payload["image_3_description"]}] if use_target_crop else []
        ),
    }


def parse_prediction(result: Dict[str, Any], candidate_labels: List[str]) -> Tuple[str, bool]:
    """``(label, valid)``: the matching candidate string, or the normalized raw label if none matches."""
    by_normalized = {normalize_label(label): label for label in candidate_labels}
    predicted = normalize_label(result.get("predicted_label", result.get("label", "")))
    if predicted in by_normalized:
        return by_normalized[predicted], True
    return predicted, False


def _level_for_label(label: str) -> str:
    depth = label.count("'s ")
    return "object" if depth == 0 else "part" if depth == 1 else "subpart"


def tag_scene_closed_set(
    payload: Dict[str, Any],
    image: Image.Image,
    generator: QwenVLGenerator,
    candidate_labels: Optional[List[str]] = None,
    target_tags: Iterable[str] = ("unknown", "texture"),
    min_area: int = 64,
    max_nodes: int = 256,
    require_object_ancestor: bool = True,
    batch_size: int = 1,
    crop_margin_ratio: float = 0.15,
    setting: str = DEFAULT_SETTING,
    generation: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Assign each ``unknown``/``texture`` node in a labeled subtree one of ``candidate_labels``.

    Nodes are processed parent-first so new tags become context for deeper nodes. Candidates
    default to the scene's ``dataset_vlm_ontology.part_labels``. A prediction outside the
    candidate list (or ``"unknown"``) leaves the node unchanged. Returns ``(payload, rows)``.
    """
    generation = dict(generation or {})
    labels = normalize_candidate_labels(candidate_labels or ontology_part_labels(payload))
    if not labels:
        raise ValueError("No candidate labels: pass candidate_labels or provide dataset_vlm_ontology.part_labels")
    scene = payload.get("scene") or {}
    image_id = Path(str(scene.get("image_path") or "")).stem
    index = node_index(payload)
    depths = node_depths(payload)
    targets = select_target_nodes(payload, target_tags, min_area, max_nodes, require_object_ancestor)
    rows: List[Dict[str, Any]] = []

    def build(node: Node) -> Dict[str, Any]:
        request = build_closed_set_request(
            node, index, image, labels, setting=setting, image_id=image_id, crop_margin_ratio=crop_margin_ratio
        )
        return {"node": node, "request": request}

    def record_error(node: Node, exc: BaseException) -> None:
        rows.append({"node_id": node.get("node_id"), "error": str(exc)})

    def run(batch: List[Dict[str, Any]]) -> None:
        outcomes = run_with_batch_splitting(
            lambda items: generator.generate_json_batch([item["request"] for item in items], **generation),
            batch,
            generator.clear_cache,
        )
        for item, outcome in zip(batch, outcomes):
            node = item["node"]
            if isinstance(outcome, BaseException):
                record_error(node, outcome)
                continue
            (result, _raw), batch_count = outcome
            label, valid = parse_prediction(result, labels)
            accepted = valid and label not in {"unknown", "background"}
            row = {
                "node_id": node.get("node_id"),
                "parent_id": node.get("parent_id"),
                "old_tag": node.get("tag"),
                "old_level": node.get("level"),
                "bbox_xyxy": item["request"]["user_payload"]["bbox_xyxy"],
                "predicted_label": label,
                "prediction_valid": valid,
                "confidence": result.get("confidence"),
                "reason": str(result.get("reason", "")).strip(),
                "accepted": accepted,
                "batch_size": batch_count,
            }
            if accepted:
                node.setdefault("original_tag", node.get("tag"))
                node.setdefault("original_level", node.get("level"))
                node["tag"] = label
                node["level"] = _level_for_label(label)
                node["status"] = f"vlm_tagged_{node['level']}"
                node.pop("tag_source", None)
                node["vlm_tagging"] = {"model": generator.model_name, "mode": "closed_set", "tag": label}
            rows.append(row)

    run_depth_ordered(targets, depths, build, run, record_error, batch_size)

    payload["vlm_tagging"] = {
        "mode": "closed_set",
        "model": generator.model_name,
        "setting": setting,
        "candidate_label_count": len(labels),
        "candidate_nodes": len(targets),
        "processed_nodes": len(rows),
        "updated_nodes": sum(int(row.get("accepted", False)) for row in rows),
        "errors": sum(1 for row in rows if "error" in row),
        "annotations": rows,
    }
    return payload, rows
