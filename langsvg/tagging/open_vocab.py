"""Open-vocabulary hierarchical tagging of ``unknown`` / ``texture`` nodes in a scene graph.

Nodes inside subtrees rooted at semantically labeled objects are tagged parent-first (depth
order), so each prediction becomes context for the nodes below it. The VLM predicts a category
(object / part / subpart / texture / unknown) and a tag; the tag is re-rooted under the parent's
tag, e.g. ``"wheel"`` under ``"car"`` becomes ``"car's wheel"``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from PIL import Image

from . import prompts
from .qwen import QwenVLGenerator, run_with_batch_splitting
from .tree_context import (
    Node,
    NodeIndex,
    ancestor_tags,
    crop_with_margin,
    is_internal_model_tag,
    label_leaf,
    node_depths,
    node_index,
    normalize_bbox,
    normalize_tag,
    select_target_nodes,
    semantic_context_tag,
    semantic_object_ancestor,
    sibling_and_child_tags,
)

CATEGORIES = {"object", "part", "subpart", "texture", "unknown"}


def tag_leaf(tag: Any) -> str:
    return label_leaf(normalize_tag(tag))


def normalize_category(category: Any, tag: Any, ancestors: List[str]) -> str:
    """Validate a predicted category, inferring one from the tag and hierarchy depth if missing."""
    category = normalize_tag(category)
    if category in CATEGORIES:
        return category
    tag = normalize_tag(tag)
    if tag == "unknown":
        return "unknown"
    if tag == "texture" or tag.endswith("'s texture"):
        return "texture"
    if not ancestors:
        return "object"
    return "part" if len(ancestors) == 1 else "subpart"


def compose_tag(ancestors: List[str], predicted_tag: Any, category: str) -> str:
    """Hierarchical tag ``"<parent tag>'s <leaf>"`` (``"<parent tag>'s texture"`` for texture)."""
    predicted_tag = normalize_tag(predicted_tag)
    category = normalize_tag(category)
    if is_internal_model_tag(predicted_tag):
        return "unknown"
    if category == "unknown" or predicted_tag in {"", "unknown", "background"}:
        return "unknown"
    if category == "texture" or predicted_tag == "texture":
        return f"{ancestors[-1]}'s texture" if ancestors else "texture"
    if not ancestors:
        return predicted_tag
    parent_tag = semantic_context_tag(ancestors[-1])
    if not parent_tag:
        return predicted_tag
    if predicted_tag == parent_tag or predicted_tag.startswith(parent_tag + "'s "):
        return predicted_tag
    leaf = tag_leaf(predicted_tag)
    if leaf in {"", "unknown", "background"}:
        return "unknown"
    return f"{parent_tag}'s {leaf}"


def build_open_vocab_request(
    node: Node,
    index: NodeIndex,
    image: Image.Image,
    image_name: str,
    crop_margin_ratio: float = 0.12,
    pascal_prior: bool = False,
) -> Dict[str, Any]:
    """VLM request (full image + target crop + hierarchy context) for one target node."""
    bbox = normalize_bbox(node.get("visible_bbox_xyxy"))
    if bbox is None:
        raise ValueError(f"Invalid bbox for node={node.get('node_id')}: {node.get('visible_bbox_xyxy')}")
    object_node = semantic_object_ancestor(node, index)
    ancestors = ancestor_tags(node, index)
    sibling_tags, child_tags = sibling_and_child_tags(node, index)
    target_level = "object" if not ancestors else "part" if len(ancestors) == 1 else "subpart"
    user_payload: Dict[str, Any] = {
        "task": "open_world_infinite_hierarchical_tagging",
        "image_name": image_name,
        "target_node_id": node.get("node_id"),
        "current_tag": "unknown" if is_internal_model_tag(node.get("tag")) else node.get("tag"),
        "current_level": node.get("level"),
        "target_level_prior": target_level,
        "bbox_xyxy": bbox,
        "bbox_reference": "original full-image coordinate system",
        "visible_bbox_area": node.get("visible_bbox_area"),
        "visible_mask_pixels": node.get("visible_mask_pixels"),
        "ancestor_tags": ancestors,
        "ancestor_labels": [tag_leaf(tag) for tag in ancestors],
        "sibling_tags": sibling_tags,
        "child_tags": child_tags,
        "object_context_node": {
            "node_id": object_node.get("node_id") if object_node else None,
            "tag": semantic_context_tag(object_node.get("tag")) if object_node else "unknown",
            "level": object_node.get("level") if object_node else None,
            "bbox_xyxy": object_node.get("visible_bbox_xyxy") if object_node else None,
        },
        "instruction": prompts.NODE_TAGGING_INSTRUCTION,
    }
    system_prompt = prompts.OPEN_WORLD_SYSTEM_PROMPT
    if pascal_prior:
        system_prompt = prompts.PASCAL_SYSTEM_PROMPT
        user_payload.update(
            {
                "task": "pascal_part116_hierarchical_tagging",
                "pascal_objects": list(prompts.PASCAL_OBJECT_CATEGORIES),
                "pascal_object_prior_scope": "object identity only; parts and subparts remain open-world",
                "instruction": prompts.PASCAL_NODE_INSTRUCTION,
            }
        )
    return {
        "system_prompt": system_prompt,
        "user_payload": user_payload,
        "full_image": image,
        "crop_image": crop_with_margin(image, bbox, crop_margin_ratio),
        "extra_images": [],
    }


def _predicted_tag(result: Dict[str, Any]) -> Any:
    return result.get("tag", result.get("label", result.get("predicted_tag", "")))


def build_verification_request(request: Dict[str, Any], first_result: Dict[str, Any], raw_text: str) -> Dict[str, Any]:
    """Second-pass request asking the VLM to keep or revise its first prediction."""
    user_payload = dict(request["user_payload"])
    user_payload["first_pass_prediction"] = {
        "category": first_result.get("category"),
        "tag": _predicted_tag(first_result),
        "raw_response": str(raw_text or "")[:2000],
    }
    user_payload["verification_instruction"] = prompts.VERIFICATION_INSTRUCTION
    return {
        "system_prompt": prompts.GENERAL_VERIFICATION_SYSTEM_PROMPT,
        "user_payload": user_payload,
        "full_image": request.get("full_image"),
        "crop_image": request.get("crop_image"),
        "extra_images": request.get("extra_images", []),
    }


def _generate_with_verification(
    generator: QwenVLGenerator, requests: List[Dict[str, Any]], generation: Dict[str, Any]
) -> List[Tuple[Dict[str, Any], str]]:
    first = generator.generate_json_batch(requests, **generation)
    verified = generator.generate_json_batch(
        [build_verification_request(req, result, raw) for req, (result, raw) in zip(requests, first)], **generation
    )
    outputs = []
    for (first_result, first_raw), (verified_result, verified_raw) in zip(first, verified):
        final = dict(verified_result or first_result)
        final["first_pass_prediction"] = {"category": first_result.get("category"), "tag": _predicted_tag(first_result)}
        final["verification_prediction"] = {
            "category": verified_result.get("category"),
            "tag": _predicted_tag(verified_result),
            "decision": verified_result.get("decision", ""),
            "reason": verified_result.get("reason", ""),
        }
        final["verification_decision"] = verified_result.get("decision", "")
        final["verification_reason"] = verified_result.get("reason", "")
        final["verification_raw_response"] = str(verified_raw or "")[:2000]
        outputs.append((final, f"[first_pass]\n{first_raw}\n[verification_pass]\n{verified_raw}"))
    return outputs


def run_depth_ordered(
    targets: List[Node],
    depths: Dict[str, int],
    build: Callable[[Node], Dict[str, Any]],
    run: Callable[[List[Dict[str, Any]]], None],
    on_build_error: Callable[[Node, Exception], None],
    batch_size: int,
) -> None:
    """Build and run batches depth by depth (parents first).

    Requests are built lazily, right before their batch runs, so they see the tags written by
    earlier batches. ``build(node)`` returns a request item; ``run(items)`` processes a batch.
    """
    for depth in sorted({depths.get(str(node.get("node_id")), 0) for node in targets}):
        batch: List[Dict[str, Any]] = []
        for node in targets:
            if depths.get(str(node.get("node_id")), 0) != depth:
                continue
            try:
                batch.append(build(node))
            except Exception as exc:  # noqa: BLE001 - a bad node is logged, not fatal
                on_build_error(node, exc)
                continue
            if len(batch) >= max(1, batch_size):
                run(batch)
                batch = []
        run(batch)


def tag_scene_open_vocab(
    payload: Dict[str, Any],
    image: Image.Image,
    generator: QwenVLGenerator,
    image_name: Optional[str] = None,
    target_tags: Iterable[str] = ("unknown", "texture"),
    min_area: int = 64,
    max_nodes: int = 256,
    require_object_ancestor: bool = True,
    batch_size: int = 1,
    crop_margin_ratio: float = 0.12,
    pascal_prior: bool = False,
    verify: bool = False,
    generation: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Tag ``unknown``/``texture`` nodes of ``payload`` in place; returns ``(payload, rows)``.

    Args:
        payload: scene-graph JSON (``tree.json`` written by ``scripts/vectorize.py``).
        image: the full RGB image the scene graph was built from.
        generator: a loaded ``QwenVLGenerator``.
        image_name: file name shown to the model (default: basename of ``scene.image_path``).
        pascal_prior: use PascalPart116 object categories as a soft prior on object identity.
        verify: run a second keep-or-revise verification pass per node.
        generation: keyword arguments for ``generate_json_batch`` (``max_new_tokens``, ``temperature``...).
    Each row records one node's prediction (or ``"error"``). Accepted predictions overwrite the
    node's ``tag``/``level`` and keep the old values in ``original_tag``/``original_level``.
    """
    generation = dict(generation or {})
    if image_name is None:
        image_name = Path(str((payload.get("scene") or {}).get("image_path") or "")).name
    index = node_index(payload)
    depths = node_depths(payload)
    target_tags = sorted({normalize_tag(tag) for tag in target_tags})
    targets = select_target_nodes(payload, target_tags, min_area, max_nodes, require_object_ancestor)
    rows: List[Dict[str, Any]] = []

    def generate(requests: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], str]]:
        if verify:
            return _generate_with_verification(generator, requests, generation)
        return generator.generate_json_batch(requests, **generation)

    def build(node: Node) -> Dict[str, Any]:
        return {"node": node, "request": build_open_vocab_request(node, index, image, image_name, crop_margin_ratio, pascal_prior)}

    def record_error(node: Node, exc: BaseException) -> None:
        rows.append({"node_id": node.get("node_id"), "error": str(exc)})

    def record(item: Dict[str, Any], outcome: Any) -> None:
        node = item["node"]
        if isinstance(outcome, BaseException):
            record_error(node, outcome)
            return
        user_payload = item["request"]["user_payload"]
        (result, _raw), batch_count = outcome
        ancestors = ancestor_tags(node, index)
        raw_tag = _predicted_tag(result)
        category = normalize_category(result.get("category"), raw_tag, ancestors)
        if ancestors and category == "object":
            category = "part" if len(ancestors) == 1 else "subpart"
        final_tag = compose_tag(ancestors, raw_tag, category)
        accepted = category in CATEGORIES and bool(final_tag)
        old_tag = normalize_tag(node.get("tag"))
        row = {
            "node_id": node.get("node_id"),
            "parent_id": node.get("parent_id"),
            "old_tag": node.get("tag"),
            "old_level": node.get("level"),
            "bbox_xyxy": user_payload["bbox_xyxy"],
            "ancestor_tags": ancestors,
            "sibling_tags": user_payload["sibling_tags"],
            "child_tags": user_payload["child_tags"],
            "predicted_tag": final_tag,
            "category": category,
            "accepted": accepted,
            "updated": False,
            "batch_size": batch_count,
        }
        for key in ("first_pass_prediction", "verification_prediction", "verification_decision",
                    "verification_reason", "verification_raw_response"):
            if key in result:
                row[key] = result[key]
        if accepted:
            node.setdefault("original_tag", node.get("tag"))
            node.setdefault("original_level", node.get("level"))
            node["tag"] = final_tag
            node["level"] = category
            node["status"] = f"vlm_tagged_{category}"
            node.pop("tag_source", None)
            node["vlm_tagging"] = {
                "model": generator.model_name,
                "category": category,
                "tag": final_tag,
                "ancestor_tags": ancestors,
            }
            row["updated"] = final_tag != old_tag
        rows.append(row)

    def run(batch: List[Dict[str, Any]]) -> None:
        outcomes = run_with_batch_splitting(
            lambda items: generate([item["request"] for item in items]), batch, generator.clear_cache
        )
        for item, outcome in zip(batch, outcomes):
            record(item, outcome)

    run_depth_ordered(targets, depths, build, run, record_error, batch_size)

    for node in payload.get("nodes", []):
        node.pop("tag_source", None)
    payload["vlm_tagging"] = {
        "mode": "open_vocab",
        "model": generator.model_name,
        "pascal_prior": pascal_prior,
        "verify": verify,
        "target_tags": target_tags,
        "min_area": min_area,
        "candidate_nodes": len(targets),
        "processed_nodes": len(rows),
        "updated_nodes": sum(int(row.get("updated", False)) for row in rows),
        "finalized_nodes": sum(int(row.get("accepted", False)) for row in rows),
        "errors": sum(1 for row in rows if "error" in row),
        "annotations": rows,
    }
    return payload, rows
