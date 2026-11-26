"""Export the SVG tree: scene-graph JSON (read by the tagger) and SVG node attributes."""
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from .masks import MaskRegistry, bbox_area, bbox_xyxy, encode_rle
from .svg import render_node_mask

SCHEMA_VERSION = "vectorization_semantic_scene_graph_v1"


def _level(meta: dict, layer_index: int) -> str:
    if meta.get("level") is not None:
        return str(meta["level"])
    tag = str(meta.get("tag", "unknown"))
    if tag == "background" or layer_index == 0:
        return "background"
    if tag and tag.lower() != "unknown":
        return "part" if "'s" in tag else "object"
    return "unknown"


def _source_summary(layers: list, registry: MaskRegistry) -> dict:
    sources = Counter(str(registry.meta(m).get("source", "unknown")).lower() for layer in layers for m in layer)
    return {"total_svg_paths": int(sum(sources.values())), "by_source": dict(sorted(sources.items()))}


def build_scene_graph(layers: list, registry: MaskRegistry, shapes, groups, width: int, height: int, device,
                      image_path: str, path_cfg: dict, history: Optional[List[dict]] = None) -> dict:
    """One node per SVG path (in render order) with tag, level, parent/children, and full/visible masks."""
    nodes, node_of_mask, parent_ref = [], {}, {}
    index = 0
    for layer_index, layer in enumerate(layers):
        for mask_index, mask in enumerate(layer):
            meta = dict(registry.meta(mask))
            node_id = f"svg_{index}"
            tag = str(meta.get("tag", "unknown"))
            nodes.append({
                "node_id": node_id, "svg_element_id": node_id, "shape_index": index,
                "layer_index": layer_index, "mask_index": mask_index,
                "tag": tag, "level": _level(meta, layer_index), "tag_source": str(meta.get("source", "unknown")),
                "proposal_source": meta.get("proposal_source"), "proposal_refiner": meta.get("proposal_refiner"),
                "candidate_source": meta.get("candidate_source"),
                "status": "known" if tag.lower() != "unknown" else "unknown",
                "confidence": meta.get("confidence"), "annotation_id": meta.get("annotation_id"),
                "_source_mask": mask,
            })
            node_of_mask[id(mask)] = node_id
            parent_ref[node_id] = meta.get("layer_parent_mask_id")
            index += 1

    background_id = next((n["node_id"] for n in nodes if n["level"].lower() == "background"), None)
    by_id = {n["node_id"]: n for n in nodes}
    for node in nodes:
        ref = parent_ref[node["node_id"]]
        parent = node_of_mask.get(ref) if ref is not None else None
        node["parent_id"] = None if node["level"].lower() == "background" and parent is None else (parent or background_id)
        node["children_ids"] = []
    roots = []
    for node in nodes:
        if node["parent_id"] is None:
            roots.append(node["node_id"])
        elif node["parent_id"] in by_id:
            by_id[node["parent_id"]]["children_ids"].append(node["node_id"])

    for node in nodes:
        full = render_node_mask(width, height, shapes, groups, node["shape_index"], device)
        visible = render_node_mask(width, height, shapes, groups, node["shape_index"], device, visible_only=True)
        source = np.where(node.pop("_source_mask") > 0, 255, 0).astype(np.uint8)
        box, vbox, sbox = bbox_xyxy(full), bbox_xyxy(visible), bbox_xyxy(source)
        path = shapes[node["shape_index"]]
        cps = [int(v) for v in path.num_control_points.detach().cpu().tolist()]
        node.update({
            "render_order": node["shape_index"],
            "bbox_xyxy": box, "bbox_area": bbox_area(box), "mask_pixels": int(cv2.countNonZero(full)),
            "source_bbox_xyxy": sbox, "source_bbox_area": bbox_area(sbox), "source_mask_pixels": int(cv2.countNonZero(source)),
            "source_segmentation": encode_rle(source),
            "visible_bbox_xyxy": vbox, "visible_bbox_area": bbox_area(vbox), "visible_mask_pixels": int(cv2.countNonZero(visible)),
            "segmentation": encode_rle(full), "visible_segmentation": encode_rle(visible),
            "path_complexity": {"segment_count": len(cps), "total_control_points": int(sum(cps)),
                                "point_tensor_size": int(path.points.shape[0])},
            "fill_rgba": [round(float(v), 6) for v in groups[node["shape_index"]].fill_color.detach().cpu().tolist()],
        })

    return {
        "schema_version": SCHEMA_VERSION,
        "scene": {"image_path": str(image_path), "image_size": [int(width), int(height)]},
        "roots": roots,
        "nodes": nodes,
        "total_svg_paths": len(nodes),
        "total_points": int(sum(int(s.points.shape[0]) for s in shapes)),
        "path_parameterization_config": {"control_points_per_segment": int(path_cfg["control_points_per_segment"])},
        "svg_source_summary": _source_summary(layers, registry),
        "refinement_history": history or [],
    }


def annotate_svg(svg_path: str, scene: dict) -> None:
    """Write node id / tag / level / parent attributes onto the SVG path elements."""
    tree = ET.parse(svg_path)
    elements = [e for e in tree.getroot().iter() if e.tag.split("}")[-1] in {"path", "circle", "ellipse", "rect", "polygon"}]
    for element, node in zip(elements, sorted(scene["nodes"], key=lambda n: int(n["shape_index"]))):
        element.set("id", str(node["svg_element_id"]))
        element.set("data-node-id", str(node["node_id"]))
        element.set("data-tag", str(node.get("tag", "unknown")))
        element.set("data-level", str(node.get("level", "unknown")))
        if node.get("parent_id") is not None:
            element.set("data-parent-id", str(node["parent_id"]))
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    tree.write(svg_path, encoding="unicode", xml_declaration=True)


def save_json(path: str, payload: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
