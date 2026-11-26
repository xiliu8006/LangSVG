#!/usr/bin/env python3
"""ADE20KPart234 hidden-tag recovery benchmark (closed-set tagging, paper setting A4 by default).

For each of the 431 masked part/subpart nodes in the manifest, the node's tag (and the tags of the
other masked nodes of the same scene) is hidden and Qwen3.5 must pick it from the 234 ADE20KPart234
part labels. Reports exact / leaf / parent accuracy over valid predictions and exact accuracy
over all samples.

Example:
    python scripts/eval_ade_tag_recovery.py \\
        --scenes-root outputs/ade_gt/points2304 --output-dir outputs/ade_tag_recovery_a4

Scenes (``--scenes-root``):

* ``tree.json`` outputs of ``scripts/vectorize.py --dataset-gt <ade20k_instance_val.json>`` (one
  ``<image_id>/tree.json`` per image, the default). The GT part nodes keep their annotation id, so each
  sample is found by the manifest's ``gt_annotation_id``; the same GT parts are hidden as in the paper.
* the paper's scene graphs: ``--scene-json-name final_render_with_tags.json`` (one
  ``<image_id>_<image_id>/final_render_with_tags.json`` per image); samples are found by node id. The
  scene JSON path is part of the prompt metadata, so pass the same root string to reproduce a run
  token-for-token. Batches are formed within ``--num-shards`` rank-strided shards (default 2, the
paper's 2-GPU layout); pass ``--shard-index`` to run a single shard per process/GPU.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from langsvg.tagging import (  # noqa: E402
    DEFAULT_MODEL,
    SETTINGS,
    QwenVLGenerator,
    build_closed_set_request,
    normalize_candidate_labels,
    ontology_part_labels,
    parse_prediction,
    run_with_batch_splitting,
)
from langsvg.tagging.tree_context import label_leaf, label_parent, node_index, normalize_label, safe_int  # noqa: E402

DEFAULT_MANIFEST = REPO_ROOT / "data" / "ade20k_tag_recovery" / "masked_subparts_manifest.json"
PAPER_SCENE_JSON_NAME = "final_render_with_tags.json"


def load_samples(manifest: Dict[str, Any], scenes_root: str, images_root: Optional[str],
                 scene_json_name: str = "tree.json") -> List[Dict[str, Any]]:
    """Resolve manifest keys to scene nodes, in evaluation order.

    Paper scene graphs are matched by node id; ``tree.json`` outputs by GT annotation id. A sample whose
    node is missing or not visible enough gets an ``error`` (it is reported, not skipped)."""
    by_node_id = scene_json_name == PAPER_SCENE_JSON_NAME
    rows_by_image = {row["image_id"]: row for row in manifest["masked_subparts_by_image"]}
    scenes: Dict[str, Dict[str, Any]] = {}
    samples = []
    for key in manifest["selected_sample_keys"]:
        image_id, paper_node_id = key.split(":", 1)
        row = rows_by_image[image_id]
        scene_dir = row["scene_dir"] if by_node_id else image_id
        scene_json_path = os.path.join(scenes_root, scene_dir, scene_json_name)
        if scene_json_path not in scenes:
            with open(scene_json_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            by_annotation = {str(n.get("annotation_id")): str(n["node_id"]) for n in payload.get("nodes", [])
                             if n.get("annotation_id") is not None}
            scenes[scene_json_path] = (payload, by_annotation)
        payload, by_annotation = scenes[scene_json_path]

        def node_id_of(item: Dict[str, Any]) -> Optional[str]:
            if by_node_id:
                return str(item["node_id"])
            return by_annotation.get(str(item.get("gt_annotation_id")))

        item = next(it for it in row["masked_subparts"] if it["sample_key"] == key)
        node = node_index(payload).get(node_id_of(item) or "")
        image_path = (payload.get("scene") or {}).get("image_path")
        if images_root:
            image_path = os.path.join(images_root, row["image_file"])
        error = None
        if node is None:
            error = f"{key}: GT part {item.get('gt_annotation_id')} not found in {scene_json_path}"
            node = {"node_id": None, "parent_id": None, "tag": item["label"], "level": item["level"]}
        gt_label = normalize_label(node.get("tag"))
        level = str(node.get("level", "")).strip().lower()
        if error is None and (level not in set(manifest["target_levels"]) or gt_label in {"", "unknown", "background"}):
            error = f"{key}: node is not a labeled part/subpart (level={level}, tag={gt_label})"
        if error is None and safe_int(node.get("visible_bbox_area")) < manifest["min_area"]:
            error = f"{key}: visible bbox area below min_area"
        if error is not None and by_node_id:
            raise ValueError(error)
        samples.append(
            {
                "sample_key": key,
                "image_id": image_id,
                "scene_json_path": scene_json_path,
                "image_path": image_path,
                "payload": payload,
                "node": node,
                "gt_label": gt_label,
                "gt_level": level,
                "masked_node_ids": {nid for nid in (node_id_of(it) for it in row["masked_subparts"]) if nid},
                "error": error,
            }
        )
    return samples


def candidate_labels_for(sample: Dict[str, Any]) -> List[str]:
    """All ontology part labels of the scene (the ground truth is always included)."""
    labels = normalize_candidate_labels(ontology_part_labels(sample["payload"])) or [sample["gt_label"]]
    if sample["gt_label"] not in labels:
        labels = sorted(set(labels) | {sample["gt_label"]})
    return labels


def build_item(sample: Dict[str, Any], setting: str, crop_margin_ratio: float) -> Dict[str, Any]:
    image = Image.open(sample["image_path"]).convert("RGB")
    candidates = candidate_labels_for(sample)
    request = build_closed_set_request(
        sample["node"],
        node_index(sample["payload"]),
        image,
        candidates,
        hidden_ids=sample["masked_node_ids"],
        setting=setting,
        image_id=sample["image_id"],
        scene_json_path=sample["scene_json_path"],
        crop_margin_ratio=crop_margin_ratio,
    )
    return {"sample": sample, "candidates": candidates, "request": request}


def base_row(sample: Dict[str, Any], setting: str) -> Dict[str, Any]:
    node = sample["node"]
    return {
        "sample_key": sample["sample_key"],
        "image_id": sample["image_id"],
        "node_id": node.get("node_id"),
        "parent_id": node.get("parent_id"),
        "gt_label": sample["gt_label"],
        "gt_level": sample["gt_level"],
        "setting": setting,
    }


def compute_summary(rows: List[Dict[str, Any]], total: int) -> Dict[str, Any]:
    """Exact / leaf / parent accuracy over valid predictions, plus exact accuracy over all samples."""
    valid = [row for row in rows if row.get("prediction_valid")]
    errors = [row for row in rows if row.get("error")]
    denom = len(valid) or 1
    per_class: Dict[str, Dict[str, Any]] = {}
    confusions: collections.Counter = collections.Counter()
    for row in valid:
        entry = per_class.setdefault(row["gt_label"], {"label": row["gt_label"], "support": 0, "correct": 0})
        entry["support"] += 1
        entry["correct"] += int(row["exact_match"])
        if not row["exact_match"]:
            confusions[(row["gt_label"], row["predicted_label"])] += 1
    for entry in per_class.values():
        entry["accuracy"] = entry["correct"] / entry["support"]
    exact = sum(int(row["exact_match"]) for row in valid)
    return {
        "samples": total,
        "evaluated": len(rows),
        "valid_predictions": len(valid),
        "invalid_predictions": len(rows) - len(valid),
        "errors": len(errors),
        "exact_correct": exact,
        "exact_accuracy": exact / denom,
        "exact_accuracy_all_samples": exact / (total or 1),
        "leaf_correct": sum(int(row["leaf_match"]) for row in valid),
        "leaf_accuracy": sum(int(row["leaf_match"]) for row in valid) / denom,
        "parent_correct": sum(int(row["parent_match"]) for row in valid),
        "parent_accuracy": sum(int(row["parent_match"]) for row in valid) / denom,
        "per_class": sorted(per_class.values(), key=lambda item: (-item["support"], item["label"])),
        "top_confusions": [
            {"gt_label": gt, "predicted_label": pred, "count": count} for (gt, pred), count in confusions.most_common(50)
        ],
        "error_details": [{"sample_key": row["sample_key"], "error": row["error"]} for row in errors],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenes-root", required=True, help="Directory with <image_id>/tree.json (vectorize.py --dataset-gt)")
    parser.add_argument("--scene-json-name", default="tree.json",
                        help=f"tree.json (default) or {PAPER_SCENE_JSON_NAME} for the paper's scene graphs")
    parser.add_argument("--images-root", default=None, help="Override image directory (default: scene.image_path)")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output-dir", default="outputs/ade_tag_recovery")
    parser.add_argument("--setting", default="a4_full_hierarchy_color", choices=sorted(SETTINGS))
    parser.add_argument("--num-shards", type=int, default=2, help="Rank-strided shards; batches never cross shards")
    parser.add_argument("--shard-index", type=int, default=None, help="Run only this shard (default: all, in order)")
    parser.add_argument("--resume", action="store_true", help="Skip samples already in the predictions file")
    parser.add_argument("--dry-run", action="store_true", help="Write the requests (no model) and exit")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF model id or local snapshot path")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-implementation", default="")
    parser.add_argument("--max-pixels", type=int, default=401408)
    parser.add_argument("--min-pixels", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--crop-margin-ratio", type=float, default=0.15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.manifest, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    samples = load_samples(manifest, args.scenes_root, args.images_root, args.scene_json_name)
    shard_ids = range(args.num_shards) if args.shard_index is None else [args.shard_index]
    shards = [samples[i :: args.num_shards] for i in shard_ids]
    os.makedirs(args.output_dir, exist_ok=True)
    suffix = "" if args.shard_index is None else f"_shard{args.shard_index}of{args.num_shards}"

    if args.dry_run:
        path = os.path.join(args.output_dir, f"requests_{args.setting}{suffix}.jsonl")
        with open(path, "w", encoding="utf-8") as handle:
            for shard in shards:
                for sample in shard:
                    if sample["error"]:
                        continue
                    request = build_item(sample, args.setting, args.crop_margin_ratio)["request"]
                    handle.write(json.dumps({"sample_key": sample["sample_key"], "user_payload": request["user_payload"]}, ensure_ascii=False) + "\n")
        print(f"[eval_ade_tag_recovery] dry run: wrote {path}", flush=True)
        return

    predictions_path = os.path.join(args.output_dir, f"predictions_{args.setting}{suffix}.jsonl")
    rows: List[Dict[str, Any]] = []
    if args.resume and os.path.exists(predictions_path):
        with open(predictions_path, "r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    done: Set[str] = {row["sample_key"] for row in rows}

    generator = QwenVLGenerator(
        model=args.model,
        dtype=args.dtype,
        device_map=args.device_map,
        max_pixels=args.max_pixels,
        min_pixels=args.min_pixels,
        attn_implementation=args.attn_implementation,
    )
    generation = {"max_new_tokens": args.max_tokens, "temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k}
    total = sum(len(shard) for shard in shards)

    def write(row: Dict[str, Any]) -> None:
        rows.append(row)
        with open(predictions_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        status = row.get("error") or f"pred={row['predicted_label']} exact={row['exact_match']}"
        print(f"[eval_ade_tag_recovery] [{len(rows)}/{total}] {row['sample_key']} gt={row['gt_label']} {status}", flush=True)

    def run(batch: List[Dict[str, Any]]) -> None:
        start = time.time()
        outcomes = run_with_batch_splitting(
            lambda items: generator.generate_json_batch([item["request"] for item in items], **generation),
            batch,
            generator.clear_cache,
            single_item_retries=1,
        )
        elapsed = time.time() - start
        for item, outcome in zip(batch, outcomes):
            row = base_row(item["sample"], args.setting)
            if isinstance(outcome, BaseException):
                row.update({"prediction_valid": False, "error": str(outcome)})
            else:
                (result, raw), batch_count = outcome
                label, valid = parse_prediction(result, item["candidates"])
                gt = row["gt_label"]
                row.update(
                    {
                        "prediction_valid": valid,
                        "predicted_label": label,
                        "exact_match": bool(valid and normalize_label(label) == gt),
                        "leaf_match": bool(valid and label_leaf(label) == label_leaf(gt)),
                        "parent_match": bool(valid and label_parent(label) == label_parent(gt)),
                        "confidence": result.get("confidence"),
                        "reason": str(result.get("reason", "")).strip(),
                        "raw_response": raw,
                        "batch_size": batch_count,
                        "candidate_count": len(item["candidates"]),
                    }
                )
            row["batch_elapsed_seconds"] = elapsed
            write(row)

    for shard in shards:
        pending: List[Dict[str, Any]] = []
        for sample in shard:
            if sample["sample_key"] in done:
                continue
            if sample["error"]:
                write({**base_row(sample, args.setting), "prediction_valid": False, "error": sample["error"]})
                continue
            try:
                pending.append(build_item(sample, args.setting, args.crop_margin_ratio))
            except Exception as exc:  # noqa: BLE001 - a broken sample is an error row, not a crash
                write({**base_row(sample, args.setting), "prediction_valid": False, "error": str(exc)})
                continue
            if len(pending) >= max(1, args.batch_size):
                run(pending)
                pending = []
        run(pending)

    summary = compute_summary(rows, total)
    summary["config"] = {key: value for key, value in vars(args).items()}
    summary_path = os.path.join(args.output_dir, f"summary_{args.setting}{suffix}.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(
        f"[eval_ade_tag_recovery] {args.setting}: exact {summary['exact_correct']}/{summary['valid_predictions']} valid "
        f"= {summary['exact_accuracy']:.4f} ({summary['exact_accuracy_all_samples']:.4f} of all {total}); "
        f"leaf {summary['leaf_accuracy']:.4f}; parent {summary['parent_accuracy']:.4f}; summary={summary_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
