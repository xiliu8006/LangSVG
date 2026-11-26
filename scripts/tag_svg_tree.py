#!/usr/bin/env python3
"""Tag unknown/texture nodes of Lang-SVG scene graphs with Qwen3.5.

Examples:
    # open-vocabulary hierarchical tags for one scene
    python scripts/tag_svg_tree.py --scene-json outputs/img/tree.json --mode open

    # closed-set tags from a label file, for every scene under a directory
    python scripts/tag_svg_tree.py --scene-json out/ --mode closed --candidate-labels labels.txt

Writes ``<name>.tagged.json`` next to each input (or ``--output`` for a single scene).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langsvg.tagging import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_SETTING,
    SETTINGS,
    QwenVLGenerator,
    tag_scene_closed_set,
    tag_scene_open_vocab,
)


def find_scene_jsons(path: Path, name: str) -> List[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob(name) if p.is_file())


def load_candidate_labels(path: Optional[str]) -> Optional[List[str]]:
    """A JSON list or a text file with one label per line."""
    if not path:
        return None
    text = Path(path).read_text(encoding="utf-8")
    if path.endswith(".json"):
        return list(json.loads(text))
    return [line.strip() for line in text.splitlines() if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene-json", required=True, help="Scene JSON file, or a directory searched recursively")
    parser.add_argument("--scene-json-name", default="tree.json", help="File name to find in a directory")
    parser.add_argument("--image", default=None, help="Image override (single scene); default: scene.image_path")
    parser.add_argument("--output", default=None, help="Output path (single scene); default: <name>.tagged.json")
    parser.add_argument("--overwrite", action="store_true", help="Re-tag scenes whose output already exists")
    parser.add_argument("--mode", choices=["open", "closed"], default="open")
    parser.add_argument("--candidate-labels", default=None, help="Closed mode: candidate labels (.txt, one per line, or a .json list); default: the scene's dataset_vlm_ontology (written by vectorize.py --dataset-gt)")
    parser.add_argument("--setting", default=DEFAULT_SETTING, choices=sorted(SETTINGS), help="Closed mode context setting")
    parser.add_argument("--pascal-prior", action="store_true", help="Open mode: PascalPart116 object categories as soft prior")
    parser.add_argument("--verify", action="store_true", help="Open mode: add a keep-or-revise verification pass")
    parser.add_argument("--target-tags", default="unknown,texture", help="Comma-separated tags to (re)label")
    parser.add_argument("--allow-root-targets", action="store_true", help="Also tag nodes outside labeled object subtrees")
    parser.add_argument("--min-area", type=int, default=64)
    parser.add_argument("--max-nodes", type=int, default=256)
    parser.add_argument("--crop-margin-ratio", type=float, default=None, help="Default: 0.12 (open), 0.15 (closed)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF model id or local snapshot path")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-implementation", default="")
    parser.add_argument("--max-pixels", type=int, default=401408)
    parser.add_argument("--min-pixels", type=int, default=None)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenes = find_scene_jsons(Path(args.scene_json), args.scene_json_name)
    if not scenes:
        raise SystemExit(f"No scene JSON found at {args.scene_json}")
    if len(scenes) > 1 and (args.image or args.output):
        raise SystemExit("--image/--output require a single scene JSON")
    outputs = [Path(args.output) if args.output else p.with_name(p.stem + ".tagged.json") for p in scenes]
    todo = [(s, o) for s, o in zip(scenes, outputs) if args.overwrite or not o.exists()]
    print(f"[tag_svg_tree] {len(todo)} scene(s) to tag ({len(scenes) - len(todo)} already done)", flush=True)
    if not todo:
        return

    generator = QwenVLGenerator(
        model=args.model,
        dtype=args.dtype,
        device_map=args.device_map,
        max_pixels=args.max_pixels,
        min_pixels=args.min_pixels,
        attn_implementation=args.attn_implementation,
        enable_thinking=args.enable_thinking,
    )
    candidate_labels = load_candidate_labels(args.candidate_labels)
    common = {
        "target_tags": [tag for tag in args.target_tags.split(",") if tag.strip()],
        "min_area": args.min_area,
        "max_nodes": args.max_nodes,
        "require_object_ancestor": not args.allow_root_targets,
        "batch_size": args.batch_size,
        "generation": {
            "max_new_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
        },
    }
    for scene_path, output_path in todo:
        payload = json.loads(scene_path.read_text(encoding="utf-8"))
        image_path = args.image or (payload.get("scene") or {}).get("image_path")
        if not image_path or not Path(image_path).exists():
            print(f"[tag_svg_tree] skip {scene_path}: image not found ({image_path}); pass --image", flush=True)
            continue
        image = Image.open(image_path).convert("RGB")
        if args.mode == "open":
            payload, rows = tag_scene_open_vocab(
                payload,
                image,
                generator,
                image_name=Path(image_path).name,
                crop_margin_ratio=0.12 if args.crop_margin_ratio is None else args.crop_margin_ratio,
                pascal_prior=args.pascal_prior,
                verify=args.verify,
                **common,
            )
        else:
            payload, rows = tag_scene_closed_set(
                payload,
                image,
                generator,
                candidate_labels=candidate_labels,
                crop_margin_ratio=0.15 if args.crop_margin_ratio is None else args.crop_margin_ratio,
                setting=args.setting,
                **common,
            )
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        updated = sum(int(bool(row.get("updated", row.get("accepted")))) for row in rows)
        errors = sum(1 for row in rows if "error" in row)
        print(f"[tag_svg_tree] {scene_path}: {len(rows)} nodes, {updated} updated, {errors} errors -> {output_path}", flush=True)


if __name__ == "__main__":
    main()
