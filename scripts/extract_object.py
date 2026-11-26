#!/usr/bin/env python3
"""Extract one object of an image as an SVG tree, selected by a click point or a bounding box.

    python scripts/extract_object.py --input examples/ADE_val_00000003.jpg --point 420 300 --output outputs/obj
    python scripts/extract_object.py --input examples/ADE_val_00000003.jpg --box 300 200 560 400 --output outputs/obj

Several prompts of the same image: give --point / --box several times; results go to <output>/obj_<k>/.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langsvg.config import load_config  # noqa: E402
from langsvg.object_tool import ObjectExtractor  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="image file")
    parser.add_argument("--output", required=True, help="output directory")
    parser.add_argument("--point", type=float, nargs=2, action="append", metavar=("X", "Y"), default=[],
                        help="click point on the object (pixels); repeatable")
    parser.add_argument("--box", type=float, nargs=4, action="append", metavar=("X0", "Y0", "X1", "Y1"), default=[],
                        help="bounding box of the object (pixels); repeatable")
    parser.add_argument("--label", default=None, help="tag of the root node (default: object)")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs/natural.yaml"))
    parser.add_argument("--max-paths", type=int, default=None, help="refinement stops at this many paths (default 64)")
    parser.add_argument("--min-mask-pixels", type=int, default=None,
                        help="smallest mask that may initialize a path; refinement stops when none is left (default 20)")
    parser.add_argument("--sam-grid", type=int, default=None, help="SAM point prompts: N x N grid over the object (default 10)")
    parser.add_argument("--semantic-sam-levels", type=int, nargs="+", default=None,
                        help="Semantic-SAM granularity levels inside the object (default: priors.semantic_sam.levels)")
    parser.add_argument("--weights-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not args.point and not args.box:
        parser.error("give --point and/or --box")

    cfg = load_config(args.config, args.weights_dir, seed=args.seed)
    overrides = {"max_paths": args.max_paths, "min_mask_pixels": args.min_mask_pixels, "sam_grid": args.sam_grid,
                 "semantic_sam_levels": args.semantic_sam_levels}
    cfg["object_tool"] = dict(cfg.get("object_tool") or {}, **{k: v for k, v in overrides.items() if v is not None})
    tool = ObjectExtractor(cfg, device=args.device)

    prompts = [{"point": p} for p in args.point] + [{"box": b} for b in args.box]
    results = []
    for k, prompt in enumerate(prompts):
        out_dir = Path(args.output) / (f"obj_{k}" if len(prompts) > 1 else "")
        result = tool.extract(args.input, str(out_dir), label=args.label, **prompt)
        results.append(dict(prompt=prompt, **result))
    print(json.dumps(results if len(results) > 1 else results[0], indent=2))


if __name__ == "__main__":
    main()
