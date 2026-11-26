#!/usr/bin/env python3
"""Vectorize images into semantic SVG trees.

    python scripts/vectorize.py --input examples/ --output outputs/ --point-budget 2304
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langsvg.config import load_config  # noqa: E402
from langsvg.pipeline import LangSVG  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="image file, directory of images, or a text file listing image paths")
    parser.add_argument("--output", required=True, help="output root; results go to <output>/<image stem>/")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs/natural.yaml"))
    parser.add_argument("--point-budget", type=int, nargs="+", default=None,
                        help="one or more point budgets (default: config value), e.g. 576 1152 2304")
    parser.add_argument("--weights-dir", default=None)
    parser.add_argument("--masks", default=None,
                        help="optional labeled masks JSON (file, or directory with <image stem>.json) "
                             "replacing Grounded-SAM as the open-vocabulary prior")
    parser.add_argument("--dataset-gt", default=None, metavar="INSTANCE_JSON",
                        help="use the GT object / part masks of this COCO-style instance json instead of Grounded-SAM "
                             "(sets priors.dataset_gt.enabled and .instance_json)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    source = Path(args.input)
    if source.is_dir():
        images = sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    elif source.suffix.lower() not in IMAGE_SUFFIXES:  # text file with one image path per line
        images = [Path(line.strip()) for line in source.read_text().splitlines() if line.strip()]
    else:
        images = [source]

    cfg = load_config(args.config, args.weights_dir, seed=args.seed)
    if args.dataset_gt:
        cfg["priors"]["dataset_gt"] = dict(cfg["priors"].get("dataset_gt") or {}, enabled=True, instance_json=args.dataset_gt)
    model = LangSVG(cfg, device=args.device)
    budgets = args.point_budget or [None]
    for budget in budgets:
        root = Path(args.output) / (f"points{budget}" if len(budgets) > 1 else "")
        for image in images:
            out_dir = root / image.stem
            if args.skip_existing and (out_dir / "tree.json").exists():
                continue
            masks = None
            if args.masks:
                masks_path = Path(args.masks)
                masks = str(masks_path / f"{image.stem}.json") if masks_path.is_dir() else str(masks_path)
            model.vectorize(str(image), str(out_dir), external_masks=masks, point_budget=budget)


if __name__ == "__main__":
    main()
