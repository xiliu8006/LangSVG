#!/usr/bin/env python3
"""Reconstruction metrics of vectorization outputs (PSNR, SSIM, LPIPS), as in the paper.

PSNR is the diffvg-render PSNR written by the pipeline (``tree.json``); SSIM (11x11 Gaussian
window) and LPIPS (AlexNet) are computed on a cairosvg rendering of ``final.svg`` over white
(falls back to ``final.png`` if cairosvg is not installed).

    python scripts/evaluate.py --outputs outputs/points2304
"""
import argparse
import io
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def load_render(case_dir: Path, size) -> Image.Image:
    try:
        import cairosvg
        png = cairosvg.svg2png(url=str(case_dir / "final.svg"), output_width=size[0], output_height=size[1])
        rgba = Image.open(io.BytesIO(png)).convert("RGBA")
        background = Image.new("RGBA", rgba.size, "white")
        return Image.alpha_composite(background, rgba).convert("RGB")
    except ImportError:
        return Image.open(case_dir / "final.png").convert("RGB")


def to_tensor(image: Image.Image, device) -> torch.Tensor:
    return torch.from_numpy(np.asarray(image, dtype=np.float32)).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0


@torch.no_grad()
def ssim(a: torch.Tensor, b: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> float:
    coords = torch.arange(window_size, device=a.device, dtype=a.dtype) - window_size // 2
    k = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    k = k / k.sum()
    window = torch.outer(k, k)[None, None].expand(a.shape[1], 1, window_size, window_size).contiguous()
    conv = lambda x: F.conv2d(x, window, padding=window_size // 2, groups=a.shape[1])
    mu_a, mu_b = conv(a), conv(b)
    var_a, var_b, cov = conv(a * a) - mu_a ** 2, conv(b * b) - mu_b ** 2, conv(a * b) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return float((((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2))).mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outputs", required=True, help="directory containing one sub-directory per image")
    parser.add_argument("--save", default=None, help="optional JSON file for per-image results")
    parser.add_argument("--no-lpips", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_model = None
    if not args.no_lpips:
        import lpips
        lpips_model = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

    rows = []
    for tree_path in sorted(Path(args.outputs).glob("*/tree.json")):
        scene = json.load(open(tree_path))
        target = Image.open(scene["scene"]["image_path"]).convert("RGB")
        render = load_render(tree_path.parent, target.size)
        a, b = to_tensor(render, device), to_tensor(target, device)
        row = {"case": tree_path.parent.name, "psnr": scene["psnr"], "ssim": ssim(a, b),
               "paths": scene["total_svg_paths"], "points": scene["total_points"]}
        if lpips_model is not None:
            row["lpips"] = float(lpips_model(a * 2 - 1, b * 2 - 1).reshape(-1)[0])
        rows.append(row)
    if not rows:
        raise SystemExit(f"no */tree.json under {args.outputs}")
    keys = [k for k in ("psnr", "ssim", "lpips", "paths", "points") if k in rows[0]]
    mean = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    print(f"{len(rows)} images | " + " | ".join(f"{k}={v:.4f}" for k, v in mean.items()))
    if args.save:
        json.dump({"mean": mean, "per_image": rows}, open(args.save, "w"), indent=2)


if __name__ == "__main__":
    main()
