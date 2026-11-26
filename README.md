# Lang-SVG: Hierarchical Image Vectorization with Language Priors

Lang-SVG turns a raster image into a **tree-structured SVG**. Every `<path>` is a node of a
semantic tree (object → part → subpart) with a tag, a parent and children, and this tree is
stored directly in the SVG file.

1. **Tree construction.** Open-vocabulary masks (Grounded-SAM: RAM → GroundingDINO → SAM) give
   labeled object nodes. Multi-granularity masks (Semantic-SAM levels 2/3, SAM) are then inserted
   coarse-to-fine by spatial containment.
2. **Tree-guided optimization.** Depth-adaptive polygon simplification, structure + visual
   fitting with [diffvg](https://github.com/BachiLi/diffvg), **L1-SAM-Prompting** and residual
   refinement under a fixed point budget, and **tree-guided pruning / merging**.
3. **Tree-guided semantic labeling.** Qwen3.5 tags the remaining `unknown` nodes from the image,
   the node crop, its color/geometry and the tags of its ancestors, siblings and children
   (open-vocabulary or closed-set).

---

## Quick start

```bash
# 1. environment (details in "Installation" below)
conda create -n langsvg python=3.10 -y && conda activate langsvg
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
git clone --recursive https://github.com/BachiLi/diffvg && (cd diffvg && pip install svgwrite svgpathtools cssutils numba torch-tools visdom && python setup.py install)

# 2. checkpoints (~9.8 GB, into ./weights)
bash scripts/download_weights.sh

# 3. vectorize the example image (run from the repository root)
python scripts/vectorize.py --input examples/ADE_val_00000003.jpg --output outputs/example
```

This writes `outputs/example/ADE_val_00000003/final.svg` (2304 points, `configs/natural.yaml`).
On one A100 it takes about 6 minutes, and the log ends with a line like

```
[lang-svg] ADE_val_00000003.jpg: paths=352 points=2300 psnr=28.1781
```

Reference results for this image are in `examples/expected/natural/` and
`examples/expected/natural_spanfit/`. Your PSNR may differ by about ±0.3 dB (see [Reproducibility](#reproducibility)).

---

## Installation

Vectorization needs Python 3.10 and a CUDA GPU. We tested on A100 (40 GB / 80 GB) and L40S (48 GB).

```bash
conda create -n langsvg python=3.10 -y && conda activate langsvg
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# diffvg (differentiable rasterizer), built from source
git clone --recursive https://github.com/BachiLi/diffvg && cd diffvg
pip install svgwrite svgpathtools cssutils numba torch-tools visdom
python setup.py install && cd ..

# checkpoints: SAM ViT-H, Semantic-SAM Swin-L, GroundingDINO Swin-T, RAM Swin-L (~9.8 GB)
bash scripts/download_weights.sh
```

Semantic-SAM, GroundingDINO and RAM are vendored (trimmed) in `langsvg/third_party`, so you
do not need detectron2 or any custom CUDA op. If a compiled `MultiScaleDeformableAttention` /
`groundingdino._C` is installed it is used; otherwise the PyTorch implementation is. On first use
GroundingDINO / RAM download `bert-base-uncased` and Semantic-SAM downloads a CLIP tokenizer
from the Hugging Face Hub. To keep checkpoints elsewhere, pass `--weights-dir /path/to/weights`.

Tagging uses a separate environment, see [Semantic labeling](#semantic-labeling-with-qwen35).

---

## Image vectorization

### Command line

```bash
# one image, default config (configs/natural.yaml) and default budget (2304 points)
python scripts/vectorize.py --input image.jpg --output outputs/

# a directory of images (.jpg .jpeg .png .bmp .webp), or a .txt file with one image path per line
python scripts/vectorize.py --input images/ --output outputs/
python scripts/vectorize.py --input list.txt --output outputs/

# the three point budgets of the paper in one call
python scripts/vectorize.py --input images/ --output outputs/ --point-budget 576 1152 2304

# span-fit config: fewer paths at the same point budget (see "Choosing a config")
python scripts/vectorize.py --input images/ --output outputs/ --config configs/natural_spanfit.yaml
```

| option | meaning |
|---|---|
| `--input` | an image, a directory of images, or a text file with one image path per line |
| `--output` | output root. With one budget, results go to `<output>/<image stem>/`. With several budgets, they go to `<output>/points<B>/<image stem>/` |
| `--config` | `configs/natural.yaml` (default) or `configs/natural_spanfit.yaml` |
| `--point-budget` | total number of path points, one or more values (default: `point_budget` in the config, 2304). The paper uses 576 / 1152 / 2304 |
| `--dataset-gt` | GT object / part masks of a COCO-style part dataset instead of Grounded-SAM (see below) |
| `--masks` | your own labeled masks instead of Grounded-SAM (see below) |
| `--seed` | random seed (default: `seed: 0` in the config) |
| `--weights-dir` | checkpoint directory (default: `./weights`) |
| `--device` | default `cuda` |
| `--skip-existing` | skip images whose `tree.json` already exists (useful to resume a batch) |

The point budget is the main knob. The number of paths follows from it (roughly
`budget / 8` paths), and more points give higher fidelity at the cost of a larger SVG.
The initial semantic tree may use at most `init_budget_ratio` (0.8) of the budget: if it would
need more, the smallest unlabeled masks are dropped (the background and labeled objects are
always kept). The remaining points go to the refinement paths.

The models print a lot of log output (e.g. `... is tied`, `$UNUSED$ criterion.empty_weight`,
deprecation warnings); this is expected. Each image ends with one
`[lang-svg] <image>: paths=... points=... psnr=...` line.

### Outputs

For each image (`<output>/<stem>/`):

| file | content |
|---|---|
| `final.svg` | the vectorized image. Each `<path>` has `id`, `data-node-id`, `data-tag`, `data-level`, `data-parent-id` |
| `final.png` | diffvg rendering of `final.svg` |
| `tree.json` | scene graph: per node tag, level, parent/children, render order, full/visible masks (RLE), bbox, color; plus `psnr`, `total_svg_paths`, `total_points` |
| `init.svg`, `init.png`, `init_tree.json` | the initial tree before refinement |

### Initializing the tree with ground-truth masks

For part datasets with GT annotations (ADE20K-Part234, PascalPart116 layout), the GT object and part
masks can replace Grounded-SAM. Set `priors.dataset_gt.enabled: true` (and `instance_json`) in the
config, or pass the instance json on the command line:

```bash
ADE=/path/to/ADE20KPart234
python scripts/vectorize.py --input $ADE/images/validation --output outputs/ade_gt --point-budget 2304 \
    --config configs/natural_spanfit.yaml --dataset-gt $ADE/ade20k_instance_val.json
```

- **Objects** are the non-crowd annotations of `ade20k_instance_val.json` (`car`, `bus`, ...).
- **Parts** are read from the part label PNGs (`annotations_detectron2_part/validation/<image>.png`):
  for each object, the pixels inside the object mask with one of its `part_category_id`s
  (`car's wheel`, ...). Part names come from `ade20k_joint_val.json`. Set `joint_json` / `part_dir`
  in the config if your files are elsewhere.
- Every GT mask is a labeled node with its GT tag and annotation id. GT nodes are **never pruned,
  merged or dropped** by the budget. Semantic-SAM, SAM and the refinement add the unlabeled nodes
  as usual.
- `tree.json` also stores the dataset's label lists as `dataset_vlm_ontology`, so closed-set
  tagging uses the 234 ADE part labels without `--candidate-labels`.

Images without GT annotations fall back to having no labeled objects (a message is printed).

### Using your own masks

`--masks` replaces Grounded-SAM (the open-vocabulary prior) with your own labeled masks, for
example ground truth or a part segmenter. Semantic-SAM and SAM are still used for the unlabeled
fine nodes. Pass a JSON file, or a directory with one `<image stem>.json` per image. Each file
holds a list of

```json
{"segmentation": "<COCO RLE or polygons>", "label": "car" | "car's wheel",
 "level": "object" | "part", "id": 3, "parent_id": 1}
```

### Python API

```python
from langsvg.config import load_config
from langsvg.pipeline import LangSVG

model = LangSVG(load_config("configs/natural.yaml"), device="cuda")   # loads the priors once
scene = model.vectorize("examples/ADE_val_00000003.jpg", "outputs/example", point_budget=1152)
print(scene["psnr"], scene["total_svg_paths"], scene["total_points"])
```

`vectorize` can be called repeatedly on the same `model` (the segmentation models are loaded
once).

---

## Choosing a config

| config | when to use |
|---|---|
| `configs/natural.yaml` (default) | **To align with the NeurIPS paper.** This is the paper version: quadratic segments, straight (polygon) initialization. Use it to reproduce or compare with the paper. |
| `configs/natural_spanfit.yaml` | **When you want fewer paths with comparable reconstruction fidelity** (recommended for new applications). |

`natural_spanfit` changes only how a mask becomes a Bézier path (everything else is inherited
from `natural.yaml`):

- **Initial tree: cubic span fit.** Each cubic segment spans 3 polygon edges, and its 2 inner
  control points are least-squares fitted to the mask contour.
- **Refinement paths: quadratic span fit.** Each segment spans 2 polygon edges, with 1 fitted
  control point.
- **Anchors.** The vertices of a coarse polygon (approxPolyDP tolerance 30) are always segment
  endpoints, so sharp corners such as the four corners of the background stay sharp.
- Finer polygons (`epsilon: 3`, `max_polygon_edges: 20` instead of 6 / 10), because span fit
  uses about one path point per polygon vertex.

What this means in practice, at the same point budget:

- **More of the budget goes to the semantic tree.** The cubic span-fit paths of the
  object / part / subpart nodes follow the mask contours closely, so a larger share of the
  points is spent on these nodes and less on the visual-refinement paths that fill in details.
  On ADE20K at 576 points, tree nodes hold 52% of the final points with `natural_spanfit`
  versus 44% with `natural`.
- **Fewer paths.** About 12–14% fewer paths at 576 points; the smooth cubic paths give a cleaner,
  easier-to-edit SVG.
- **Comparable reconstruction fidelity on natural photos.** PSNR is within about 0.2 dB of
  `natural`; the largest gap is at 576 points, where there are very few paths.

---

## Semantic labeling with Qwen3.5

`vectorize.py` already tags the object nodes found by Grounded-SAM (e.g. `car`, `bus`). The
finer nodes (parts, subparts) are left as `unknown`. `scripts/tag_svg_tree.py` labels those nodes
with Qwen3.5, parent-first, so each new tag is context for the nodes below it.

### Environment

Qwen3.5 needs `transformers >= 5.2`, which conflicts with the vectorization environment, so
tagging uses its own environment:

```bash
conda create -n langsvg-tag python=3.10 -y && conda activate langsvg-tag
pip install torch --index-url https://download.pytorch.org/whl/cu128   # any CUDA build of torch >= 2.4
pip install -r requirements-tagging.txt
```

The default model `Qwen/Qwen3.5-27B` is downloaded from the Hugging Face Hub on first use. It
needs about 55–60 GB of GPU memory in bfloat16. With `--device-map auto` (the default) it is split
across all visible GPUs, e.g. 2 × 40/48 GB. To use a local copy, pass its directory to `--model`.

### Run

Tagging reads the `tree.json` files written by `vectorize.py`:

```bash
conda activate langsvg-tag

# example image from the quick start: open-vocabulary hierarchical tags ("car's wheel")
python scripts/tag_svg_tree.py --scene-json outputs/example/ADE_val_00000003/tree.json --mode open

# every tree.json under a directory (searched recursively)
python scripts/tag_svg_tree.py --scene-json outputs/ --mode open

# closed-set: each node gets one label from your list (.txt, one label per line), or stays unknown
python scripts/tag_svg_tree.py --scene-json outputs/ --mode closed --candidate-labels labels.txt
```

Two modes:

- **`--mode open`** (default): open-vocabulary tags, composed with the parent's tag
  (`wheel` under `car` becomes `car's wheel`). `--verify` adds a second keep-or-revise pass per
  node.
- **`--mode closed`**: the model picks one label from `--candidate-labels` (for `--dataset-gt` outputs
  the default is the dataset's part labels). A prediction that is
  not in the list leaves the node unchanged.

| option | meaning |
|---|---|
| `--scene-json` | one `tree.json`, or a directory searched recursively for `tree.json` |
| `--output` | output file (single scene only). Default: `tree.tagged.json` next to the input |
| `--overwrite` | re-tag scenes whose output already exists (default: skip them) |
| `--target-tags` | node tags to (re)label, default `unknown,texture` |
| `--allow-root-targets` | also tag nodes that are not inside a labeled object (by default only nodes under a Grounded-SAM / `--masks` object are tagged) |
| `--min-area`, `--max-nodes` | skip nodes whose visible bbox area is below 64 px²; tag at most 256 nodes per scene |
| `--model` | Hugging Face id or local directory (default `Qwen/Qwen3.5-27B`) |
| `--batch-size` | nodes per generation batch (default 1) |

### Output

`tree.tagged.json` is a copy of `tree.json` with updated nodes. Each tagged node has its new
`tag` / `level`, and the previous values are kept in `original_tag` / `original_level`.
`final.svg` is not modified (its `data-tag` attributes keep the vectorizer's tags). The log
prints one line per scene:

```
[tag_svg_tree] .../tree.json: <n> nodes, <k> updated, 0 errors -> .../tree.tagged.json
```

---

## Evaluation

```bash
python scripts/evaluate.py --outputs outputs/points2304 --save outputs/points2304/eval.json
```

This prints the mean PSNR / SSIM / LPIPS (AlexNet) / paths / points over all `*/tree.json` in
the directory. `--save` writes per-image values, and `--no-lpips` skips LPIPS.

### Reproducibility

The optimization is stochastic (random structure colors, GPU atomics in diffvg). `seed: 0`
makes runs repeatable on the same hardware; across GPUs expect about ±0.3 dB PSNR per image.

---

## Repository layout

```
configs/
  natural.yaml          # NeurIPS paper version (default)
  natural_spanfit.yaml  # span-fit paths: fewer paths, comparable fidelity
examples/
  ADE_val_00000003.jpg  # example image for the quick start
  expected/             # reference final.svg / final.png for both configs (2304 points)
langsvg/
  pipeline.py           # LangSVG: image -> SVG tree (main loop)
  priors.py             # Grounded-SAM, Semantic-SAM, SAM, external-mask priors
  tree.py               # mask tree construction / insertion / re-layering
  budget.py             # point budget, depth-adaptive simplification
  svg.py                # mask -> Bezier path, colors, diffvg rendering
  curve_fit.py          # span fit (batched least-squares control points)
  anchor_span_fit.py    # span-fit anchors from a coarse polygon
  optimize.py           # structure / visual optimization
  residual.py           # residual (L1) candidate regions
  masks.py              # mask registry and mask utilities
  export.py             # scene-graph JSON + SVG node attributes
  tagging/              # Qwen3.5 tree-guided tagging (open-vocab and closed-set)
  third_party/          # vendored, trimmed Semantic-SAM, GroundingDINO, RAM (see their LICENSE files)
scripts/
  download_weights.sh   # prior checkpoints -> weights/
  vectorize.py          # images -> SVG trees
  tag_svg_tree.py       # tag unknown nodes of SVG trees with Qwen3.5
  evaluate.py           # PSNR / SSIM / LPIPS
  eval_ade_tag_recovery.py  # ADE20K hidden-tag recovery benchmark
data/ade20k_tag_recovery/   # sample manifest for that benchmark
```

## Citation

```bibtex
@inproceedings{langsvg2026,
  title     = {Lang-SVG: Hierarchical Image Vectorization with Language Priors},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## Acknowledgements

Built on [Layered Vectorization](https://github.com/SZUVIZ/layered_vectorization),
[diffvg](https://github.com/BachiLi/diffvg), [SAM](https://github.com/facebookresearch/segment-anything),
[Semantic-SAM](https://github.com/UX-Decoder/Semantic-SAM),
[Grounded-SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything) /
[GroundingDINO](https://github.com/IDEA-Research/GroundingDINO),
[Recognize Anything](https://github.com/xinyu1205/recognize-anything) and
[Qwen3.5](https://github.com/QwenLM). The example image is from the ADE20K validation set.
