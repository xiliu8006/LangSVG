# Lang-SVG: Hierarchical Image Vectorization with Language Priors

Lang-SVG converts an image into a **tree-structured SVG**: every `<path>` is a node of a semantic
tree (object → part → subpart) with a tag, a parent and children.

1. **Tree construction**: Grounded-SAM gives labeled objects; Semantic-SAM and SAM masks are inserted coarse-to-fine.
2. **Tree-guided optimization**: [diffvg](https://github.com/BachiLi/diffvg) fitting, L1-SAM-Prompting refinement and tree-guided pruning under a point budget.
3. **Tree-guided labeling**: Qwen3.5 tags the remaining `unknown` nodes using their ancestors, siblings and children.

## Installation

Python 3.10 and a CUDA GPU (tested on A100 and L40S).

```bash
conda create -n langsvg python=3.10 -y && conda activate langsvg
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# diffvg
git clone --recursive https://github.com/BachiLi/diffvg && cd diffvg
pip install svgwrite svgpathtools cssutils numba torch-tools visdom
python setup.py install && cd ..

# checkpoints (SAM, Semantic-SAM, GroundingDINO, RAM; ~9.8 GB) -> ./weights
bash scripts/download_weights.sh
```

## Vectorization

```bash
python scripts/vectorize.py --input path/to/image.jpg --output path/to/outputs
```

| option | meaning |
|---|---|
| `--input` | an image, a directory of images, or a `.txt` file with one image path per line |
| `--output` | output directory; results go to `<output>/<image name>/` (`<output>/points<B>/<image name>/` with several budgets) |
| `--config` | `configs/natural.yaml` (default) or `configs/natural_spanfit.yaml` |
| `--point-budget` | total path points, one or more values (default 2304; the paper uses 576 1152 2304) |
| `--masks` | your own labeled masks instead of Grounded-SAM |
| `--dataset-gt` | GT object/part masks of an ADE20K-Part234-style dataset instead of Grounded-SAM |
| `--skip-existing` | skip images that already have a `tree.json` |

Outputs per image:

| file | content |
|---|---|
| `final.svg` | the SVG; each `<path>` has `data-node-id`, `data-tag`, `data-level`, `data-parent-id` |
| `final.png` | rendering of `final.svg` |
| `tree.json` | scene graph (tags, hierarchy, masks, bboxes, colors) and `psnr`, `total_svg_paths`, `total_points` |

Python API:

```python
from langsvg.config import load_config
from langsvg.pipeline import LangSVG

model = LangSVG(load_config("configs/natural.yaml"), device="cuda")
scene = model.vectorize("path/to/image.jpg", "path/to/outputs", point_budget=2304)
```

### Configs

| config | use |
|---|---|
| `natural.yaml` | the NeurIPS paper version (quadratic segments); use it to reproduce the paper |
| `natural_spanfit.yaml` | cubic span-fit paths: ~12% fewer paths at similar PSNR (within ~0.2 dB) |

### Custom masks

`--masks` takes a JSON file (or a directory with one `<image name>.json` per image) with a list of

```json
{"segmentation": "<COCO RLE or polygons>", "label": "car's wheel", "level": "part", "id": 3, "parent_id": 1}
```

`--dataset-gt path/to/ade20k_instance_val.json` uses the GT objects and parts as labeled nodes;
they are never pruned. Part masks and names are read from the dataset files next to it
(`joint_json` / `part_dir` in the config).

## Semantic labeling

Qwen3.5 needs `transformers >= 5.2`, so it uses a separate environment. The default model
`Qwen/Qwen3.5-27B` needs ~60 GB of GPU memory and is split over all visible GPUs.

```bash
conda create -n langsvg-tag python=3.10 -y && conda activate langsvg-tag
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-tagging.txt

# open vocabulary: "wheel" under "car" becomes "car's wheel"
python scripts/tag_svg_tree.py --scene-json path/to/outputs --mode open

# closed set: one label per node from a list (one label per line)
python scripts/tag_svg_tree.py --scene-json path/to/outputs --mode closed --candidate-labels path/to/labels.txt
```

`--scene-json` is a `tree.json` or a directory searched for them. The result is written to
`tree.tagged.json` next to each input; old tags are kept in `original_tag`.

## Evaluation

```bash
# PSNR / SSIM / LPIPS / paths / points
python scripts/evaluate.py --outputs path/to/outputs --save path/to/eval.json

# ADE20K hidden-tag recovery (scenes from vectorize.py --dataset-gt)
python scripts/eval_ade_tag_recovery.py --scenes-root path/to/outputs
```

Results are repeatable with `seed: 0` on the same GPU; across GPUs PSNR varies by about ±0.3 dB.

## Citation

If you find our work helpful for your research or project, please consider citing our paper:

```bibtex
@article{liu2026lang,
  title={Lang-SVG: Hierarchical Image Vectorization with Language Priors},
  author={Liu, Xi and Zhou, Chaoyi and Wang, Run and Li, Jiaang and Luo, Feng and Huang, Junxiang and Huang, Siyu},
  journal={arXiv preprint arXiv:2611.15941},
  year={2026}
}
```

## Acknowledgements

Built on [Layered Vectorization](https://github.com/SZUVIZ/layered_vectorization),
[diffvg](https://github.com/BachiLi/diffvg), [SAM](https://github.com/facebookresearch/segment-anything),
[Semantic-SAM](https://github.com/UX-Decoder/Semantic-SAM),
[Grounded-SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything),
[Recognize Anything](https://github.com/xinyu1205/recognize-anything) and [Qwen3.5](https://github.com/QwenLM).
