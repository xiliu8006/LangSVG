#!/usr/bin/env bash
# Download the segmentation-prior checkpoints into ./weights (about 9.8 GB in total).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p weights && cd weights
fetch() { [[ -f "$2" ]] || { echo "downloading $2"; wget -q --show-progress -O "$2.part" "$1" && mv "$2.part" "$2"; }; }
fetch https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth sam_vit_h_4b8939.pth
fetch https://github.com/UX-Decoder/Semantic-SAM/releases/download/checkpoint/swinl_only_sam_many2many.pth swinl_only_sam_many2many.pth
fetch https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth groundingdino_swint_ogc.pth
fetch https://huggingface.co/spaces/xinyu1205/Recognize_Anything-Tag2Text/resolve/main/ram_swin_large_14m.pth ram_swin_large_14m.pth
echo "done: $(pwd)"
