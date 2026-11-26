# --------------------------------------------------------
# Semantic-SAM: Segment and Recognize Anything at Any Granularity
# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Feng Li (fliay@connect.ust.hk)
# Trimmed for Lang-SVG: model construction only.
# --------------------------------------------------------
from pathlib import Path

from langsvg.third_party.semantic_sam.utils.arguments import load_opt_from_config_file
from langsvg.third_party.semantic_sam.semantic_sam.BaseModel import BaseModel
from langsvg.third_party.semantic_sam.semantic_sam.architectures import build_model
from langsvg.third_party.semantic_sam.tasks.automatic_mask_generator import SemanticSamAutomaticMaskGenerator

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"
CONFIGS = {"L": CONFIG_DIR / "semantic_sam_only_sa-1b_swinL.yaml"}


def build_semantic_sam(model_type: str, ckpt: str, device="cuda"):
    """Build Semantic-SAM (SwinL many-to-many) and load `ckpt`."""
    opt = load_opt_from_config_file(str(CONFIGS[model_type]))
    return BaseModel(opt, build_model(opt)).from_pretrained(ckpt).eval().to(device)
