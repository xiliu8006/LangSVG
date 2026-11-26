# Semantic-SAM (vendored, trimmed)

Source: https://github.com/UX-Decoder/Semantic-SAM (MIT License, Copyright (c) 2023 Microsoft).

Only the modules needed to run `SemanticSamAutomaticMaskGenerator` with the SwinL
many-to-many checkpoint are kept. Changes for Lang-SVG:

* imports rewritten into the `langsvg.third_party.semantic_sam` namespace;
* detectron2 / fvcore replaced by `d2.py` (code copied from detectron2 and fvcore, Apache-2.0);
* the `MultiScaleDeformableAttention` CUDA op is optional (PyTorch fallback is used if it is not compiled);
* interactive demos, visualizer and FocalNet backbones removed.
