"""Tree-guided semantic tagging of SVG scene-graph nodes with a Qwen3.5 vision-language model.

Open-vocabulary mode predicts a free-form hierarchical tag (``"car's wheel's tire"``) for each
``unknown``/``texture`` node inside a labeled object subtree; closed-set mode picks one label
from a candidate list. Both process nodes parent-first so earlier tags become context.
"""

from .closed_set import (
    DEFAULT_SETTING,
    SETTINGS,
    build_closed_set_request,
    normalize_candidate_labels,
    ontology_part_labels,
    parse_prediction,
    tag_scene_closed_set,
)
from .open_vocab import build_open_vocab_request, build_verification_request, compose_tag, tag_scene_open_vocab
from .qwen import DEFAULT_MODEL, QwenVLGenerator, extract_json_object, run_with_batch_splitting
from .tree_context import node_depths, node_index, select_target_nodes

__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_SETTING",
    "SETTINGS",
    "QwenVLGenerator",
    "build_closed_set_request",
    "build_open_vocab_request",
    "build_verification_request",
    "compose_tag",
    "extract_json_object",
    "node_depths",
    "node_index",
    "normalize_candidate_labels",
    "ontology_part_labels",
    "parse_prediction",
    "run_with_batch_splitting",
    "select_target_nodes",
    "tag_scene_closed_set",
    "tag_scene_open_vocab",
]
