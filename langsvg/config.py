"""Configuration loading. Checkpoint paths are resolved relative to ``weights_dir``."""
import copy
from pathlib import Path
from typing import Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str, weights_dir: Optional[str] = None, **overrides) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    base = cfg.pop("base", None)
    if base:  # shallow inheritance: `base: natural.yaml`
        parent = load_config(str(Path(path).parent / base), weights_dir)
        cfg = _merge(parent, cfg)
    weights = Path(weights_dir or cfg.get("weights_dir", "weights"))
    if not weights.is_absolute():
        weights = REPO_ROOT / weights
    cfg["checkpoints"] = {k: str(v if Path(v).is_absolute() else weights / v) for k, v in cfg["checkpoints"].items()}
    for key, value in overrides.items():
        if value is not None:
            cfg[key] = value
    return cfg


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        out[key] = _merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out
