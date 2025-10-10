"""Configuration utilities for loading YAML training configs."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .utils import project_root


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_update(base: Dict[str, Any], overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    if not overrides:
        return result
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result

@dataclass
class TrainingConfig:
    epochs: int = 5
    batch_size: int = 64
    lr: float = 3e-4
    max_len: int = 256
    patience: int = 3
    num_workers: int = 0

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "TrainingConfig":
        return cls(**{k: cfg.get(k, getattr(cls, k)) for k in cls.__annotations__})


def load_training_config(default_path: Optional[Path] = None, overrides: Optional[Dict[str, Any]] = None) -> TrainingConfig:
    cfg_dir = default_path or project_root() / "configs" / "default.yaml"
    data = load_yaml(cfg_dir)
    merged = deep_update(data, overrides)
    return TrainingConfig.from_dict(merged)
