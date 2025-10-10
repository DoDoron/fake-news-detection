"""Utility helpers for the fake news detection baseline framework."""
from __future__ import annotations

import csv
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

_LOGGER_NAME = "fake_news"


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure root logger used across the project."""
    logger = logging.getLogger(_LOGGER_NAME)
    if logger.handlers:
        # Already configured – just update the level
        logger.setLevel(level)
        return logger

    logger.setLevel(level)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger(_LOGGER_NAME)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def dataset_root() -> Path:
    return project_root() / "dataset"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def read_text(path: Path, encoding: str = "utf-8") -> str:
    with path.open("r", encoding=encoding) as f:
        return f.read()


def find_readme(directory: Path) -> Optional[Path]:
    candidates = [
        directory / name
        for name in ("README.md", "README.txt", "readme.md", "readme.txt")
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    # Fallback: first text/markdown file in directory
    for candidate in directory.glob("*.md"):
        return candidate
    for candidate in directory.glob("*.txt"):
        return candidate
    return None


def list_dataset_dirs(root: Optional[Path] = None) -> List[Path]:
    base = root or dataset_root()
    if not base.exists():
        raise FileNotFoundError(f"Dataset root not found: {base}")
    return sorted([p for p in base.iterdir() if p.is_dir()])


def detect_delimiter(sample: str) -> str:
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
        return dialect.delimiter
    except Exception:
        if sample.count(";") > sample.count(","):
            return ";"
        if sample.count("\t") > 0:
            return "\t"
        return ","


def infer_encoding(path: Path) -> str:
    # Simple heuristic – default utf-8, fallback to latin-1.
    try:
        path.open("r", encoding="utf-8").read(1024)
        return "utf-8"
    except UnicodeDecodeError:
        return "latin-1"


def save_json(data, path: Path, indent: int = 2) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def as_path(path_like: os.PathLike | str) -> Path:
    return Path(path_like).expanduser().resolve()


def chunk(iterable: Iterable, size: int) -> Iterable[List]:
    batch: List = []
    for item in iterable:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch
