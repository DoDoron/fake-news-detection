#!/usr/bin/env python3
"""
Split a merged CSV (with a 'label' column) into train/val/test CSVs.

Balanced-enough strategy:
- If 'label' exists, perform per-label splits with the given ratios (approximate),
  then concatenate. This avoids extreme imbalance without requiring sklearn.
- If 'label' is missing, do a global random split.

Defaults:
  input:  /home/gamejoongsa/hackathon/dataset/fake-news-detection-datasets/merged.csv
  outputs: train.csv, val.csv, test.csv in the same directory
  ratios: 0.6 / 0.2 / 0.2

Usage example:
  python scripts/split_train_val_test.py \
    --input /home/gamejoongsa/hackathon/dataset/fake-news-detection-datasets/merged.csv \
    --train-out /home/gamejoongsa/hackathon/dataset/fake-news-detection-datasets/train.csv \
    --val-out /home/gamejoongsa/hackathon/dataset/fake-news-detection-datasets/val.csv \
    --test-out /home/gamejoongsa/hackathon/dataset/fake-news-detection-datasets/test.csv \
    --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --seed 42

Only Python standard library is used.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def read_csv_rows(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows: List[Dict[str, str]] = [dict(r) for r in reader]
    return fieldnames, rows


def write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def split_counts(n: int, r_train: float, r_val: float, r_test: float) -> Tuple[int, int, int]:
    if n <= 0:
        return 0, 0, 0
    # Initial floors
    t = int(math.floor(n * r_train))
    v = int(math.floor(n * r_val))
    # Assign remainder to test
    s = t + v
    te = max(0, n - s)
    # Adjust if off by more than 1 due to rounding (rare)
    while t + v + te > n:
        te -= 1
    while t + v + te < n:
        te += 1
    return t, v, te


def per_label_split(
    rows: List[Dict[str, str]],
    label_key: str,
    r_train: float,
    r_val: float,
    r_test: float,
    rng: random.Random,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    by_label: Dict[str, List[Dict[str, str]]] = {}
    for r in rows:
        lbl = r.get(label_key, "")
        by_label.setdefault(lbl, []).append(r)

    train_rows: List[Dict[str, str]] = []
    val_rows: List[Dict[str, str]] = []
    test_rows: List[Dict[str, str]] = []

    for lbl, group in by_label.items():
        rng.shuffle(group)
        n = len(group)
        n_train, n_val, n_test = split_counts(n, r_train, r_val, r_test)
        train_rows.extend(group[:n_train])
        val_rows.extend(group[n_train : n_train + n_val])
        test_rows.extend(group[n_train + n_val : n_train + n_val + n_test])

    # Final shuffle to mix labels
    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    rng.shuffle(test_rows)
    return train_rows, val_rows, test_rows


def global_split(
    rows: List[Dict[str, str]],
    r_train: float,
    r_val: float,
    r_test: float,
    rng: random.Random,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    rng.shuffle(rows)
    n = len(rows)
    n_train, n_val, n_test = split_counts(n, r_train, r_val, r_test)
    train_rows = rows[:n_train]
    val_rows = rows[n_train : n_train + n_val]
    test_rows = rows[n_train + n_val : n_train + n_val + n_test]
    return train_rows, val_rows, test_rows


def parse_args() -> argparse.Namespace:
    default_input = \
        "/home/gamejoongsa/hackathon/dataset/fake-news-detection-datasets/merged.csv"
    default_train = \
        "/home/gamejoongsa/hackathon/dataset/fake-news-detection-datasets/train.csv"
    default_val = \
        "/home/gamejoongsa/hackathon/dataset/fake-news-detection-datasets/val.csv"
    default_test = \
        "/home/gamejoongsa/hackathon/dataset/fake-news-detection-datasets/test.csv"

    p = argparse.ArgumentParser(
        description="Split merged CSV into train/val/test with approximate per-label balance."
    )
    p.add_argument("--input", type=Path, default=Path(default_input), help="Input merged CSV path")
    p.add_argument("--train-out", type=Path, default=Path(default_train), help="Output train CSV path")
    p.add_argument("--val-out", type=Path, default=Path(default_val), help="Output val CSV path")
    p.add_argument("--test-out", type=Path, default=Path(default_test), help="Output test CSV path")
    p.add_argument("--label-col", type=str, default="label", help="Label column name (default: label)")
    p.add_argument("--train-ratio", type=float, default=0.6, help="Train ratio (default: 0.6)")
    p.add_argument("--val-ratio", type=float, default=0.2, help="Val ratio (default: 0.2)")
    p.add_argument("--test-ratio", type=float, default=0.2, help="Test ratio (default: 0.2)")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Basic ratio validation
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if not (0.99 <= total_ratio <= 1.01):
        raise ValueError(
            f"Ratios must sum to 1.0 (got {total_ratio:.3f}). Adjust --train-ratio/--val-ratio/--test-ratio."
        )

    fieldnames, rows = read_csv_rows(args.input)
    rng = random.Random(args.seed)

    if args.label_col in fieldnames:
        train_rows, val_rows, test_rows = per_label_split(
            rows, args.label_col, args.train_ratio, args.val_ratio, args.test_ratio, rng
        )
    else:
        train_rows, val_rows, test_rows = global_split(
            rows, args.train_ratio, args.val_ratio, args.test_ratio, rng
        )

    write_csv(args.train_out, fieldnames, train_rows)
    write_csv(args.val_out, fieldnames, val_rows)
    write_csv(args.test_out, fieldnames, test_rows)


if __name__ == "__main__":
    main()



