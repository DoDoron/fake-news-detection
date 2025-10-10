#!/usr/bin/env python3
"""
Merge True.csv and Fake.csv into a single CSV with an added 'label' column.

Defaults assume the dataset path:
  /home/gamejoongsa/hackathon/dataset/fake-news-detection-datasets

Usage examples:
  python scripts/merge_true_fake.py \
    --input-dir /home/gamejoongsa/hackathon/dataset/fake-news-detection-datasets \
    --true-file True.csv --fake-file Fake.csv \
    --output /home/gamejoongsa/hackathon/dataset/fake-news-detection-datasets/merged_true_fake.csv

This script uses only the Python standard library (csv, pathlib, argparse),
so no additional dependencies are required.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List


def read_labeled_rows(csv_path: Path, label_value: str) -> Iterable[Dict[str, str]]:
    """Yield rows from csv_path with an added 'label' field set to label_value.

    Assumes the CSV has headers. Uses utf-8 encoding with newline handling.
    """
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Avoid mutating the original row dict in case the CSV implementation reuses it.
            new_row: Dict[str, str] = dict(row)
            new_row["label"] = label_value
            yield new_row


def determine_fieldnames(true_csv: Path, fake_csv: Path) -> List[str]:
    """Determine unified fieldnames from the two CSV files and append 'label'.

    If headers differ, a union is used preserving order by first occurrence.
    """
    def read_header(p: Path) -> List[str]:
        with p.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                header = []
        return header

    header_true = read_header(true_csv)
    header_fake = read_header(fake_csv)

    seen = set()
    unified: List[str] = []
    for name in header_true + [h for h in header_fake if h not in header_true]:
        if name and name not in seen:
            seen.add(name)
            unified.append(name)

    if "label" not in unified:
        unified.append("label")
    return unified


def merge_csvs(input_dir: Path, true_file: str, fake_file: str, output_path: Path) -> None:
    true_path = input_dir / true_file
    fake_path = input_dir / fake_file

    if not true_path.exists():
        raise FileNotFoundError(f"True CSV not found: {true_path}")
    if not fake_path.exists():
        raise FileNotFoundError(f"Fake CSV not found: {fake_path}")

    fieldnames = determine_fieldnames(true_path, fake_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()

        # Write TRUE rows
        for row in read_labeled_rows(true_path, label_value="true"):
            writer.writerow(row)

        # Write FAKE rows
        for row in read_labeled_rows(fake_path, label_value="fake"):
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    default_input_dir = \
        "/home/gamejoongsa/hackathon/dataset/fake-news-detection-datasets"
    default_output = \
        "/home/gamejoongsa/hackathon/dataset/fake-news-detection-datasets/merged_true_fake.csv"

    parser = argparse.ArgumentParser(
        description=(
            "Merge True.csv and Fake.csv into one CSV with a 'label' column (true/fake)."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(default_input_dir),
        help="Directory containing True.csv and Fake.csv",
    )
    parser.add_argument(
        "--true-file",
        type=str,
        default="True.csv",
        help="Filename for the TRUE-labeled CSV (default: True.csv)",
    )
    parser.add_argument(
        "--fake-file",
        type=str,
        default="Fake.csv",
        help="Filename for the FAKE-labeled CSV (default: Fake.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(default_output),
        help="Output CSV path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merge_csvs(
        input_dir=args.input_dir,
        true_file=args.true_file,
        fake_file=args.fake_file,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()


