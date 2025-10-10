#!/usr/bin/env python3
"""CLI entrypoint for training a single experiment."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from core.config import load_training_config
from core.data import available_datasets, build_datasets
from core.train_eval import train_and_evaluate
from core.utils import setup_logging
from model import MODEL_REGISTRY


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a fake news classifier on a dataset")
    parser.add_argument("--dataset_name", required=True, help="Name of the dataset subdirectory")
    parser.add_argument("--model", required=True, choices=MODEL_REGISTRY.keys(), help="Model identifier")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--epochs", type=int, help="Override epochs")
    parser.add_argument("--batch_size", type=int, help="Override batch size")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--max_len", type=int, help="Override maximum sequence length")
    parser.add_argument("--patience", type=int, help="Override patience for early stopping")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--max_vocab_size", type=int, default=20000, help="Maximum vocabulary size")
    parser.add_argument("--run_dir", default="runs", help="Directory to store run artifacts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logging()
    dataset_choices = available_datasets()
    if args.dataset_name not in dataset_choices:
        raise SystemExit(
            f"Dataset '{args.dataset_name}' not found. Available: {', '.join(dataset_choices)}"
        )

    overrides = {
        key: value
        for key, value in {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "max_len": args.max_len,
            "patience": args.patience,
        }.items()
        if value is not None
    }
    config = load_training_config(Path(args.config), overrides)

    loaders, vocab, _, info = build_datasets(
        name=args.dataset_name,
        batch_size=config.batch_size,
        max_len=config.max_len,
        num_workers=args.num_workers,
        max_vocab_size=args.max_vocab_size,
    )
    model_cls = MODEL_REGISTRY[args.model]
    model = model_cls(vocab_size=len(vocab), num_classes=2)
    results, run_dir = train_and_evaluate(
        model,
        loaders,
        config,
        dataset_name=args.dataset_name,
        model_name=args.model,
        run_root=Path(args.run_dir),
    )

    for split, metrics in results.items():
        metric_str = ", ".join(f"{k}={v if v is not None else 'NA'}" for k, v in metrics.items())
        logger.info("%s metrics: %s", split, metric_str)
    logger.info("Artifacts stored in %s", run_dir.resolve())


if __name__ == "__main__":
    main()
