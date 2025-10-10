#!/usr/bin/env python3
"""Quick benchmarking script across all datasets and models."""
from __future__ import annotations

import argparse
import traceback
from pathlib import Path
import sys
from typing import Dict, List

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from core.config import load_training_config
from core.data import available_datasets, build_datasets
from core.train_eval import train_and_evaluate
from core.utils import setup_logging
from model import MODEL_REGISTRY


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all model/dataset combinations for a quick benchmark")
    parser.add_argument("--config", default="configs/default.yaml", help="Base YAML config path")
    parser.add_argument("--epochs", type=int, default=2, help="Epochs for quick evaluation")
    parser.add_argument("--batch_size", type=int, help="Override batch size")
    parser.add_argument("--max_len", type=int, help="Override maximum sequence length")
    parser.add_argument("--run_dir", default="runs", help="Root directory for runs")
    parser.add_argument("--output", default="sweep_results.csv", help="Path to aggregate CSV output")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--max_vocab_size", type=int, default=20000, help="Vocabulary cap")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logging()
    datasets = available_datasets()
    run_root = Path(args.run_dir)

    if not datasets:
        raise SystemExit("No datasets found in dataset/ directory")

    results: List[Dict[str, object]] = []
    for dataset_name in datasets:
        for model_name, model_cls in MODEL_REGISTRY.items():
            overrides = {
                key: value
                for key, value in {
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "max_len": args.max_len,
                }.items()
                if value is not None
            }
            config = load_training_config(Path(args.config), overrides)
            logger.info(
                "Running quick eval | dataset=%s model=%s epochs=%d",
                dataset_name,
                model_name,
                config.epochs,
            )
            try:
                loaders, vocab, _, _ = build_datasets(
                    name=dataset_name,
                    batch_size=config.batch_size,
                    max_len=config.max_len,
                    num_workers=args.num_workers,
                    max_vocab_size=args.max_vocab_size,
                )
                model = model_cls(vocab_size=len(vocab), num_classes=2)
                metrics, run_dir = train_and_evaluate(
                    model,
                    loaders,
                    config,
                    dataset_name=dataset_name,
                    model_name=model_name,
                    run_root=run_root,
                )
                val_metrics = metrics.get("val", {})
                result_row = {
                    "dataset": dataset_name,
                    "model": model_name,
                    "status": "ok",
                }
                for key, value in val_metrics.items():
                    result_row[f"val_{key}"] = value
                result_row["run_dir"] = str(run_dir)
                results.append(result_row)
            except Exception as exc:
                logger.error(
                    "Run failed | dataset=%s model=%s error=%s",
                    dataset_name,
                    model_name,
                    exc,
                )
                logger.debug(traceback.format_exc())
                results.append(
                    {
                        "dataset": dataset_name,
                        "model": model_name,
                        "status": "failed",
                        "error": str(exc),
                    }
                )
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    df = pd.DataFrame(results)
    output_path = Path(args.output)
    df.to_csv(output_path, index=False)
    logger.info("Sweep complete -> %s", output_path.resolve())


if __name__ == "__main__":
    main()
