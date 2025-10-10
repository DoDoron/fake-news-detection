#!/usr/bin/env python3
"""Few-shot evaluation script without additional training on target datasets."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from core.config import load_training_config
from core.train_eval import train_and_evaluate, evaluate
from core.utils import ensure_dir, get_logger, setup_logging
from core import data as data_utils
from core.vocab import WhitespaceTokenizer, Vocab, collate_batch
from model import MODEL_REGISTRY

logger = get_logger()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Few-shot cross-dataset evaluation")
    parser.add_argument("--train_dataset", required=True, help="Dataset used for training")
    parser.add_argument("--model", required=True, choices=MODEL_REGISTRY.keys(), help="Model identifier")
    parser.add_argument("--few_shot_k", type=int, default=100, help="Number of samples for few-shot eval")
    parser.add_argument("--train_max_samples", type=int, default=None, help="Optional cap on training samples")
    parser.add_argument(
        "--train_sample_mode",
        choices=["random", "head"],
        default="random",
        help="Sampling strategy when limiting training data",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to base YAML config")
    parser.add_argument("--epochs", type=int, help="Override epochs")
    parser.add_argument("--batch_size", type=int, help="Override batch size")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--max_len", type=int, help="Override maximum sequence length")
    parser.add_argument("--patience", type=int, help="Override early stopping patience")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--device", default=None, help="Torch device (e.g. cuda, cpu)")
    parser.add_argument("--max_vocab_size", type=int, default=20000, help="Vocabulary cap")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_dataframe(df: pd.DataFrame, limit: Optional[int], mode: str, seed: int) -> pd.DataFrame:
    if limit is None or limit <= 0 or len(df) <= limit:
        return df
    if mode == "head":
        return df.head(limit).reset_index(drop=True)
    return df.sample(n=limit, random_state=seed).reset_index(drop=True)


def make_loader(
    df: pd.DataFrame,
    info: data_utils.DatasetInfo,
    tokenizer: WhitespaceTokenizer,
    vocab: Vocab,
    batch_size: int,
    max_len: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    dataset = data_utils.FakeNewsDataset(df, info.text_fields, tokenizer, vocab, max_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda batch, ml=max_len: collate_batch(batch, max_len=ml),
    )


def select_eval_split(splits: Dict[str, pd.DataFrame]) -> Optional[Tuple[str, pd.DataFrame]]:
    df = splits.get("test")
    if df is None or df.empty:
        return None
    return "test", df.reset_index(drop=True)


def evaluate_split(
    model: nn.Module,
    loader: Optional[DataLoader],
    criterion: nn.Module,
    device: torch.device,
) -> Optional[Dict[str, float]]:
    if loader is None:
        return None
    loss, metrics, _ = evaluate(model, loader, criterion, device)
    metrics = dict(metrics)
    metrics["loss"] = loss
    return metrics


def main() -> None:
    args = parse_args()
    setup_logging()
    set_seed(args.seed)

    overrides = {
        key: value
        for key, value in {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "max_len": args.max_len,
            "patience": args.patience,
            "num_workers": args.num_workers,
        }.items()
        if value is not None
    }
    config = load_training_config(Path(args.config), overrides)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    logger.info("Training dataset: %s | model=%s", args.train_dataset, args.model)

    # Load training dataset frames
    train_splits, train_info = data_utils.load_dataset_frames(data_utils.dataset_root() / args.train_dataset)
    train_df = train_splits["train"].copy()
    val_df = train_splits["val"].copy()
    test_df = train_splits["test"].copy()

    train_df = sample_dataframe(train_df, args.train_max_samples, args.train_sample_mode, args.seed)

    tokenizer = WhitespaceTokenizer()
    vocab = Vocab(max_size=args.max_vocab_size)
    train_texts = [data_utils.compose_record_text(row, train_info.text_fields) for _, row in train_df.iterrows()]
    vocab.build(train_texts, tokenizer)

    loaders = {
        "train": make_loader(train_df, train_info, tokenizer, vocab, config.batch_size, config.max_len, True, config.num_workers),
        "val": make_loader(val_df, train_info, tokenizer, vocab, config.batch_size, config.max_len, False, config.num_workers),
        "test": make_loader(test_df, train_info, tokenizer, vocab, config.batch_size, config.max_len, False, config.num_workers),
    }

    model_cls = MODEL_REGISTRY[args.model]
    model = model_cls(vocab_size=len(vocab), num_classes=2)

    results, run_dir = train_and_evaluate(
        model,
        loaders,
        config,
        dataset_name=args.train_dataset,
        model_name=args.model,
        run_root=Path("runs"),
        device=device,
    )
    run_dir = Path(run_dir)

    criterion = nn.CrossEntropyLoss()

    # In-domain evaluation (always test split)
    in_domain_loader = loaders.get("test")
    in_domain_split = "test"
    in_domain_metrics = evaluate_split(model, in_domain_loader, criterion, device)

    few_shot_rows: List[Dict[str, object]] = []
    if in_domain_metrics:
        few_shot_rows.append(
            {
                "dataset": args.train_dataset,
                "split": in_domain_split,
                "in_domain": True,
                **in_domain_metrics,
            }
        )

    # Few-shot evaluations
    all_datasets = data_utils.available_datasets()
    for dataset_name in all_datasets:
        if dataset_name == args.train_dataset:
            continue
        try:
            splits, info = data_utils.load_dataset_frames(data_utils.dataset_root() / dataset_name)
        except Exception as exc:
            logger.warning("Skipping dataset %s due to load error: %s", dataset_name, exc)
            continue

        selection = select_eval_split(splits)
        if not selection:
            logger.warning("Skipping dataset %s because test split is unavailable or empty", dataset_name)
            continue
        split_name, labeled_df = selection
        sample_df = sample_dataframe(labeled_df, args.few_shot_k, "random", args.seed)
        if sample_df.empty:
            logger.warning("Dataset %s split %s has no samples after sampling; skipping", dataset_name, split_name)
            continue

        loader = make_loader(sample_df, info, tokenizer, vocab, config.batch_size, config.max_len, False, config.num_workers)
        metrics = evaluate_split(model, loader, criterion, device)
        if metrics is None:
            logger.warning("Metrics unavailable for dataset %s; skipping", dataset_name)
            continue
        few_shot_rows.append(
            {
                "dataset": dataset_name,
                "split": split_name,
                "in_domain": False,
                **metrics,
            }
        )

    if not few_shot_rows:
        logger.warning("No evaluation results produced")
        return

    results_df = pd.DataFrame(few_shot_rows)
    results_json = results_df.to_dict(orient="records")

    output_csv = run_dir / "few_shot_results.csv"
    output_json = run_dir / "few_shot_results.json"
    ensure_dir(output_csv.parent)
    results_df.to_csv(output_csv, index=False)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(results_json, f, ensure_ascii=False, indent=2)

    display_cols = [col for col in ["dataset", "split", "in_domain", "accuracy", "precision", "recall", "f1", "auroc"] if col in results_df.columns]
    print("\nFew-shot evaluation summary:\n")
    print(results_df[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
