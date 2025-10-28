"""Training and evaluation utilities."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from .config import TrainingConfig
from .metrics import compute_classification_metrics, plot_confusion_matrix
from .utils import ensure_dir, get_logger, save_json, timestamp

logger = get_logger()


def build_run_directory(base_dir: Path, dataset_name: str, model_name: str) -> Path:
    run_dir = base_dir / dataset_name / model_name / timestamp()
    ensure_dir(run_dir)
    return run_dir


def move_batch_to_device(batch, device: torch.device):
    inputs, lengths, labels = batch
    return inputs.to(device), lengths.to(device), labels.to(device)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_norm: float = 5.0,
) -> float:
    model.train()
    running_loss = 0.0
    total_examples = 0
    for batch in loader:
        inputs, lengths, labels = move_batch_to_device(batch, device)
        optimizer.zero_grad()
        logits = model(inputs, lengths)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        running_loss += loss.item() * labels.size(0)
        total_examples += labels.size(0)
    return running_loss / max(total_examples, 1)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, float], Dict[str, list]]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    y_true: list = []
    y_pred: list = []
    y_prob: list = []
    with torch.no_grad():
        for batch in loader:
            inputs, lengths, labels = move_batch_to_device(batch, device)
            logits = model(inputs, lengths)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            total_examples += labels.size(0)
            probabilities = torch.softmax(logits, dim=-1)[:, 1]
            predictions = torch.argmax(logits, dim=-1)
            y_true.extend(labels.tolist())
            y_pred.extend(predictions.tolist())
            y_prob.extend(probabilities.tolist())
    average_loss = total_loss / total_examples if total_examples > 0 else float("nan")
    metrics = compute_classification_metrics(y_true, y_pred, y_prob)
    return average_loss, metrics, {"y_true": y_true, "y_pred": y_pred, "y_prob": y_prob}


def train_and_evaluate(
    model: nn.Module,
    loaders: Dict[str, DataLoader],
    config: TrainingConfig,
    dataset_name: str,
    model_name: str,
    run_root: Path = Path("runs"),
    device: Optional[torch.device] = None,
) -> Tuple[Dict[str, Dict[str, Optional[float]]], Path]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = build_run_directory(run_root, dataset_name, model_name)
    ensure_dir(run_dir)
    ensure_dir(run_dir / "artifacts")

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=config.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=1, factor=0.5)

    history: list = []
    best_f1 = -1.0
    best_state = None
    patience_counter = 0

    train_loader = loaders.get("train")
    val_loader = loaders.get("val")
    test_loader = loaders.get("test")
    if not train_loader:
        raise ValueError("Training loader is required")

    # GPU 정보 출력
    if device.type == "cuda":
        logger.info(
            "Using CUDA device: %s | %s | Memory: %.2f GB",
            torch.cuda.get_device_name(0),
            device,
            torch.cuda.get_device_properties(0).total_memory / 1024**3,
        )
    else:
        logger.info("Using CPU device (CUDA not available)")
    
    logger.info(
        "Starting training | dataset=%s model=%s epochs=%d batch_size=%d",
        dataset_name,
        model_name,
        config.epochs,
        config.batch_size,
    )
    model.to(device)

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        eval_loader = val_loader or train_loader
        val_loss, val_metrics, _ = evaluate(model, eval_loader, criterion, device)
        scheduler_metric = val_metrics.get("f1", 0.0)
        if math.isnan(scheduler_metric):
            scheduler_metric = 0.0
        scheduler.step(scheduler_metric)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        history.append(row)

        logger.info(
            "Epoch %d | train_loss=%.4f val_loss=%.4f val_f1=%.4f",
            epoch,
            train_loss,
            val_loss,
            val_metrics.get("f1", 0.0),
        )

        current_f1 = val_metrics.get("f1", float("nan"))
        if not math.isnan(current_f1) and current_f1 > best_f1:
            best_f1 = current_f1
            patience_counter = 0
            best_state = {"model": model.state_dict(), "epoch": epoch}
            torch.save(model.state_dict(), run_dir / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                logger.info("Early stopping triggered at epoch %d", epoch)
                break

    metrics_path = run_dir / "metrics.csv"
    pd.DataFrame(history).to_csv(metrics_path, index=False)

    if best_state:
        model.load_state_dict(best_state["model"])

    final_results: Dict[str, Dict[str, float]] = {}
    split_outputs: Dict[str, Dict[str, list]] = {}
    for split_name, loader in ("train", train_loader), ("val", val_loader), ("test", test_loader):
        if loader is None:
            continue
        loss, metrics, outputs = evaluate(model, loader, criterion, device)
        metrics = dict(metrics)
        metrics["loss"] = loss
        final_results[split_name] = _sanitize(metrics)
        split_outputs[split_name] = outputs

    best_metrics_path = run_dir / "best_metrics.json"
    save_json({split: metrics for split, metrics in final_results.items()}, best_metrics_path)

    if "val" in split_outputs and split_outputs["val"]["y_true"]:
        plot_confusion_matrix(
            split_outputs["val"]["y_true"],
            split_outputs["val"]["y_pred"],
            run_dir / "confusion_matrix.png",
        )

    config_path = run_dir / "config.json"
    save_json(config.__dict__, config_path)

    logger.info("Training complete | best_val_f1=%.4f run_dir=%s", best_f1, run_dir)
    return final_results, run_dir


def _sanitize(values: Dict[str, float]) -> Dict[str, Optional[float]]:
    sanitized: Dict[str, Optional[float]] = {}
    for key, value in values.items():
        if isinstance(value, float) and math.isnan(value):
            sanitized[key] = None
        else:
            sanitized[key] = float(value)
    return sanitized
