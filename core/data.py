"""Dataset loading utilities for fake news datasets.

Each dataset directory is expected to contain `train.csv`, `val.csv`, and `test.csv`
files with `title`, `text`, and `label` columns. Only the `title` and `text` columns are
used as input features; the `label` column is normalised to integer targets.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
from torch.utils.data import DataLoader, Dataset

from . import vocab as vocab_lib
from .utils import dataset_root, detect_delimiter, get_logger, infer_encoding

logger = get_logger()

SUPPORTED_FORMATS = {".csv"}
DEFAULT_TEXT_FIELDS = ["title", "text"]
LABEL_KEYWORDS = ["label", "target", "class", "y"]


@dataclass
class DatasetInfo:
    name: str
    path: Path
    text_fields: List[str] = field(default_factory=list)
    context_fields: List[str] = field(default_factory=list)
    label_field: str = "label"
    metadata: Dict[str, str] = field(default_factory=dict)
    label_mapping: Dict[int, str] = field(default_factory=dict)


class FakeNewsDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        text_fields: List[str],
        tokenizer: vocab_lib.WhitespaceTokenizer,
        vocab: vocab_lib.Vocab,
        max_len: int,
    ) -> None:
        if dataframe.empty:
            raise ValueError("Dataset received an empty DataFrame")
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.max_len = max_len
        self.text_fields = text_fields
        self.texts = [
            compose_record_text(row, self.text_fields)
            for _, row in dataframe.iterrows()
        ]
        self.labels = dataframe["label"].astype(int).tolist()

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        text = self.texts[idx]
        token_ids = self.vocab.encode(text, self.tokenizer)
        return {"input_ids": token_ids, "label": self.labels[idx]}


def compose_record_text(row: pd.Series, text_fields: List[str]) -> str:
    def wrap(tag: str, content: str) -> str:
        stripped = content.strip()
        if stripped:
            return f"<{tag}> {stripped} </{tag}>"
        return f"<{tag}> </{tag}>"

    title_value = ""
    text_value = ""
    if "title" in text_fields:
        value = row.get("title")
        if pd.notna(value):
            title_value = str(value)
    if "text" in text_fields:
        value = row.get("text")
        if pd.notna(value):
            text_value = str(value)

    return f"{wrap('title', title_value)} {wrap('text', text_value)}".strip()


def read_table_file(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported file extension '{ext}' for path {path}")
    encoding = infer_encoding(path)
    with path.open("r", encoding=encoding, errors="ignore") as f:
        sample = "".join([f.readline() for _ in range(5)])
    sep = detect_delimiter(sample)
    try:
        data = pd.read_csv(path, encoding=encoding, sep=sep)
    except Exception:
        data = pd.read_csv(path, encoding=encoding, sep=sep, engine="python", quoting=3)
    data = data.copy()
    data.columns = [col.strip() for col in data.columns]
    drop_cols = [col for col in data.columns if col.lower().startswith("unnamed")]
    if drop_cols:
        data = data.drop(columns=drop_cols)
    return data


def pick_column(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    lookup = {col.lower(): col for col in columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    for col in columns:
        lower = col.lower()
        if any(keyword in lower for keyword in candidates):
            return col
    return None

LABEL_NORMALISATION = {
    "fake": 1,
    "false": 1,
    "fraud": 1,
    "real": 0,
    "true": 0,
    "legit": 0,
}


def normalise_label_value(value) -> int:
    if isinstance(value, (int, float)) and not pd.isna(value):
        return int(round(float(value)))
    if value is None or (isinstance(value, float) and pd.isna(value)):
        raise ValueError("Label value is missing")
    text = str(value).strip().lower()
    if text in LABEL_NORMALISATION:
        return LABEL_NORMALISATION[text]
    if text.isdigit():
        return int(text)
    if text in {"f", "t"}:
        return 1 if text == "f" else 0
    raise ValueError(f"Could not normalise label value: {value}")


def ensure_label_column(
    df: pd.DataFrame, label_field: str = "label"
) -> Tuple[pd.DataFrame, Dict[int, str]]:
    df = df.copy()
    candidate_columns = [label_field] + LABEL_KEYWORDS
    label_col = pick_column(df.columns, candidate_columns)
    if label_col is None:
        raise ValueError("Could not identify label column")
    df = df[df[label_col].notna()]
    mapping: Dict[int, str] = {}

    def convert(value):
        raw = str(value).strip()
        norm = normalise_label_value(value)
        mapping.setdefault(norm, raw)
        return norm

    df["label"] = df[label_col].apply(convert)
    return df, mapping


def load_dataset_frames(dataset_dir: Path) -> Tuple[Dict[str, pd.DataFrame], DatasetInfo]:
    info = DatasetInfo(name=dataset_dir.name, path=dataset_dir)
    splits: Dict[str, pd.DataFrame] = {}

    for split in ["train", "val", "test"]:
        split_path = dataset_dir / f"{split}.csv"
        if not split_path.exists():
            raise FileNotFoundError(f"Expected {split_path} to exist for split '{split}'")
        data = read_table_file(split_path)
        data, mapping = ensure_label_column(data)
        for key, value in mapping.items():
            info.label_mapping.setdefault(key, value.lower())
        splits[split] = data

    primary_df = splits["train"]
    text_fields = [field for field in DEFAULT_TEXT_FIELDS if field in primary_df.columns]
    if not text_fields:
        object_cols = [col for col in primary_df.columns if primary_df[col].dtype == "object" and col != "label"]
        if not object_cols:
            raise ValueError(f"Could not determine text columns for dataset {dataset_dir.name}")
        text_fields = object_cols[:1]
    info.text_fields = text_fields
    info.context_fields = []
    info.label_field = "label"
    return splits, info

def build_datasets(
    name: str,
    batch_size: int,
    max_len: int,
    num_workers: int = 0,
    max_vocab_size: int = 20000,
) -> Tuple[Dict[str, DataLoader], vocab_lib.Vocab, vocab_lib.WhitespaceTokenizer, DatasetInfo]:
    dataset_dir = dataset_root() / name
    splits, info = load_dataset_frames(dataset_dir)
    tokenizer = vocab_lib.WhitespaceTokenizer()
    vocab = vocab_lib.Vocab(max_size=max_vocab_size)
    train_texts = [
        compose_record_text(row, info.text_fields) for _, row in splits["train"].iterrows()
    ]
    train_texts = [text for text in train_texts if text]
    if not train_texts:
        train_texts = [str(row.get("text", "")) for _, row in splits["train"].iterrows()]
    vocab.build(train_texts, tokenizer)

    loaders: Dict[str, DataLoader] = {}
    for split_name, frame in splits.items():
        if frame.empty:
            continue
        dataset = FakeNewsDataset(
            frame,
            info.text_fields,
            tokenizer,
            vocab,
            max_len,
        )
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split_name == "train",
            num_workers=num_workers,
            collate_fn=lambda batch, max_len=max_len: vocab_lib.collate_batch(batch, max_len=max_len),
        )
    return loaders, vocab, tokenizer, info


def available_datasets() -> List[str]:
    return [p.name for p in sorted(dataset_root().iterdir()) if p.is_dir()]


def load_dataset_dataframe(name: str) -> Tuple[pd.DataFrame, DatasetInfo]:
    dataset_dir = dataset_root() / name
    splits, info = load_dataset_frames(dataset_dir)
    frames: List[pd.DataFrame] = []
    for split_name, df in splits.items():
        df = df.copy()
        df["split"] = split_name
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return combined, info
