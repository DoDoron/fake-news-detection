"""Bag-of-words style MLP classifier."""
from __future__ import annotations

import torch
from torch import nn


class BoWMLP(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int = 2,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.pad_idx = padding_idx

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(input_ids)
        mask = (input_ids != self.pad_idx).unsqueeze(-1)
        masked_embeddings = embeddings * mask
        lengths = lengths.clamp(min=1).unsqueeze(-1).to(embeddings.dtype)
        pooled = masked_embeddings.sum(dim=1) / lengths
        pooled = self.dropout(pooled)
        logits = self.fc(pooled)
        return logits
