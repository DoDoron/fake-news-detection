"""Tiny Transformer encoder for text classification."""
from __future__ import annotations

import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim))
        pe = torch.zeros(max_len, embedding_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TinyTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int = 2,
        embedding_dim: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.position = PositionalEncoding(embedding_dim, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.padding_idx = padding_idx

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        mask = input_ids == self.padding_idx
        embeddings = self.embedding(input_ids)
        embeddings = self.position(embeddings)
        encoded = self.encoder(embeddings, src_key_padding_mask=mask)
        mask_float = (~mask).unsqueeze(-1).to(encoded.dtype)
        summed = (encoded * mask_float).sum(dim=1)
        lengths = lengths.clamp(min=1).unsqueeze(-1).to(encoded.dtype)
        pooled = summed / lengths
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits
