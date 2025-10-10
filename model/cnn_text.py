"""Simple CNN text classifier."""
from __future__ import annotations

import torch
from torch import nn


class CNNText(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int = 2,
        embedding_dim: int = 128,
        num_filters: int = 100,
        kernel_sizes: tuple[int, ...] = (3, 4, 5),
        dropout: float = 0.3,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=k)
                for k in kernel_sizes
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)  # (B, L, D)
        embedded = embedded.transpose(1, 2)  # (B, D, L)
        conv_outs = [torch.relu(conv(embedded)) for conv in self.convs]
        pooled = [torch.max(out, dim=2)[0] for out in conv_outs]
        concatenated = torch.cat(pooled, dim=1)
        concatenated = self.dropout(concatenated)
        logits = self.classifier(concatenated)
        return logits
