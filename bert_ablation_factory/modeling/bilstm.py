from __future__ import annotations
import torch
from torch import nn


class BiLSTMEncoder(nn.Module):
    """轻量 BiLSTM：可用于在 BERT 顶部再叠一层（消融用）。"""

    def __init__(self, hidden_size: int, num_layers: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        lengths = mask.long().sum(-1) if mask is not None else None
        out, _ = self.lstm(x)
        return out
