from __future__ import annotations
import torch
from torch import nn


class BiLSTMEncoder(nn.Module):
    """
    Lightweight BiLSTM encoder that can be used to stack on top of BERT for ablation studies.

    Args:
        hidden_size: Size of the input/output hidden representations
        num_layers: Number of LSTM layers
        dropout: Dropout rate between LSTM layers (only applied if num_layers > 1)
    """

    def __init__(
        self, hidden_size: int, num_layers: int = 1, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Forward pass of the BiLSTM encoder.

        Args:
            x: Input tensor of shape (B, T, H) where B is batch size,
               T is sequence length, and H is hidden size
            mask: Optional attention mask indicating valid positions (B, T)

        Returns:
            Output tensor of shape (B, T, H) with the same dimensions as input
        """
        out, _ = self.lstm(x)
        return out
