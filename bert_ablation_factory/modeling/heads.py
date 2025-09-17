from __future__ import annotations
import torch
from torch import nn
from .bilstm import BiLSTMEncoder


class ClassificationHead(nn.Module):
    """[CLS] 分类头，可选 BiLSTM 过渡。"""

    def __init__(self, hidden_size: int, num_labels: int, use_bilstm: bool = False) -> None:
        super().__init__()
        self.use_bilstm = use_bilstm
        if use_bilstm:
            self.bridge = BiLSTMEncoder(hidden_size)
        else:
            self.bridge = None
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # hidden_states: (B, T, H)
        x = hidden_states
        if self.use_bilstm:
            x = self.bridge(x, attention_mask)
        cls = x[:, 0, :]
        cls = self.dropout(cls)
        return self.classifier(cls)


class SpanHead(nn.Module):
    """SQuAD 起止位置头，可选 BiLSTM 过渡。"""

    def __init__(self, hidden_size: int, use_bilstm: bool = False) -> None:
        super().__init__()
        self.use_bilstm = use_bilstm
        if use_bilstm:
            self.bridge = BiLSTMEncoder(hidden_size)
        else:
            self.bridge = None
        self.qa_outputs = nn.Linear(hidden_size, 2)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = hidden_states
        if self.use_bilstm:
            x = self.bridge(x, attention_mask)
        logits = self.qa_outputs(x)  # (B, T, 2)
        start_logits, end_logits = logits.split(1, dim=-1)
        return start_logits.squeeze(-1), end_logits.squeeze(-1)
