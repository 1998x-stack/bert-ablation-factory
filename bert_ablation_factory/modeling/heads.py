from __future__ import annotations
import torch
from torch import nn
from .bilstm import BiLSTMEncoder


class ClassificationHead(nn.Module):
    """
    Classification head that takes the [CLS] token representation and applies 
    a linear layer to produce logits for classification.
    
    Args:
        hidden_size: Size of the hidden representations
        num_labels: Number of classification labels
        use_bilstm: Whether to apply BiLSTM encoder before classification
    """

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
        """
        Forward pass of the classification head.
        
        Args:
            hidden_states: Hidden representations from the transformer (B, T, H)
            attention_mask: Attention mask to indicate valid tokens (B, T)
            
        Returns:
            Logits for classification (B, num_labels)
        """
        # hidden_states: (B, T, H)
        x = hidden_states
        if self.use_bilstm:
            x = self.bridge(x, attention_mask)
        cls = x[:, 0, :]  # Take the [CLS] token representation
        cls = self.dropout(cls)
        return self.classifier(cls)


class SpanHead(nn.Module):
    """
    Head for SQuAD-style question answering that predicts start and end positions.
    Optionally applies BiLSTM encoder before prediction.
    
    Args:
        hidden_size: Size of the hidden representations
        use_bilstm: Whether to apply BiLSTM encoder before span prediction
    """

    def __init__(self, hidden_size: int, use_bilstm: bool = False) -> None:
        super().__init__()
        self.use_bilstm = use_bilstm
        if use_bilstm:
            self.bridge = BiLSTMEncoder(hidden_size)
        else:
            self.bridge = None
        self.qa_outputs = nn.Linear(hidden_size, 2)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the span prediction head.
        
        Args:
            hidden_states: Hidden representations from the transformer (B, T, H)
            attention_mask: Attention mask to indicate valid tokens (B, T)
            
        Returns:
            Tuple of (start_logits, end_logits) representing probability distributions
            over start and end positions (B, T) each
        """
        x = hidden_states
        if self.use_bilstm:
            x = self.bridge(x, attention_mask)
        logits = self.qa_outputs(x)  # (B, T, 2)
        start_logits, end_logits = logits.split(1, dim=-1)
        return start_logits.squeeze(-1), end_logits.squeeze(-1)
