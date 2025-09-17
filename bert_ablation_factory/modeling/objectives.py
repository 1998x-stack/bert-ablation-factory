from __future__ import annotations
from typing import Dict, Any, Tuple
import torch
from torch import nn


class LossCombine(nn.Module):
    """将不同预训练损失组合（例如 MLM + NSP）。"""

    def __init__(self, use_mlm: bool = True, use_nsp: bool = False, use_ltr: bool = False) -> None:
        super().__init__()
        self.use_mlm = use_mlm
        self.use_nsp = use_nsp
        self.use_ltr = use_ltr
        self.ce = nn.CrossEntropyLoss()

    def forward(self, outputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        losses = {}
        total = torch.tensor(0.0, device=outputs["hidden_states"].device)
        if self.use_mlm:
            # outputs["mlm_logits"]: (B, T, V); labels: (B, T)
            mlm_loss = self.ce(outputs["mlm_logits"].permute(0, 2, 1), outputs["mlm_labels"])
            losses["mlm"] = mlm_loss
            total = total + mlm_loss
        if self.use_nsp:
            nsp_loss = self.ce(outputs["nsp_logits"], outputs["next_sentence_label"])
            losses["nsp"] = nsp_loss
            total = total + nsp_loss
        if self.use_ltr:
            ltr_loss = self.ce(outputs["ltr_logits"].permute(0, 2, 1), outputs["ltr_labels"])
            losses["ltr"] = ltr_loss
            total = total + ltr_loss
        return total, {k: v.item() for k, v in losses.items()}
