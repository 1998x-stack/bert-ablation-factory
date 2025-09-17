from __future__ import annotations
from typing import Dict, Any
import torch


def accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    """分类准确率。"""
    pred_label = pred.argmax(dim=-1)
    correct = (pred_label == target).sum().item()
    return correct / max(1, target.numel())


def squad_em_f1(pred_start: torch.Tensor, pred_end: torch.Tensor,
                gold_start: torch.Tensor, gold_end: torch.Tensor) -> Dict[str, float]:
    """简化版 EM/F1：严格匹配与 token 级 F1（近似）。"""
    em = ((pred_start == gold_start) & (pred_end == gold_end)).float().mean().item()
    # token 级重叠
    f1s = []
    for ps, pe, gs, ge in zip(pred_start.tolist(), pred_end.tolist(), gold_start.tolist(), gold_end.tolist()):
        pset = set(range(ps, pe + 1))
        gset = set(range(gs, ge + 1))
        inter = len(pset & gset)
        if len(pset) == 0 or len(gset) == 0:
            f1s.append(0.0)
        else:
            prec = inter / max(1, len(pset))
            rec = inter / max(1, len(gset))
            f1s.append(0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec))
    return {"em": em, "f1": sum(f1s) / max(1, len(f1s))}
