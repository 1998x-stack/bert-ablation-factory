from __future__ import annotations
from typing import Dict, Any
import torch


def accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Calculate classification accuracy.
    
    Args:
        pred: Predicted logits or probabilities (B, C) where B is batch size and C is number of classes
        target: Ground truth labels (B,)
        
    Returns:
        Accuracy as a float between 0 and 1
    """
    if target.numel() == 0:
        return 1.0  # If there are no targets, accuracy is 100%
    
    pred_label = pred.argmax(dim=-1)
    correct = (pred_label == target).sum().item()
    return correct / target.numel()


def squad_em_f1(pred_start: torch.Tensor, pred_end: torch.Tensor,
                gold_start: torch.Tensor, gold_end: torch.Tensor) -> Dict[str, float]:
    """
    Simplified EM/F1 calculation for SQuAD-style QA: exact match and token-level F1 (approximation).
    
    Args:
        pred_start: Predicted start positions (B,)
        pred_end: Predicted end positions (B,)
        gold_start: Ground truth start positions (B,)
        gold_end: Ground truth end positions (B,)
        
    Returns:
        Dictionary containing 'em' (exact match) and 'f1' scores
    """
    em = ((pred_start == gold_start) & (pred_end == gold_end)).float().mean().item()
    # Token-level overlap
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
