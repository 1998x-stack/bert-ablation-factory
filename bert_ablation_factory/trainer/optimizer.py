from __future__ import annotations
from typing import Dict, Any, Iterable
import torch
from torch.optim import AdamW


def build_optimizer(params: Iterable, cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    lr = float(cfg["OPTIM"]["lr"])
    wd = float(cfg["OPTIM"]["weight_decay"])
    betas = tuple(cfg["OPTIM"].get("betas", [0.9, 0.999]))
    eps = float(cfg["OPTIM"].get("eps", 1e-8))
    return AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=wd)
