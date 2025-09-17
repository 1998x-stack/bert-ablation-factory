from __future__ import annotations
from typing import Dict, Any
from torch.optim import Optimizer
from transformers import get_linear_schedule_with_warmup


def build_warmup_linear(optimizer: Optimizer, cfg: Dict[str, Any], num_training_steps: int):
    warmup_steps = int(cfg["OPTIM"]["warmup_steps"])
    return get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)
