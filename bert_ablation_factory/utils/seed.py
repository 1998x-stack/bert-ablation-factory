from __future__ import annotations
import os
import random
import numpy as np
import torch


def fix_seed(seed: int) -> None:
    """固定随机种子，确保可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
