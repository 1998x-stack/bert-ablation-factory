from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import torch
import re


def atomic_save(state: Dict[str, Any], path: Path) -> None:
    """原子保存，避免中途中断导致坏文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def find_latest_checkpoint(dir_: Path, prefix: str = "ckpt_epoch_", suffix: str = ".pt") -> Optional[Path]:
    """在目录下寻找最近的断点文件，命名如 ckpt_epoch_0003.pt。"""
    if not dir_.exists():
        return None
    pat = re.compile(rf"^{re.escape(prefix)}(\d+){re.escape(suffix)}$")
    best_n, best_path = -1, None
    for p in dir_.iterdir():
        if p.is_file():
            m = pat.match(p.name)
            if m:
                n = int(m.group(1))
                if n > best_n:
                    best_n, best_path = n, p
    return best_path


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    epoch: int,
    step: int,
) -> None:
    """保存模型与优化器/调度器/AMP 的训练状态。"""
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": getattr(scheduler, "state_dict", lambda: {})(),
        "scaler": getattr(scaler, "state_dict", lambda: {})(),
        "epoch": epoch,
        "step": step,
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
    }
    atomic_save(state, path)


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
) -> Tuple[int, int]:
    """加载训练状态，返回 (epoch, step) 以便恢复循环指针。"""
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    try:
        scheduler.load_state_dict(state.get("scheduler", {}))
    except Exception:
        pass
    try:
        scaler.load_state_dict(state.get("scaler", {}))
    except Exception:
        pass
    rng = state.get("rng", {})
    torch.set_rng_state(rng.get("torch", torch.get_rng_state()))
    if torch.cuda.is_available() and rng.get("cuda"):
        torch.cuda.set_rng_state_all(rng["cuda"])
    return int(state.get("epoch", 0)), int(state.get("step", 0))
