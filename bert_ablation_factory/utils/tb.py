from __future__ import annotations
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter


def create_tb_writer(out_dir: Path, name: str) -> SummaryWriter:
    """创建 TensorBoard 写入器。"""
    run_dir = out_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(run_dir))
