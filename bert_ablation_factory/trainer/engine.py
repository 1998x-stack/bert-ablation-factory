from __future__ import annotations
from typing import Dict, Any, Callable
from loguru import logger
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from ..utils.tb import create_tb_writer
from .optimizer import build_optimizer
from .schedulers import build_warmup_linear


def _to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Move a batch of tensors to the specified device.
    
    Args:
        batch: Dictionary mapping keys to torch tensors
        device: Target device to move tensors to
        
    Returns:
        Dictionary with tensors moved to the target device
    """
    return {k: v.to(device) for k, v in batch.items()}


def train_loop(
    model: torch.nn.Module,
    collate_loss_fn: Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor] | torch.Tensor],
    train_loader: DataLoader,
    valid_loader: DataLoader | None,
    cfg: Dict[str, Any],
    out_dir: Path,
    max_steps: int,
) -> None:
    """
    Generic training loop that supports Automatic Mixed Precision (AMP), 
    TensorBoard logging, and checkpoint saving.
    
    Args:
        model: The PyTorch model to train
        collate_loss_fn: Function that computes loss from a batch
        train_loader: DataLoader for training data
        valid_loader: Optional DataLoader for validation data
        cfg: Configuration dictionary
        out_dir: Directory to save checkpoints and logs
        max_steps: Maximum number of training steps
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optim = build_optimizer(model.parameters(), cfg)
    sched = build_warmup_linear(optim, cfg, num_training_steps=max_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.get("FP16", True)))
    writer = create_tb_writer(out_dir, "tb")

    step = 0
    model.train()
    for epoch in range(10_000_000):
        for batch in train_loader:
            step += 1
            batch = _to_device(batch, device)
            with torch.cuda.amp.autocast(enabled=bool(cfg.get("FP16", True))):
                outputs = collate_loss_fn(batch)
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs

            scaler.scale(loss).backward()
            if step % int(cfg.get("GRAD_ACCUM_STEPS", 1)) == 0:
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
                sched.step()

            if step % int(cfg.get("LOG_EVERY", 50)) == 0:
                logger.info(f"step={step} loss={float(loss):.4f}")
                writer.add_scalar("train/loss", float(loss), step)

            if valid_loader is not None and step % int(cfg.get("EVAL_EVERY", 1000)) == 0:
                evaluate(model, valid_loader, collate_loss_fn, device, writer, step)

            if step % int(cfg.get("SAVE_EVERY", 1000)) == 0:
                save_path = out_dir / f"checkpoint_step_{step}.pt"
                torch.save({"model": model.state_dict(), "step": step}, save_path)
                logger.info(f"Saved checkpoint to {save_path}")

            if step >= max_steps:
                logger.info("Training finished.")
                return


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader,
             collate_loss_fn: Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor] | torch.Tensor],
             device: torch.device, writer, step: int) -> None:
    """
    Evaluate the model on the validation dataset.
    
    Args:
        model: The PyTorch model to evaluate
        loader: DataLoader for validation data
        collate_loss_fn: Function that computes loss from a batch
        device: Device to run evaluation on
        writer: TensorBoard writer for logging
        step: Current training step for logging
    """
    model.eval()
    losses = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = collate_loss_fn(batch)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs
        losses.append(float(loss))
    avg = sum(losses) / max(1, len(losses))
    writer.add_scalar("valid/loss", avg, step)
    model.train()
