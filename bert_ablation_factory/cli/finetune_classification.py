from __future__ import annotations
import argparse
import shutil
from pathlib import Path
from typing import Any, Dict, Tuple, List
import numpy as np
from loguru import logger
import torch
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification
from ..utils.io import load_yaml, merge_dict
from ..utils.logging import setup_logger
from ..utils.seed import fix_seed
from ..data.tokenization import build_tokenizer
from ..trainer.optimizer import build_optimizer
from ..trainer.schedulers import build_warmup_linear
from ..utils.tb import create_tb_writer
from ..registry import TASKS
from ..tasks.glue import build_glue_task, compute_glue_metrics, pick_main_score
from ..trainer.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    find_latest_checkpoint,
)


def parse_args():
    p = argparse.ArgumentParser("BERT Finetune - GLUE (Classification/Regression)")
    p.add_argument("--cfg", type=str, required=True, help="YAML configuration file")
    p.add_argument(
        "--restarts", type=int, default=None, help="Override YAML RESTARTS setting"
    )
    p.add_argument(
        "--resume", action="store_true", help="Override YAML RESUME=True setting"
    )
    return p.parse_args()


def _build_model(
    cfg: Dict[str, Any], num_labels: int, problem_type: str
) -> BertForSequenceClassification:
    model = BertForSequenceClassification.from_pretrained(
        cfg["MODEL"]["name"], num_labels=num_labels
    )
    if problem_type == "regression":
        model.config.problem_type = "regression"
    return model


def _evaluate_once(
    model: BertForSequenceClassification,
    device: torch.device,
    loader: DataLoader,
    task: str,
    metric_obj,
) -> Dict[str, float]:
    """
    Evaluate the model on the given dataset.

    Args:
        model: The model to evaluate
        device: The device to run evaluation on
        loader: DataLoader containing the evaluation dataset
        task: Name of the task
        metric_obj: Metric object for computing metrics

    Returns:
        Dictionary of computed metrics
    """
    model.eval()
    logits_all, labels_all = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            logits_all.append(out.logits.detach().cpu().numpy())
            labels_all.append(batch["labels"].detach().cpu().numpy())
    logits = np.concatenate(logits_all, axis=0)
    labels = np.concatenate(labels_all, axis=0)
    return compute_glue_metrics(task, metric_obj, logits, labels)


def run_single_restart(
    cfg: Dict[str, Any],
    task_bundle: Dict[str, Any],
    run_dir: Path,
    seed: int,
) -> Tuple[float, Dict[str, float]]:
    """
    Run a single restart of training and evaluate on dev set.

    Args:
        cfg: Configuration dictionary
        task_bundle: Bundle containing task information
        run_dir: Directory to save run results
        seed: Random seed for this run

    Returns:
        Tuple of (main metric score, metrics dictionary)
    """
    fix_seed(seed)
    tb = create_tb_writer(run_dir, "tb")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据
    train_ds = task_bundle["train_ds"]
    dev_ds = task_bundle["dev_ds"]
    task = task_bundle["task_name"]
    metric = task_bundle["metric"]
    main_key = task_bundle["main_metric"]

    model = _build_model(
        cfg, task_bundle["num_labels"], task_bundle["problem_type"]
    ).to(device)
    optim = build_optimizer(model.parameters(), cfg)

    steps_per_epoch = (len(train_ds) // int(cfg["TRAIN"]["per_device_batch_size"])) + 1
    num_epochs = int(cfg["TRAIN"]["num_epochs"])
    max_steps = steps_per_epoch * num_epochs
    sched = build_warmup_linear(optim, cfg, max_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.get("FP16", True)))

    # Loader
    train_loader = DataLoader(
        train_ds, batch_size=int(cfg["TRAIN"]["per_device_batch_size"]), shuffle=True
    )
    dev_loaders: Dict[str, DataLoader] | DataLoader
    if isinstance(dev_ds, dict):
        dev_loaders = {k: DataLoader(v, batch_size=64) for k, v in dev_ds.items()}
    else:
        dev_loaders = DataLoader(dev_ds, batch_size=64)

    # 自动续训
    start_epoch, global_step = 0, 0
    latest = find_latest_checkpoint(run_dir)
    want_resume = bool(cfg["TRAIN"].get("RESUME", False))
    if want_resume and latest is not None:
        logger.info(f"[RESUME] Found checkpoint: {latest}")
        start_epoch, global_step = load_checkpoint(
            latest, model, optim, sched, scaler, device
        )
        logger.info(f"[RESUME] epoch={start_epoch}, step={global_step}")

    # 训练循环
    model.train()
    best_score, best_metrics = -1e9, {}
    for epoch in range(start_epoch, num_epochs):
        for batch_idx, batch in enumerate(train_loader):
            global_step += 1
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=bool(cfg.get("FP16", True))):
                out = model(**batch)
                loss = out.loss
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none=True)
            sched.step()

            if global_step % int(cfg.get("LOG_EVERY", 50)) == 0:
                tb.add_scalar("train/loss", float(loss), global_step)

            # Periodic evaluation
            if global_step % int(cfg["TRAIN"].get("eval_steps", 0) or 0) == 0:
                if isinstance(dev_loaders, dict):
                    all_metrics = {}
                    for k, ld in dev_loaders.items():
                        res = _evaluate_once(model, device, ld, task, metric)
                        all_metrics.update({f"{k}/{kk}": vv for kk, vv in res.items()})
                    main = all_metrics.get(f"matched/{main_key}", None)
                    if (
                        main is None
                    ):  # If main metric is not on matched, fall back to any available main metric
                        # For example, if you want to use average as the main metric, you can modify accordingly
                        main = max(
                            [v for k, v in all_metrics.items() if k.endswith(main_key)]
                        )
                    score = float(main)
                    for k, v in all_metrics.items():
                        tb.add_scalar(f"dev/{k}", v, global_step)
                else:
                    res = _evaluate_once(model, device, dev_loaders, task, metric)
                    score = pick_main_score(task, res)
                    for k, v in res.items():
                        tb.add_scalar(f"dev/{k}", v, global_step)

                if score > best_score:
                    best_score, best_metrics = (
                        score,
                        res if not isinstance(dev_loaders, dict) else all_metrics,
                    )
                    # 保存当前最优
                    save_checkpoint(
                        run_dir / "best.pt",
                        model,
                        optim,
                        sched,
                        scaler,
                        epoch,
                        global_step,
                    )

        # 每个 epoch 末尾保存断点
        ckpt_path = run_dir / f"ckpt_epoch_{epoch:04d}.pt"
        save_checkpoint(ckpt_path, model, optim, sched, scaler, epoch, global_step)

    return float(best_score), {k: float(v) for k, v in best_metrics.items()}


def main() -> None:
    """
    Main function to run BERT fine-tuning on GLUE classification/regression tasks.
    Performs multiple random restarts to get reliable performance estimates.
    """
    args = parse_args()
    cfg = load_yaml(args.cfg)
    base_path = cfg.get("_base_")
    if base_path:
        base = load_yaml((Path(args.cfg).parent / base_path).resolve())
        cfg = merge_dict(base, {k: v for k, v in cfg.items() if k != "_base_"})

    # Override YAML settings if provided via command line
    if args.restarts is not None:
        cfg["TRAIN"]["RESTARTS"] = int(args.restarts)
    if args.resume:
        cfg["TRAIN"]["RESUME"] = True

    setup_logger()

    # Build task
    tokenizer = build_tokenizer(cfg)
    bundle = build_glue_task(cfg, tokenizer)
    task = bundle["task_name"]

    # Multiple random restarts (different seeds)
    base_seed = int(cfg.get("SEED", 42))
    restarts = int(cfg["TRAIN"].get("RESTARTS", 1))
    root_out = Path(cfg.get("OUTPUT_DIR", "runs")) / f"{bundle['task_name']}"
    root_out.mkdir(parents=True, exist_ok=True)

    best_overall, best_detail, best_run = -1e9, {}, -1
    for r in range(restarts):
        run_dir = root_out / f"run_{r:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        score, detail = run_single_restart(cfg, bundle, run_dir, seed=base_seed + r)
        logger.info(f"[RUN {r}] main={score:.4f} metrics={detail}")
        if score > best_overall:
            best_overall, best_detail, best_run = score, detail, r

    logger.info(f"[BEST] run={best_run} main={best_overall:.4f} metrics={best_detail}")
    # Copy the best run's best.pt to root directory (for deployment/evaluation)
    src = root_out / f"run_{best_run:02d}" / "best.pt"
    if src.exists():
        shutil.copy2(src, root_out / "best.pt")
        logger.info(f"Saved overall best checkpoint to {root_out / 'best.pt'}")


if __name__ == "__main__":
    main()
