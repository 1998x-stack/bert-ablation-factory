from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any, Dict
from loguru import logger
import torch
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification
from ..utils.io import load_yaml, merge_dict
from ..utils.logging import setup_logger
from ..utils.seed import fix_seed
from ..data.tokenization import build_tokenizer
from ..data.glue import load_sst2
from ..trainer.optimizer import build_optimizer
from ..trainer.schedulers import build_warmup_linear
from ..trainer.eval import accuracy
from ..utils.tb import create_tb_writer


def parse_args():
    p = argparse.ArgumentParser("BERT Finetune - SST-2")
    p.add_argument("--cfg", type=str, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.cfg)
    base_path = cfg.get("_base_")
    if base_path:
        base = load_yaml((Path(args.cfg).parent / base_path).resolve())
        cfg = merge_dict(base, {k: v for k, v in cfg.items() if k != "_base_"})

    setup_logger()
    fix_seed(int(cfg.get("SEED", 42)))
    out_dir = Path(cfg.get("OUTPUT_DIR", "runs")) / "sst2"
    out_dir.mkdir(parents=True, exist_ok=True)
    tb = create_tb_writer(out_dir, "tb")

    tokenizer = build_tokenizer(cfg)
    train_ds, dev_ds = load_sst2(tokenizer, int(cfg["DATA"]["max_seq_len"]))

    model = BertForSequenceClassification.from_pretrained(cfg["MODEL"]["name"], num_labels=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader = DataLoader(train_ds, batch_size=int(cfg["TRAIN"]["per_device_batch_size"]), shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=64)

    optim = build_optimizer(model.parameters(), cfg)
    steps_per_epoch = len(train_loader)
    num_epochs = int(cfg["TRAIN"]["num_epochs"])
    max_steps = steps_per_epoch * num_epochs
    sched = build_warmup_linear(optim, cfg, max_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.get("FP16", True)))
    step = 0
    model.train()
    for epoch in range(num_epochs):
        for batch in train_loader:
            step += 1
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=bool(cfg.get("FP16", True))):
                out = model(**batch)
                loss = out.loss
            scaler.scale(loss).backward()
            scaler.step(optim); scaler.update(); optim.zero_grad(set_to_none=True); sched.step()
            if step % int(cfg.get("LOG_EVERY", 50)) == 0:
                tb.add_scalar("train/loss", float(loss), step)
                logger.info(f"epoch={epoch} step={step} loss={float(loss):.4f}")

        # 验证
        model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for vb in dev_loader:
                vb = {k: v.to(device) for k, v in vb.items()}
                logits = model(**vb).logits
                preds.append(logits.cpu())
                gts.append(vb["labels"].cpu())
        acc = accuracy(torch.cat(preds, 0), torch.cat(gts, 0))
        tb.add_scalar("valid/acc", acc, step)
        logger.info(f"[DEV] epoch={epoch} acc={acc:.4f}")
        model.train()

    torch.save(model.state_dict(), out_dir / "bert_sst2.pt")
    logger.info(f"Saved weights to {out_dir / 'bert_sst2.pt'}")


if __name__ == "__main__":
    main()
