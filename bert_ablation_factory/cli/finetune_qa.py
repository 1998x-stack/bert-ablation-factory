from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any, Dict
from loguru import logger
import torch
from torch.utils.data import DataLoader
from transformers import BertModel
from ..utils.io import load_yaml, merge_dict
from ..utils.logging import setup_logger
from ..utils.seed import fix_seed
from ..data.tokenization import build_tokenizer
from ..data.squad import load_squad_v1
from ..modeling.build import build_qa_model
from ..trainer.optimizer import build_optimizer
from ..trainer.schedulers import build_warmup_linear
from ..trainer.eval import squad_em_f1
from ..utils.tb import create_tb_writer


def parse_args():
    p = argparse.ArgumentParser("BERT Finetune - SQuAD v1.1")
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
    out_dir = Path(cfg.get("OUTPUT_DIR", "runs")) / "squad_v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    tb = create_tb_writer(out_dir, "tb")

    tokenizer = build_tokenizer(cfg)
    train_ds, dev_ds = load_squad_v1(
        tokenizer,
        int(cfg["DATA"]["max_seq_len"]),
        int(cfg["DATA"]["doc_stride"]),
        int(cfg["DATA"]["max_query_len"]),
    )

    base, head = build_qa_model(cfg, use_bilstm=bool(cfg["ABLATION"].get("use_bilstm_head", False)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base.to(device); head.to(device)

    train_loader = DataLoader(train_ds, batch_size=int(cfg["TRAIN"]["per_device_batch_size"]), shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=16)

    params = list(base.parameters()) + list(head.parameters())
    optim = build_optimizer(params, cfg)
    steps_per_epoch = len(train_loader)
    num_epochs = int(cfg["TRAIN"]["num_epochs"])
    max_steps = steps_per_epoch * num_epochs
    sched = build_warmup_linear(optim, cfg, max_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.get("FP16", True)))

    step = 0
    base.train(); head.train()
    for epoch in range(num_epochs):
        for batch in train_loader:
            step += 1
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=bool(cfg.get("FP16", True))):
                out = base(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                start_logits, end_logits = head(out.last_hidden_state, batch["attention_mask"])
                loss_f = torch.nn.CrossEntropyLoss()
                loss = (loss_f(start_logits, batch["start_positions"])
                        + loss_f(end_logits, batch["end_positions"])) / 2.0
            scaler.scale(loss).backward()
            scaler.step(optim); scaler.update(); optim.zero_grad(set_to_none=True); sched.step()

            if step % int(cfg.get("LOG_EVERY", 50)) == 0:
                tb.add_scalar("train/loss", float(loss), step)
                logger.info(f"epoch={epoch} step={step} loss={float(loss):.4f}")

        # 验证
        base.eval(); head.eval()
        preds_s, preds_e = [], []
        gts_s, gts_e = [], []
        with torch.no_grad():
            for vb in dev_loader:
                vb = {k: v.to(device) for k, v in vb.items()}
                hs = base(input_ids=vb["input_ids"], attention_mask=vb["attention_mask"]).last_hidden_state
                s_log, e_log = head(hs, vb["attention_mask"])
                preds_s.append(s_log.argmax(-1).cpu()); preds_e.append(e_log.argmax(-1).cpu())
                gts_s.append(vb["start_positions"].cpu()); gts_e.append(vb["end_positions"].cpu())
        metric = squad_em_f1(torch.cat(preds_s), torch.cat(preds_e), torch.cat(gts_s), torch.cat(gts_e))
        tb.add_scalar("valid/em", metric["em"], step)
        tb.add_scalar("valid/f1", metric["f1"], step)
        logger.info(f"[DEV] epoch={epoch} EM={metric['em']:.4f} F1={metric['f1']:.4f}")
        base.train(); head.train()

    torch.save({"base": base.state_dict(), "head": head.state_dict()}, out_dir / "bert_squad_v1.pt")
    logger.info(f"Saved weights to {out_dir / 'bert_squad_v1.pt'}")


if __name__ == "__main__":
    main()
