from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, Any
from loguru import logger
from datasets import load_dataset, interleave_datasets
from torch.utils.data import DataLoader
import torch
from ..utils.io import load_yaml, merge_dict
from ..utils.logging import setup_logger
from ..utils.seed import fix_seed
from ..data.tokenization import build_tokenizer
from ..data.collators import MLMConfig, MLMNSPCollator, MLMOnlyCollator, LTRCollator
from ..modeling.build import build_pretrain_model
from ..trainer.engine import train_loop


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("BERT Ablation Pretrain")
    p.add_argument("--cfg", type=str, required=True, help="YAML 配置")
    return p.parse_args()


def build_books_wiki_stream(tokenizer, max_len: int):
    """使用 bookcorpusopen + wikipedia 简单拼接，流式读取，构造 NSP 句对。"""
    ds1 = load_dataset("bookcorpusopen", split="train", streaming=True)
    ds2 = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)
    mixed = interleave_datasets([ds1, ds2], probabilities=[0.5, 0.5], seed=42)
    # 将每行 text 分句（粗略），拼成句对或单句
    sep = tokenizer.sep_token or "[SEP]"

    def gen_examples():
        prev = None
        for ex in mixed:
            text = (ex.get("text") or "").strip()
            if not text:
                continue
            sents = [s.strip() for s in text.split(".") if s.strip()]
            for s in sents:
                if prev is None:
                    prev = s
                    continue
                # 50% 正例（下一句），50% 负例（随机）
                import random
                if random.random() < 0.5:
                    a, b, label = prev, s, 0  # IsNext
                else:
                    b = sents[random.randrange(len(sents))]
                    a, label = prev, 1       # NotNext
                enc = tokenizer(
                    a, b,
                    truncation=True,
                    max_length=max_len,
                    padding="max_length",
                    return_token_type_ids=True,
                    return_attention_mask=True,
                )
                enc["next_sentence_label"] = label
                yield enc
                prev = s

    return gen_examples()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.cfg)
    base_path = cfg.get("_base_")
    if base_path:
        base = load_yaml((Path(args.cfg).parent / base_path).resolve())
        cfg = merge_dict(base, {k: v for k, v in cfg.items() if k != "_base_"})
    setup_logger()
    fix_seed(int(cfg.get("SEED", 42)))

    out_dir = Path(cfg.get("OUTPUT_DIR", "runs")) / "pretrain"
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = build_tokenizer(cfg)
    model, kind = build_pretrain_model(cfg, cfg["ABLATION"]["objective"])
    model.train()

    # 数据
    max_len = int(cfg["DATA"].get("max_seq_len") or cfg["DATASET"].get("max_seq_len", 128))
    stream = build_books_wiki_stream(tokenizer, max_len)
    # 为了 DataLoader，需要把流变成有限 iterable（示例：每步从流中拉样本）
    from itertools import islice

    def take(n):
        for item in islice(stream, n):
            yield item

    # collator
    if kind == "mlm_nsp":
        collator = MLMNSPCollator(MLMConfig(mask_strategy=cfg["ABLATION"]["mask_strategy"],
                                            pad_token_id=tokenizer.pad_token_id,
                                            mask_token_id=tokenizer.mask_token_id))
    elif kind == "mlm_only":
        collator = MLMOnlyCollator(MLMConfig(mask_strategy=cfg["ABLATION"]["mask_strategy"],
                                             pad_token_id=tokenizer.pad_token_id,
                                             mask_token_id=tokenizer.mask_token_id))
    else:
        collator = LTRCollator()

    # 简易 DataLoader（每轮从流里抓固定步数）
    per_device_bs = int(cfg["TRAIN"]["per_device_batch_size"])
    steps = int(cfg["TRAIN"]["max_steps"])
    eval_every = int(cfg["TRAIN"].get("eval_steps", 1000))
    save_every = int(cfg["TRAIN"].get("save_steps", 1000))

    class StreamDataset(torch.utils.data.IterableDataset):
        def __iter__(self):
            return take(steps * per_device_bs)

    train_loader = DataLoader(StreamDataset(), batch_size=per_device_bs, collate_fn=collator, num_workers=0)

    # 损失封装：直接用 HF 模型自带 loss（ForPreTraining/ForMaskedLM/LMHead 都返回 loss）
    def step_fn(batch):
        outputs = model(**batch)
        return {"loss": outputs.loss}

    logger.info(f"Start pretraining: ablation={kind}")
    train_loop(model, step_fn, train_loader, None, cfg, out_dir, max_steps=steps)


if __name__ == "__main__":
    main()
