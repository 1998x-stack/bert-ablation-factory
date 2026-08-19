from __future__ import annotations
from tempfile import TemporaryDirectory
from typing import Any, Dict, Tuple
from datasets import Dataset, DatasetDict, load_dataset
from transformers import BertTokenizerFast, PreTrainedTokenizerBase


def _build_synthetic_squad(n: int):
    rows = {
        "question": ["What is the capital of France?"] * n,
        "context": ["The capital of France is Paris. It is a large city."] * n,
        "answers": [{"answer_start": [28], "text": ["Paris"]}] * n,
    }
    return DatasetDict({"train": Dataset.from_dict(rows),
                        "validation": Dataset.from_dict(rows)})


def _load_squad(cfg):
    source = (cfg.get("DATA") or {}).get("source", "hf")
    if source == "hf":
        return load_dataset("squad")
    if source == "json":
        files = {"train": cfg["DATA"]["train_path"],
                 "validation": cfg["DATA"]["dev_path"]}
        return load_dataset("json", data_files=files)
    if source == "synthetic":
        return _build_synthetic_squad(int((cfg.get("DATA") or {}).get("n", 4)))
    raise ValueError(f"Unknown DATA.source: {source}")


def _ensure_fast(tokenizer):
    """返回一个支持 offset_mapping 的快速分词器。

    ``return_offsets_mapping``/``sequence_ids`` 仅对 fast tokenizer 可用，而本
    代码库用 ``data/tokenization.py`` 构建的是慢版 ``BertTokenizer``。这里用同一份
    vocab 重建对应的 fast tokenizer，使滑窗特征化的正文逻辑保持不变。
    """
    if getattr(tokenizer, "is_fast", False):
        return tokenizer
    with TemporaryDirectory() as tmp:
        tokenizer.save_pretrained(tmp)
        return BertTokenizerFast.from_pretrained(tmp, use_fast=True)


def load_squad_v1(tokenizer: PreTrainedTokenizerBase, cfg):
    """加载 SQuAD v1.1 并进行滑窗切分与特征化（简实现）。"""
    max_seq_len = int(cfg["DATA"]["max_seq_len"])
    doc_stride = int(cfg["DATA"]["doc_stride"])
    max_query_len = int(cfg["DATA"]["max_query_len"])
    ds = _load_squad(cfg)
    enc_tokenizer = _ensure_fast(tokenizer)

    def prepare_train_features(examples):
        result = {"input_ids": [], "attention_mask": [],
                  "start_positions": [], "end_positions": []}
        cls_id = enc_tokenizer.cls_token_id
        sep_id = enc_tokenizer.sep_token_id
        pad_id = enc_tokenizer.pad_token_id

        for q, ctx, answer in zip(examples["question"],
                                  examples["context"],
                                  examples["answers"]):
            q = q.strip()
            # 用 fast tokenizer 拿到字符级 offset（不触发后端 stride 溢出），
            # 然后手动滑窗，避免 stride >= 短窗口长度时 tokenizers 的 panic。
            enc = enc_tokenizer(q, ctx, truncation=False,
                                return_offsets_mapping=True)
            ids = enc["input_ids"]
            om = enc["offset_mapping"]
            # 兼容 fast tokenizer 返回 flat（单条）或 nested（batch）两种结构
            if len(om) > 0 and isinstance(om[0], (list, tuple)) \
                    and len(om[0]) == 2 and not isinstance(om[0][0], (list, tuple)):
                offsets = om
            else:
                offsets = om[0] if om else []
            seq_ids = enc.sequence_ids(0)

            # context 的 token 边界（seq id == 1）
            ctx_start = next((i for i, s in enumerate(seq_ids) if s == 1),
                             len(ids))
            ctx_end = max((i for i, s in enumerate(seq_ids) if s == 1),
                          default=ctx_start)
            q_part = ids[:ctx_start]           # [CLS] question [SEP]
            ctx_ids = ids[ctx_start:ctx_end + 1]
            ctx_offs = offsets[ctx_start:ctx_end + 1]

            has_answer = len(answer["answer_start"]) > 0
            if has_answer:
                start_char = int(answer["answer_start"][0])
                end_char = start_char + len(answer["text"][0])
            else:
                start_char = end_char = -1

            # 答案落在 context 的哪些 token（绝对 ctx 下标）
            a_start = a_end = None
            if has_answer:
                for i, (a, b) in enumerate(ctx_offs):
                    if a <= start_char < b and a_start is None:
                        a_start = i
                    if a < end_char <= b:
                        a_end = i
                if a_start is None or a_end is None:
                    has_answer = False

            # 手动滑窗生成 multiple features
            cut = len(q_part)                       # 预留 question 块
            capacity = max_seq_len - cut - 1        # context 可容纳数 + 尾部 [SEP]
            if capacity < 1:
                capacity = 1
            ctx_len = len(ctx_ids)
            if ctx_len <= capacity:
                windows = [(0, ctx_len)]
            else:
                windows = []
                start = 0
                while True:
                    end = min(start + capacity, ctx_len)
                    windows.append((start, end))
                    if end == ctx_len:
                        break
                    start += doc_stride
                    if start >= end:
                        start = end - 1 if end > 0 else 0
                    if start >= ctx_len:
                        windows.append((max(0, ctx_len - capacity), ctx_len))
                        break

            for w_start, w_end in windows:
                w_ids = q_part + ctx_ids[w_start:w_end] + [sep_id]
                if len(w_ids) > max_seq_len:
                    w_ids = w_ids[:max_seq_len]
                w_len = len(w_ids)
                w_ids = w_ids + [pad_id] * (max_seq_len - w_len)
                attn = [1] * w_len + [0] * (max_seq_len - w_len)
                if not has_answer or not (w_start <= a_start and a_end <= w_end):
                    sp = ep = 0                       # CLS
                else:
                    sp = cut + (a_start - w_start)
                    ep = cut + (a_end - w_start)
                result["input_ids"].append(w_ids)
                result["attention_mask"].append(attn)
                result["start_positions"].append(sp)
                result["end_positions"].append(ep)
        return result

    train_ds = ds["train"].map(prepare_train_features, batched=True, remove_columns=ds["train"].column_names)
    valid_ds = ds["validation"].map(prepare_train_features, batched=True, remove_columns=ds["validation"].column_names)
    train_ds.set_format(type="torch")
    valid_ds.set_format(type="torch")
    return train_ds, valid_ds