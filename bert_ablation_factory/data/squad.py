from __future__ import annotations
from tempfile import TemporaryDirectory
from typing import Any, Dict, Tuple
from datasets import Dataset, DatasetDict, load_dataset
from transformers import BertTokenizerFast, PreTrainedTokenizerBase


def _build_synthetic_squad(n: int, max_seq_len: int, doc_stride: int,
                           max_query_len: int):
    """Build an offline SQuAD-v1-shaped dataset with in-vocab `wNNN` tokens.

    Context is several hundred chars of space-separated `wNNN` tokens so it
    overflows into >1 feature window at the given config. The answer is an
    early in-vocab token (`w12`) whose char span sits inside the FIRST crafted
    window, so `start_positions`/`end_positions` align to a real (non-CLS) token.
    """
    context = " ".join(f"w{i}" for i in range(5, 250))  # ~700 chars, >1 window
    answer_text = "w12"
    answer_start = context.index(answer_text)
    rows = {
        "question": ["What token comes after w11?"] * n,
        "context": [context] * n,
        "answers": [{"answer_start": [answer_start], "text": [answer_text]}] * n,
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
        return _build_synthetic_squad(
            int((cfg.get("DATA") or {}).get("n", 4)),
            int(cfg["DATA"].get("max_seq_len", 64)),
            int(cfg["DATA"].get("doc_stride", 16)),
            int(cfg["DATA"].get("max_query_len", 16)),
        )
    raise ValueError(f"Unknown DATA.source: {source}")


def _ensure_fast(tokenizer):
    """Return a fast tokenizer supporting `offset_mapping`/`sequence_ids`.

    The repo builds a slow ``BertTokenizer`` via ``data/tokenization.py``, but
    the original SQuAD feature logic needs fast-only features. Rebuild the same
    vocab as a ``BertTokenizerFast`` so the body stays byte-for-byte original.
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
        questions = [q.strip() for q in examples["question"]]
        enc = enc_tokenizer(
            questions,
            examples["context"],
            truncation="only_second",
            max_length=max_seq_len,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
        sample_mapping = enc.pop("overflow_to_sample_mapping")
        offset_mapping = enc.pop("offset_mapping")
        start_positions, end_positions = [], []
        for i, offsets in enumerate(offset_mapping):
            input_ids = enc["input_ids"][i]
            cls_index = input_ids.index(enc_tokenizer.cls_token_id)
            sample_idx = sample_mapping[i]
            answer = examples["answers"][sample_idx]
            if len(answer["answer_start"]) == 0:
                start_positions.append(cls_index)
                end_positions.append(cls_index)
                continue
            start_char = answer["answer_start"][0]
            end_char = start_char + len(answer["text"][0])
            sequence_ids = enc.sequence_ids(i)
            # 找到 context 的 token 起止
            idx = 0
            while idx < len(sequence_ids) and sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while idx < len(sequence_ids) and sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1
            # 如果答案不在当前滑窗，标为 CLS
            if not (offsets[context_start][0] <= start_char and offsets[context_end][1] >= end_char):
                start_positions.append(cls_index)
                end_positions.append(cls_index)
            else:
                # 精确对齐到 token
                start_token = context_start
                while start_token <= context_end and offsets[start_token][0] <= start_char:
                    start_token += 1
                end_token = context_end
                while end_token >= context_start and offsets[end_token][1] >= end_char:
                    end_token -= 1
                start_positions.append(start_token - 1)
                end_positions.append(end_token + 1)
        enc["start_positions"] = start_positions
        enc["end_positions"] = end_positions
        return enc

    train_ds = ds["train"].map(prepare_train_features, batched=True, remove_columns=ds["train"].column_names)
    valid_ds = ds["validation"].map(prepare_train_features, batched=True, remove_columns=ds["validation"].column_names)
    train_ds.set_format(type="torch")
    valid_ds.set_format(type="torch")
    return train_ds, valid_ds