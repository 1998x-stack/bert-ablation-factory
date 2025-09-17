from __future__ import annotations
from typing import Dict, Any, Tuple
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase


def load_sst2(tokenizer: PreTrainedTokenizerBase, max_len: int) -> Tuple[Any, Any]:
    """加载 GLUE/SST-2 并分词。"""
    ds = load_dataset("glue", "sst2")
    def preprocess(batch):
        enc = tokenizer(batch["sentence"], truncation=True, max_length=max_len, padding="max_length")
        enc["labels"] = batch["label"]
        return enc
    ds = ds.map(preprocess, batched=True, remove_columns=ds["train"].column_names)
    ds.set_format(type="torch")
    return ds["train"], ds["validation"]
