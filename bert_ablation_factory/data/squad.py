from __future__ import annotations
from typing import Dict, Any, Tuple
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase


def load_squad_v1(tokenizer: PreTrainedTokenizerBase, max_seq_len: int, doc_stride: int, max_query_len: int):
    """加载 SQuAD v1.1 并进行滑窗切分与特征化（简实现）。"""
    ds = load_dataset("squad")

    def prepare_train_features(examples):
        questions = [q.strip() for q in examples["question"]]
        enc = tokenizer(
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
            cls_index = input_ids.index(tokenizer.cls_token_id)
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
