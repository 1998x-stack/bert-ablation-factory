from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple
import random
import torch


@dataclass
class MLMConfig:
    mask_prob: float = 0.15
    mask_strategy: str = "80_10_10"  # or "100_mask"
    pad_token_id: int = 0
    mask_token_id: int = 103


def _apply_mlm(tokens: torch.Tensor, attention_mask: torch.Tensor, cfg: MLMConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    """对 batch 输入应用 MLM 掩码策略。"""
    labels = tokens.clone()
    # 不在 padding 的位置才能被选中
    probability_matrix = (attention_mask == 1).float() * cfg.mask_prob
    masked_indices = torch.bernoulli(probability_matrix).bool()

    labels[~masked_indices] = -100  # 仅计算被掩码 token 的 loss

    if cfg.mask_strategy == "80_10_10":
        # 80% -> [MASK]
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        tokens[indices_replaced] = cfg.mask_token_id
        # 10% -> random token
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint_like(tokens, low=0, high=tokens.max().item() + 1)
        tokens[indices_random] = random_words[indices_random]
        # 剩余 10% 保持不变
    elif cfg.mask_strategy == "100_mask":
        tokens[masked_indices] = cfg.mask_token_id
    else:
        raise ValueError(f"Unknown mask_strategy: {cfg.mask_strategy}")
    return tokens, labels


class MLMNSPCollator:
    """MLM+NSP 句对组 batch collator。
    期望输入样本已构造成 (input_ids, token_type_ids, attention_mask, next_sentence_label)。
    """

    def __init__(self, mlm_cfg: MLMConfig) -> None:
        self.mlm_cfg = mlm_cfg

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = {k: torch.tensor([f[k] for f in features], dtype=torch.long)
                 for k in ["input_ids", "token_type_ids", "attention_mask", "next_sentence_label"]}
        input_ids, labels = _apply_mlm(batch["input_ids"], batch["attention_mask"], self.mlm_cfg)
        batch["input_ids"] = input_ids
        batch["labels"] = labels            # MLM labels
        batch["next_sentence_label"] = batch["next_sentence_label"]
        return batch


class MLMOnlyCollator:
    """仅 MLM 的 collator（单句或拼接形式均可）。"""

    def __init__(self, mlm_cfg: MLMConfig) -> None:
        self.mlm_cfg = mlm_cfg

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = {k: torch.tensor([f[k] for f in features], dtype=torch.long)
                 for k in ["input_ids", "token_type_ids", "attention_mask"]}
        input_ids, labels = _apply_mlm(batch["input_ids"], batch["attention_mask"], self.mlm_cfg)
        batch["input_ids"] = input_ids
        batch["labels"] = labels
        return batch


class LTRCollator:
    """LTR（单向语言模型）collator。
    简化：标签为下一个 token（右移），pad 为 -100。注意：需配合 BertLMHeadModel / is_decoder=True 使用。
    """

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = {k: torch.tensor([f[k] for f in features], dtype=torch.long)
                 for k in ["input_ids", "attention_mask"]}
        # 构造 causal LM 的 labels（右移）
        input_ids = batch["input_ids"]
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = -100  # 最后一个没有下一个词
        batch["labels"] = labels
        return batch
