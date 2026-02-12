from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple
import random
import torch


@dataclass
class MLMConfig:
    """
    Configuration for Masked Language Model preprocessing.
    
    Attributes:
        mask_prob: Probability of masking each token
        mask_strategy: Strategy for masking ("80_10_10" or "100_mask")
        pad_token_id: ID of padding token
        mask_token_id: ID of mask token
    """
    mask_prob: float = 0.15
    mask_strategy: str = "80_10_10"  # or "100_mask"
    pad_token_id: int = 0
    mask_token_id: int = 103


def _apply_mlm(tokens: torch.Tensor, attention_mask: torch.Tensor, cfg: MLMConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply MLM masking strategy to batch input.
    
    Args:
        tokens: Input token IDs tensor (B, T)
        attention_mask: Attention mask indicating valid positions (B, T)
        cfg: MLM configuration parameters
        
    Returns:
        Tuple of (masked_tokens, labels) where masked_tokens are the input tokens with some replaced
        by mask tokens, and labels contain the original token IDs for loss computation (-100 for non-masked)
    """
    labels = tokens.clone()
    # Only non-padding positions can be selected for masking
    probability_matrix = (attention_mask == 1).float() * cfg.mask_prob
    masked_indices = torch.bernoulli(probability_matrix).bool()

    labels[~masked_indices] = -100  # Only compute loss for masked tokens

    if cfg.mask_strategy == "80_10_10":
        # 80% -> [MASK]
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        tokens[indices_replaced] = cfg.mask_token_id
        # 10% -> random token
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint_like(tokens, low=0, high=tokens.max().item() + 1)
        tokens[indices_random] = random_words[indices_random]
        # Remaining 10% unchanged
    elif cfg.mask_strategy == "100_mask":
        tokens[masked_indices] = cfg.mask_token_id
    else:
        raise ValueError(f"Unknown mask_strategy: {cfg.mask_strategy}")
    return tokens, labels


class MLMNSPCollator:
    """
    Batch collator for MLM+NSP (Masked Language Model + Next Sentence Prediction).
    Expects input samples to be formatted as (input_ids, token_type_ids, attention_mask, next_sentence_label).
    
    Args:
        mlm_cfg: Configuration for MLM preprocessing
    """

    def __init__(self, mlm_cfg: MLMConfig) -> None:
        self.mlm_cfg = mlm_cfg

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of features for MLM+NSP training.
        
        Args:
            features: List of feature dictionaries containing input_ids, token_type_ids, 
                      attention_mask, and next_sentence_label
            
        Returns:
            Dictionary containing batched tensors ready for model input
        """
        batch = {k: torch.tensor([f[k] for f in features], dtype=torch.long)
                 for k in ["input_ids", "token_type_ids", "attention_mask", "next_sentence_label"]}
        input_ids, labels = _apply_mlm(batch["input_ids"], batch["attention_mask"], self.mlm_cfg)
        batch["input_ids"] = input_ids
        batch["labels"] = labels            # MLM labels
        batch["next_sentence_label"] = batch["next_sentence_label"]
        return batch


class MLMOnlyCollator:
    """
    Collator for MLM-only training (works with single sentences or concatenated format).
    
    Args:
        mlm_cfg: Configuration for MLM preprocessing
    """

    def __init__(self, mlm_cfg: MLMConfig) -> None:
        self.mlm_cfg = mlm_cfg

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of features for MLM-only training.
        
        Args:
            features: List of feature dictionaries containing input_ids, token_type_ids, 
                      and attention_mask
            
        Returns:
            Dictionary containing batched tensors ready for model input
        """
        batch = {k: torch.tensor([f[k] for f in features], dtype=torch.long)
                 for k in ["input_ids", "token_type_ids", "attention_mask"]}
        input_ids, labels = _apply_mlm(batch["input_ids"], batch["attention_mask"], self.mlm_cfg)
        batch["input_ids"] = input_ids
        batch["labels"] = labels
        return batch


class LTRCollator:
    """
    Collator for LTR (Left-to-Right language modeling).
    Simplification: labels are shifted right (next token), pad with -100. 
    Note: Should be used with BertLMHeadModel / is_decoder=True.
    """

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of features for left-to-right language modeling.
        
        Args:
            features: List of feature dictionaries containing input_ids and attention_mask
            
        Returns:
            Dictionary containing batched tensors ready for model input
        """
        batch = {k: torch.tensor([f[k] for f in features], dtype=torch.long)
                 for k in ["input_ids", "attention_mask"]}
        # Construct causal LM labels (shift right)
        input_ids = batch["input_ids"]
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = -100  # Last position has no next token
        batch["labels"] = labels
        return batch
