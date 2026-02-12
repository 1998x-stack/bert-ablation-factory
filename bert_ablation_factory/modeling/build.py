from __future__ import annotations
from typing import Dict, Any
from transformers import (BertModel, BertConfig, BertForPreTraining, BertForMaskedLM,
                          BertForSequenceClassification, BertForQuestionAnswering)
import torch
from .heads import ClassificationHead, SpanHead


def build_pretrain_model(cfg: Dict[str, Any], ablation: str):
    """
    Build a pre-training model based on the ablation objective:
    - mlm_nsp: BertForPreTraining
    - mlm_only: BertForMaskedLM
    - ltr: BertLMHeadModel (raises error if unavailable)
    
    Args:
        cfg: Configuration dictionary
        ablation: Type of ablation objective ('mlm_nsp', 'mlm_only', or 'ltr')
        
    Returns:
        Tuple of (model, kind) where kind indicates the ablation type
    """
    name = cfg["MODEL"]["name"]
    if ablation == "mlm_nsp":
        model = BertForPreTraining.from_pretrained(name)
        kind = "mlm_nsp"
    elif ablation == "mlm_only":
        model = BertForMaskedLM.from_pretrained(name)
        kind = "mlm_only"
    elif ablation == "ltr":
        try:
            from transformers import BertLMHeadModel
        except Exception as e:
            raise RuntimeError("Current environment doesn't support BertLMHeadModel, please upgrade transformers or use mlm_only/mlm_nsp") from e
        model = BertLMHeadModel.from_pretrained(name)
        # Ensure causal: some versions require is_decoder=True
        if hasattr(model.config, "is_decoder"):
            model.config.is_decoder = True
        kind = "ltr"
    else:
        raise ValueError(f"Unknown ablation objective: {ablation}")
    return model, kind


def build_classification_model(num_labels: int, cfg: Dict[str, Any], use_bilstm: bool = False):
    """
    Build classification model: reuse BERT backbone + optional BiLSTM head
    (can also use BertForSequenceClassification directly).
    
    Args:
        num_labels: Number of classification labels
        cfg: Configuration dictionary
        use_bilstm: Whether to use BiLSTM head instead of standard classification head
        
    Returns:
        Tuple of (base model, classification head)
    """
    name = cfg["MODEL"]["name"]
    base = BertModel.from_pretrained(name)
    head = ClassificationHead(base.config.hidden_size, num_labels, use_bilstm=use_bilstm)
    return base, head


def build_qa_model(cfg: Dict[str, Any], use_bilstm: bool = False):
    """
    Build SQuAD model: BERT backbone + SpanHead 
    (can also use BertForQuestionAnswering directly).
    
    Args:
        cfg: Configuration dictionary
        use_bilstm: Whether to use BiLSTM head instead of standard span head
        
    Returns:
        Tuple of (base model, span head)
    """
    name = cfg["MODEL"]["name"]
    base = BertModel.from_pretrained(name)
    head = SpanHead(base.config.hidden_size, use_bilstm=use_bilstm)
    return base, head
