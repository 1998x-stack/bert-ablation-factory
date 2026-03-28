from __future__ import annotations
from typing import Dict, Any
from transformers import (
    BertModel,
    BertConfig,
    BertForPreTraining,
    BertForMaskedLM,
    BertForSequenceClassification,
    BertForQuestionAnswering,
)
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

    Raises:
        ValueError: If ablation objective is unknown or config is invalid
        RuntimeError: If LTR model is not supported
    """
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be a dictionary")
    if not isinstance(ablation, str):
        raise TypeError("ablation must be a string")

    if "MODEL" not in cfg or "name" not in cfg["MODEL"]:
        raise ValueError("Config must contain MODEL.name")

    name = cfg["MODEL"]["name"]
    if not isinstance(name, str) or not name:
        raise ValueError("MODEL.name must be a non-empty string")

    valid_ablations = ["mlm_nsp", "mlm_only", "ltr"]
    if ablation not in valid_ablations:
        raise ValueError(
            f"Unknown ablation objective: {ablation}. Must be one of {valid_ablations}"
        )

    try:
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
                raise RuntimeError(
                    "Current environment doesn't support BertLMHeadModel, please upgrade transformers or use mlm_only/mlm_nsp"
                ) from e
            model = BertLMHeadModel.from_pretrained(name)
            # Ensure causal: some versions require is_decoder=True
            if hasattr(model.config, "is_decoder"):
                model.config.is_decoder = True
            kind = "ltr"
        else:
            raise ValueError(f"Unknown ablation objective: {ablation}")
        return model, kind
    except Exception as e:
        raise RuntimeError(
            f"Failed to load model '{name}' for ablation '{ablation}': {e}"
        ) from e


def build_classification_model(
    num_labels: int, cfg: Dict[str, Any], use_bilstm: bool = False
):
    """
    Build classification model: reuse BERT backbone + optional BiLSTM head
    (can also use BertForSequenceClassification directly).

    Args:
        num_labels: Number of classification labels
        cfg: Configuration dictionary
        use_bilstm: Whether to use BiLSTM head instead of standard classification head

    Returns:
        Tuple of (base model, classification head)

    Raises:
        ValueError: If num_labels is invalid or config is missing required fields
        RuntimeError: If model loading fails
    """
    if not isinstance(num_labels, int) or num_labels <= 0:
        raise ValueError(f"num_labels must be a positive integer, got {num_labels}")
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be a dictionary")
    if "MODEL" not in cfg or "name" not in cfg["MODEL"]:
        raise ValueError("Config must contain MODEL.name")

    name = cfg["MODEL"]["name"]
    if not isinstance(name, str) or not name:
        raise ValueError("MODEL.name must be a non-empty string")

    try:
        base = BertModel.from_pretrained(name)
        head = ClassificationHead(
            base.config.hidden_size, num_labels, use_bilstm=use_bilstm
        )
        return base, head
    except Exception as e:
        raise RuntimeError(f"Failed to build classification model: {e}") from e


def build_qa_model(cfg: Dict[str, Any], use_bilstm: bool = False):
    """
    Build SQuAD model: BERT backbone + SpanHead
    (can also use BertForQuestionAnswering directly).

    Args:
        cfg: Configuration dictionary
        use_bilstm: Whether to use BiLSTM head instead of standard span head

    Returns:
        Tuple of (base model, span head)

    Raises:
        ValueError: If config is missing required fields
        RuntimeError: If model loading fails
    """
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be a dictionary")
    if "MODEL" not in cfg or "name" not in cfg["MODEL"]:
        raise ValueError("Config must contain MODEL.name")

    name = cfg["MODEL"]["name"]
    if not isinstance(name, str) or not name:
        raise ValueError("MODEL.name must be a non-empty string")

    try:
        base = BertModel.from_pretrained(name)
        head = SpanHead(base.config.hidden_size, use_bilstm=use_bilstm)
        return base, head
    except Exception as e:
        raise RuntimeError(f"Failed to build QA model: {e}") from e
