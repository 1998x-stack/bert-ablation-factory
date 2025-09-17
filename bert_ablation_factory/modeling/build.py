from __future__ import annotations
from typing import Dict, Any
from transformers import (BertModel, BertConfig, BertForPreTraining, BertForMaskedLM,
                          BertForSequenceClassification, BertForQuestionAnswering)
import torch
from .heads import ClassificationHead, SpanHead


def build_pretrain_model(cfg: Dict[str, Any], ablation: str):
    """根据消融目标构建预训练模型：
    - mlm_nsp: BertForPreTraining
    - mlm_only: BertForMaskedLM
    - ltr: BertLMHeadModel（若不可用则报错提示）
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
            raise RuntimeError("当前环境不支持 BertLMHeadModel，请升级 transformers 或改用 mlm_only/mlm_nsp") from e
        model = BertLMHeadModel.from_pretrained(name)
        # 确保 causal：某些版本需 is_decoder=True
        if hasattr(model.config, "is_decoder"):
            model.config.is_decoder = True
        kind = "ltr"
    else:
        raise ValueError(f"Unknown ablation objective: {ablation}")
    return model, kind


def build_classification_model(num_labels: int, cfg: Dict[str, Any], use_bilstm: bool = False):
    """分类模型：复用 BERT 主体 + 可选 BiLSTM 头（也可直接用 BertForSequenceClassification）。"""
    name = cfg["MODEL"]["name"]
    base = BertModel.from_pretrained(name)
    head = ClassificationHead(base.config.hidden_size, num_labels, use_bilstm=use_bilstm)
    return base, head


def build_qa_model(cfg: Dict[str, Any], use_bilstm: bool = False):
    """SQuAD 模型：BERT 主体 + SpanHead（或直接用 BertForQuestionAnswering）。"""
    name = cfg["MODEL"]["name"]
    base = BertModel.from_pretrained(name)
    head = SpanHead(base.config.hidden_size, use_bilstm=use_bilstm)
    return base, head
