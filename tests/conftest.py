import sys
from pathlib import Path
from typing import Dict

import pytest
import yaml
from transformers import (
    BertForMaskedLM,
    BertForPreTraining,
    BertForSequenceClassification,
    BertLMHeadModel,
    BertModel,
    BertTokenizer,
)

from tests.gen import make_config, make_tokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

_MODEL_CLS = {
    "pretrain": BertForPreTraining,
    "masked": BertForMaskedLM,
    "lmhead": BertLMHeadModel,
    "cls": BertForSequenceClassification,
    "bert": BertModel,
}


def _make_tiny_dir(base: Path, key: str) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    make_tokenizer(base)  # vocab.txt + tokenizer config, so MODEL.name resolves both
    override = {"is_decoder": True} if key == "lmhead" else {}
    model = _MODEL_CLS[key](make_config(**override))
    if key == "lmhead":
        model.config.is_decoder = True
    model.save_pretrained(str(base))
    return base


@pytest.fixture(scope="session")
def tiny_model(tmp_path_factory) -> Dict[str, str]:
    base = tmp_path_factory.mktemp("tiny")
    return {key: str(_make_tiny_dir(base / key, key)) for key in _MODEL_CLS}


@pytest.fixture(scope="session")
def tiny_tokenizer(tmp_path_factory) -> BertTokenizer:
    return make_tokenizer(tmp_path_factory.mktemp("tok"))


def write_yaml(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return path
