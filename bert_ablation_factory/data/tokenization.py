from __future__ import annotations
import os
from typing import Any, Dict

from transformers import BertTokenizer


def build_tokenizer(cfg: Dict[str, Any]) -> BertTokenizer:
    """Build a BERT tokenizer for the model named in ``cfg["MODEL"]["name"]``.

    Args:
        cfg: Configuration dict containing ``MODEL.name`` (a HuggingFace id or
            a local dir holding a saved tokenizer/vocab).

    Returns:
        A configured ``BertTokenizer``.

    Raises:
        ValueError: If ``MODEL.name`` is missing, empty, or not a string/os.PathLike.
        OSError: If the tokenizer cannot be built from the given name/path.
    """
    model_cfg = cfg.get("MODEL")
    if not isinstance(model_cfg, dict):
        raise ValueError("Config must contain a MODEL section")
    name = model_cfg.get("name")
    if isinstance(name, os.PathLike):
        name = os.fspath(name)
    if not isinstance(name, str) or not name:
        raise ValueError("MODEL.name must be a non-empty string or path")

    try:
        return BertTokenizer.from_pretrained(name)
    except OSError as e:
        raise OSError(f"Failed to load tokenizer from '{name}': {e}") from e
