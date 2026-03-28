from __future__ import annotations
from typing import Any, Dict
from pathlib import Path
import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """加载 YAML 为 dict。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Config path is not a file: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            content = yaml.safe_load(f)
            if content is None:
                raise ValueError(f"Config file is empty or invalid: {path}")
            return content
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML syntax in {path}: {e}")


def merge_dict(a: dict, b: dict) -> dict:
    """浅层合并：b 覆盖 a。"""
    if not isinstance(a, dict) or not isinstance(b, dict):
        raise TypeError("Both inputs must be dictionaries")
    out = dict(a)
    out.update(b)
    return out
