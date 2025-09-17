from __future__ import annotations
from typing import Any, Dict
from pathlib import Path
import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """加载 YAML 为 dict。"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_dict(a: dict, b: dict) -> dict:
    """浅层合并：b 覆盖 a。"""
    out = dict(a)
    out.update(b)
    return out
