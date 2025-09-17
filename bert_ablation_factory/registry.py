from __future__ import annotations
from typing import Any, Callable, Dict


class Registry:
    """简单可插拔注册表。"""

    def __init__(self, name: str) -> None:
        self._name = name
        self._obj: Dict[str, Any] = {}

    def register(self, key: str) -> Callable[[Any], Any]:
        def deco(fn: Any) -> Any:
            if key in self._obj:
                raise KeyError(f"{self._name} already has key: {key}")
            self._obj[key] = fn
            return fn
        return deco

    def get(self, key: str) -> Any:
        if key not in self._obj:
            raise KeyError(f"{self._name} missing key: {key}")
        return self._obj[key]

    def keys(self) -> list[str]:
        return list(self._obj.keys())


MASKERS = Registry("masker")
OBJECTIVES = Registry("objective")
HEADS = Registry("head")
TASKS = Registry("task")
