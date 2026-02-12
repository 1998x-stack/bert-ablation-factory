from __future__ import annotations
from typing import Any, Callable, Dict


class Registry:
    """
    Simple pluggable registry for registering and retrieving objects by key.
    Used for managing different implementations of maskers, objectives, heads, and tasks.
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._obj: Dict[str, Any] = {}

    def register(self, key: str) -> Callable[[Any], Any]:
        """
        Register a function or class with the given key.
        
        Args:
            key: The key to register the object under
            
        Returns:
            A decorator function that registers the decorated object
        """
        def deco(fn: Any) -> Any:
            if key in self._obj:
                raise KeyError(f"{self._name} already has key: {key}")
            self._obj[key] = fn
            return fn
        return deco

    def get(self, key: str) -> Any:
        """
        Retrieve an object by key.
        
        Args:
            key: The key of the object to retrieve
            
        Returns:
            The registered object
            
        Raises:
            KeyError: If the key is not found in the registry
        """
        if key not in self._obj:
            raise KeyError(f"{self._name} missing key: {key}")
        return self._obj[key]

    def keys(self) -> list[str]:
        """
        Get all registered keys.
        
        Returns:
            A list of all registered keys
        """
        return list(self._obj.keys())


# Registry for different token masking strategies (e.g., 80/10/10 masking)
MASKERS = Registry("masker")

# Registry for different training objectives (e.g., MLM, NSP, LTR)
OBJECTIVES = Registry("objective")

# Registry for different head architectures (e.g., classification heads, BiLSTM heads)
HEADS = Registry("head")

# Registry for different tasks (e.g., SST-2, SQuAD)
TASKS = Registry("task")
