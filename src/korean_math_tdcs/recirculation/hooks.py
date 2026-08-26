from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def decoder_layers(model: Any) -> Sequence[Any]:
    """Locate a conventional Hugging Face decoder block list, including PEFT wrappers."""
    root = model.get_base_model() if hasattr(model, "get_base_model") else model
    candidates = [
        ("model", "layers"),
        ("model", "model", "layers"),
        ("transformer", "h"),
        ("layers",),
    ]
    for path in candidates:
        value = root
        try:
            for name in path:
                value = getattr(value, name)
        except AttributeError:
            continue
        if hasattr(value, "__len__") and len(value) > 0:
            return value
    raise TypeError(f"Could not locate decoder layers for {type(root).__name__}")


def hidden_from_output(output: Any) -> Any:
    if hasattr(output, "shape"):
        return output
    if isinstance(output, tuple) and output and hasattr(output[0], "shape"):
        return output[0]
    raise TypeError(f"Unsupported decoder layer output type: {type(output).__name__}")

