from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def relative_transfer_from_vectors(
    gradient_vectors: Sequence[Any], epsilon: float = 1e-12
) -> np.ndarray:
    """Compute Re(i,j) with rows as affected levels and columns as trained levels."""
    import torch

    if not gradient_vectors:
        raise ValueError("At least one gradient vector is required")
    vectors = [vector.detach().float().cpu() for vector in gradient_vectors]
    matrix = torch.stack(vectors)
    gram = matrix @ matrix.T
    denominators = torch.diagonal(gram).clamp_min(epsilon).unsqueeze(1)
    result = gram / denominators
    if not torch.isfinite(result).all():
        raise FloatingPointError("Relative Transfer matrix contains NaN or infinity")
    return result.numpy()


def trainable_gradient_vector(model: Any) -> Any:
    import torch

    gradients = []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            gradients.append(torch.zeros(parameter.numel(), dtype=torch.float32, device="cpu"))
        else:
            gradients.append(parameter.grad.detach().float().reshape(-1).cpu())
    if not gradients:
        raise ValueError("Model has no trainable gradients; verify LoRA setup")
    return torch.cat(gradients)


def estimate_transfer_matrix(
    model: Any,
    dataset: Any,
    collator: Any,
    probe_indices: dict[int, list[int]],
    device: Any,
) -> tuple[np.ndarray, dict[str, int]]:
    """Estimate five LoRA gradient vectors without changing parameters."""
    import torch

    was_training = model.training
    model.train()
    vectors = []
    probe_tokens = 0
    try:
        for level in range(1, 6):
            examples = [dataset[index] for index in probe_indices[level]]
            batch = {key: value.to(device) for key, value in collator(examples).items()}
            model.zero_grad(set_to_none=True)
            outputs = model(**batch, use_cache=False)
            outputs.loss.backward()
            vectors.append(trainable_gradient_vector(model))
            probe_tokens += int((batch["labels"] != -100).sum().item())
    finally:
        model.zero_grad(set_to_none=True)
        model.train(was_training)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    matrix = relative_transfer_from_vectors(vectors)
    if matrix.shape != (5, 5):
        raise AssertionError(f"Expected a 5x5 transfer matrix, got {matrix.shape}")
    return matrix, {
        "probe_examples": sum(len(indices) for indices in probe_indices.values()),
        "probe_tokens": probe_tokens,
        "probe_backward_passes": 5,
    }

