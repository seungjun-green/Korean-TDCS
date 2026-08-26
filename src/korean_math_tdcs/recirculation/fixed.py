from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .hooks import decoder_layers, hidden_from_output


@dataclass
class FixedRecirculation:
    """Feeds the previous token's deep residual stream into a shallower layer."""

    model: Any
    source_layer: int
    destination_layer: int
    alpha: float
    beta: float | None = None
    epsilon: float = 1e-6
    _source_state: Any | None = field(default=None, init=False, repr=False)
    _handles: list[Any] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        layers = decoder_layers(self.model)
        if not 0 <= self.destination_layer < self.source_layer < len(layers):
            raise ValueError(
                "Recirculation requires 0 <= destination_layer < source_layer < number of layers"
            )
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if self.beta is None:
            self.beta = 1.0 - self.alpha

    def reset(self) -> None:
        self._source_state = None

    def _capture_source(self, _module: Any, _inputs: Any, output: Any) -> None:
        hidden = hidden_from_output(output)
        self._source_state = hidden[:, -1:, :].detach()

    def _mix_destination(self, _module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]):
        import torch

        if self._source_state is None or self.alpha == 0.0:
            return args, kwargs
        if args:
            destination = args[0]
        elif "hidden_states" in kwargs:
            destination = kwargs["hidden_states"]
        else:
            raise TypeError("Decoder destination layer did not receive hidden_states")
        source = self._source_state.to(device=destination.device, dtype=destination.dtype)
        if source.shape[0] != destination.shape[0]:
            raise ValueError("Recirculation state batch size changed without reset")
        source = source.expand(-1, destination.shape[1], -1)
        source_norm = torch.linalg.vector_norm(source, dim=-1, keepdim=True).clamp_min(self.epsilon)
        destination_norm = torch.linalg.vector_norm(destination, dim=-1, keepdim=True)
        normalized_source = source * (destination_norm / source_norm)
        mixed = self.alpha * normalized_source + float(self.beta) * destination
        if args:
            return (mixed, *args[1:]), kwargs
        new_kwargs = dict(kwargs)
        new_kwargs["hidden_states"] = mixed
        return args, new_kwargs

    def __enter__(self) -> FixedRecirculation:
        layers = decoder_layers(self.model)
        self.reset()
        self._handles = [
            layers[self.destination_layer].register_forward_pre_hook(
                self._mix_destination, with_kwargs=True
            ),
            layers[self.source_layer].register_forward_hook(self._capture_source),
        ]
        return self

    def __exit__(self, *_exc_info: Any) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self.reset()

