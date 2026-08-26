from __future__ import annotations

import itertools
from typing import Any

from .hooks import decoder_layers


def resolve_sweep_grid(config: dict[str, Any], model: Any) -> list[tuple[int, int, float]]:
    recirculation = config["recirculation"]
    layer_count = len(decoder_layers(model))
    destinations = recirculation.get("destination_layers")
    sources = recirculation.get("source_layers")
    if destinations is None:
        destinations = sorted({max(0, layer_count // 6), max(0, layer_count // 4)})
    if sources is None:
        sources = sorted({layer_count // 2, (2 * layer_count) // 3, layer_count - 2})
    alphas = recirculation["alphas"]
    grid = [
        (int(source), int(destination), float(alpha))
        for source, destination, alpha in itertools.product(sources, destinations, alphas)
        if 0 <= int(destination) < int(source) < layer_count
    ]
    if not grid:
        raise ValueError(f"No valid Recirculation layer pairs for {layer_count} decoder layers")
    return grid

