#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

from korean_math_tdcs.evaluation.evaluator import evaluate_loaded, load_inference_model
from korean_math_tdcs.recirculation.sweep import resolve_sweep_grid
from korean_math_tdcs.utils.config import apply_overrides, config_argument_parser, load_config
from korean_math_tdcs.utils.io import utc_timestamp, write_json
from korean_math_tdcs.utils.logging import configure_logging
from korean_math_tdcs.utils.seed import seed_everything


def main() -> None:
    parser = config_argument_parser("Tune fixed Recirculation on validation data")
    args = parser.parse_args()
    configure_logging()
    config = apply_overrides(load_config(args.config), args.set)
    model, tokenizer = load_inference_model(config)
    rows = []
    for source, destination, alpha in resolve_sweep_grid(config, model):
        seed_everything(int(config.get("seed", 42)))
        recirculation = {
            "enabled": True,
            "source_layer": source,
            "destination_layer": destination,
            "alpha": alpha,
            "beta": 1.0 - alpha,
        }
        result = evaluate_loaded(
            model,
            tokenizer,
            config,
            benchmark_names=["validation_math"],
            recirculation=recirculation,
        )
        score = result["benchmarks"]["validation_math"]["score"]
        rows.append({**recirculation, "score": score, **result})
        print(f"source={source} destination={destination} alpha={alpha:.3f} score={score:.4f}")
    rows.sort(key=lambda row: (-row["score"], -row["tokens_per_second"]))
    output = Path(config["output"]["sweep_path"])
    write_json({"timestamp": utc_timestamp(), "best": rows[0], "runs": rows}, output)
    print(
        "Best configuration: "
        f"source={rows[0]['source_layer']}, destination={rows[0]['destination_layer']}, "
        f"alpha={rows[0]['alpha']}"
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
