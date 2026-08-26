#!/usr/bin/env python
from korean_math_tdcs.evaluation.evaluator import evaluate
from korean_math_tdcs.utils.config import apply_overrides, config_argument_parser, load_config
from korean_math_tdcs.utils.logging import configure_logging


def main() -> None:
    parser = config_argument_parser("Evaluate the selected fixed Recirculation configuration")
    args = parser.parse_args()
    configure_logging()
    config = apply_overrides(load_config(args.config), args.set)
    required = ("source_layer", "destination_layer", "alpha")
    missing = [key for key in required if config["recirculation"].get(key) is None]
    if missing:
        parser.error(
            f"Select values from the validation sweep first; missing: {', '.join(missing)}"
        )
    print(evaluate(config))


if __name__ == "__main__":
    main()
