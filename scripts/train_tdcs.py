#!/usr/bin/env python
from korean_math_tdcs.training.sft import train
from korean_math_tdcs.utils.config import apply_overrides, config_argument_parser, load_config
from korean_math_tdcs.utils.logging import configure_logging


def main() -> None:
    parser = config_argument_parser("Train LoRA SFT with TDCS")
    args = parser.parse_args()
    configure_logging()
    config = apply_overrides(load_config(args.config), args.set)
    print(train(config, method="tdcs"))


if __name__ == "__main__":
    main()

