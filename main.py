import argparse
import os

from src.eval import evaluate
from src.train import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified entrypoint for training and evaluation")
    parser.add_argument("-t", "--train", action="store_true", help="Run training")
    parser.add_argument("-e", "--eval", action="store_true", help="Run evaluation")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help="Path to TOML config file (overrides config.toml)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.train and not args.eval:
        raise SystemExit("Select at least one action: --train and/or --eval")

    if args.config:
        os.environ["CARIES_CONFIG_PATH"] = args.config

    if args.train:
        train()

    if args.eval:
        evaluate()


if __name__ == "__main__":
    main()
