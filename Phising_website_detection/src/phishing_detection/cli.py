"""Command-line interface for the phishing detector.

Usage:
    uv run phishing-detector train
    uv run phishing-detector evaluate
    uv run phishing-detector predict data/phishing_data.csv --output predictions.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from .config import DATASET_PATH, MODEL_PATH
from .data import load_dataset
from .evaluate import (
    duplicate_aware_holdout,
    standard_cross_validate,
    strict_cross_validate,
)
from .predict import load_model, predict_dataframe
from .train import run_full_pipeline


def cmd_train(args: argparse.Namespace) -> int:
    run_full_pipeline(save=not args.no_save)
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    frame = load_dataset(args.dataset)

    strict = duplicate_aware_holdout(frame)
    print("Duplicate-aware holdout evaluation")
    print(f"  rows before cleaning    : {strict['rows_before']:,}")
    print(f"  rows after cleaning     : {strict['rows_after']:,}")
    print(f"  ambiguous groups dropped: {strict['conflicting_groups_removed']:,}")
    print(f"  accuracy                : {strict['accuracy'] * 100:.2f}%")

    scores = strict_cross_validate(frame)
    print("\nCross-validation on cleaned unique patterns (strict)")
    for index, score in enumerate(scores, start=1):
        print(f"  fold {index}: {score:.4f}")
    print(f"  mean    : {scores.mean() * 100:.2f}%")
    print(f"  std dev : {scores.std() * 100:.2f} percentage points")

    optimistic = standard_cross_validate(frame)
    print("\nCross-validation on the full dataset (optimistic)")
    for index, score in enumerate(optimistic, start=1):
        print(f"  fold {index}: {score:.4f}")
    print(f"  mean    : {optimistic.mean() * 100:.2f}%")
    print(f"  std dev : {optimistic.std() * 100:.2f} percentage points")

    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    model = load_model(args.model)
    frame = pd.read_csv(args.csv)

    result = predict_dataframe(model, frame)

    counts = result["prediction"].value_counts()
    print(f"Rows predicted : {len(result):,}")
    for label, count in counts.items():
        print(f"  {label:<12}: {count:,}")

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output, index=False)
        print(f"\nPredictions written to: {output}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="phishing-detector",
        description=(
            "Train and evaluate the phishing website detection model."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser(
        "train",
        help="Train the model and print the full evaluation report.",
    )
    train_parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write the trained model to models/.",
    )
    train_parser.set_defaults(func=cmd_train)

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Run the duplicate-aware evaluation and cross-validation.",
    )
    evaluate_parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_PATH,
        help=f"Path to the dataset CSV (default: {DATASET_PATH}).",
    )
    evaluate_parser.set_defaults(func=cmd_evaluate)

    predict_parser = subparsers.add_parser(
        "predict",
        help="Predict a CSV of features and optionally write the results.",
    )
    predict_parser.add_argument(
        "csv",
        nargs="?",
        type=Path,
        default=DATASET_PATH,
        help=f"CSV to predict (default: {DATASET_PATH}).",
    )
    predict_parser.add_argument(
        "--model",
        type=Path,
        default=MODEL_PATH,
        help=f"Path to the trained model (default: {MODEL_PATH}).",
    )
    predict_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write the predictions to this CSV path.",
    )
    predict_parser.set_defaults(func=cmd_predict)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
