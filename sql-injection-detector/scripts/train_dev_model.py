"""Train a small model on the synthetic dev dataset.

This exists purely so the API and its tests can run in a fresh clone where the
real (git-ignored) SQLiV2 dataset and model artifacts are absent. It writes the
same V4 artifact shape that ``src/train_v4.py`` produces, so
``api/ml/registry.py`` loads it through the identical code path.

Usage::

    uv run python scripts/train_dev_model.py            # uses data/sqliv2_dev.csv
    uv run python scripts/train_dev_model.py --help
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from api.ml.preprocessing import normalize_text  # noqa: E402

DEFAULT_DATA = BASE_DIR / "data" / "sqliv2_dev.csv"
DEFAULT_MODEL = BASE_DIR / "models" / "sql_injection_model_v4.pkl"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()

    if not args.data.exists():
        raise SystemExit(
            f"{args.data} not found. Run `uv run python scripts/make_dev_dataset.py` first."
        )

    frame = pd.read_csv(args.data, encoding="utf-8-sig")
    frame["Sentence"] = frame["Sentence"].fillna("").astype(str).map(normalize_text)
    frame["Label"] = pd.to_numeric(frame["Label"], errors="coerce")
    frame = frame.dropna(subset=["Label"])
    frame = frame[frame["Label"].isin([0, 1])]
    frame = frame.drop_duplicates(subset=["Sentence"]).reset_index(drop=True)

    x_train, x_test, y_train, y_test = train_test_split(
        frame["Sentence"],
        frame["Label"].astype(int),
        test_size=0.2,
        random_state=42,
        stratify=frame["Label"],
    )

    features = FeatureUnion(
        [
            (
                "char",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(2, 5),
                    min_df=1,
                    max_features=100_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=50_000,
                    sublinear_tf=True,
                ),
            ),
        ]
    )

    matrix_train = features.fit_transform(x_train)
    matrix_test = features.transform(x_test)

    model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    model.fit(matrix_train, y_train)

    predictions = model.predict(matrix_test)
    print(f"Held-out accuracy: {accuracy_score(y_test, predictions):.4%}")
    print(classification_report(y_test, predictions, target_names=["Normal", "SQL Injection"]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "features": features,
            "model": model,
            "version": "V4-dev",
            "labels": {0: "Normal Input", 1: "SQL Injection"},
            "normalization": "URL decode up to twice; normalize whitespace",
        },
        args.output,
    )
    print(f"Saved dev model to {args.output}")


if __name__ == "__main__":
    main()
