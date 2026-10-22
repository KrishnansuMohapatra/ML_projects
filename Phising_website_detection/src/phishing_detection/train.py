"""Training pipeline: fit, evaluate, analyse, and persist the model."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (
    FEATURES,
    MODEL_PATH,
    N_ESTIMATORS,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)
from .data import load_dataset, split_features_target
from .evaluate import (
    build_model,
    duplicate_aware_holdout,
    error_analysis,
    evaluate_model,
    feature_importance,
    standard_cross_validate,
    strict_cross_validate,
)


def train_model(
    n_estimators: int = N_ESTIMATORS,
    random_state: int = RANDOM_STATE,
    test_size: float = TEST_SIZE,
) -> tuple:
    """Train the Random Forest and return `(model, results)`.

    The returned model is fitted on the training split only, so that the
    reported metrics describe how it generalises.
    """
    frame = load_dataset()
    X, y = split_features_target(frame)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    model = build_model(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)

    results = evaluate_model(model, X_test, y_test)
    results["model"] = model
    results["importance"] = feature_importance(model, X)
    results["errors"] = error_analysis(
        model,
        X_test,
        y_test,
        results["predictions"],
    )

    return model, results


def save_model(model, path: str | Path = MODEL_PATH) -> Path:
    """Persist a trained model to disk, creating the directory if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def run_full_pipeline(save: bool = True) -> dict:
    """Run training, the strict evaluation, and cross-validation.

    Prints a report and optionally saves the model to `models/`.
    """
    report: dict = {}

    print("=" * 62)
    print("PHISHING WEBSITE DETECTION - TRAINING REPORT")
    print("=" * 62)

    frame = load_dataset()
    print(f"\nDataset : {len(frame):,} rows x {len(frame.columns)} columns")
    print(f"Features: {len(FEATURES)}")
    print(f"Target  : {TARGET_COLUMN}")
    print("\nClass distribution:")
    print(frame[TARGET_COLUMN].value_counts().to_string())

    print("\n" + "-" * 62)
    print("1. Standard holdout evaluation (random split)")
    print("-" * 62)

    model, results = train_model()

    print(f"Accuracy      : {results['accuracy'] * 100:.2f}%")
    print(f"Confusion matrix (rows = actual, cols = predicted):")
    print("                 predicted")
    print("               Legit  Phish")
    for label, row in zip(("Legit", "Phish"), results["confusion_matrix"]):
        print(f"actual {label}   {row[0]:6d}  {row[1]:5d}")

    print(f"\nFalse negatives: {results['false_negatives']}")
    print(f"False positives: {results['false_positives']}")

    print("\nClassification report:")
    print(results["report"])

    print("-" * 62)
    print("2. Top 10 feature importances")
    print("-" * 62)
    print(results["importance"].head(10).to_string())

    report["standard"] = results

    print("\n" + "-" * 62)
    print("3. Duplicate-aware (strict) holdout evaluation")
    print("-" * 62)

    strict = duplicate_aware_holdout(frame)
    print(f"Rows before cleaning : {strict['rows_before']:,}")
    print(f"Rows after cleaning  : {strict['rows_after']:,}")
    print(f"Ambiguous groups dropped: {strict['conflicting_groups_removed']:,}")
    print(f"Strict accuracy      : {strict['accuracy'] * 100:.2f}%")

    report["strict"] = strict

    print("\n" + "-" * 62)
    print("4. Cross-validation")
    print("-" * 62)

    print("4a. Standard 5-fold CV on the full dataset (optimistic)")
    optimistic_scores = standard_cross_validate(frame)
    for index, score in enumerate(optimistic_scores, start=1):
        print(f"  fold {index}: {score:.4f}")
    print(f"  mean    : {optimistic_scores.mean() * 100:.2f}%")
    print(f"  std dev : {optimistic_scores.std() * 100:.2f} percentage points")

    print("\n4b. 5-fold CV on cleaned unique patterns (strict)")
    scores = strict_cross_validate(frame)
    for index, score in enumerate(scores, start=1):
        print(f"  fold {index}: {score:.4f}")
    print(f"  mean    : {scores.mean() * 100:.2f}%")
    print(f"  std dev : {scores.std() * 100:.2f} percentage points")

    report["cv_scores"] = scores
    report["cv_scores_optimistic"] = optimistic_scores

    print("\n" + "-" * 62)
    print("5. Error analysis (false negatives)")
    print("-" * 62)

    errors = results["errors"]
    print(f"False negatives: {errors['false_negative_count']}")
    if not errors["comparison"].empty:
        print(errors["comparison"].to_string())

    if save:
        path = save_model(model)
        print(f"\nModel saved to: {path}")

    print("=" * 62)

    return report
