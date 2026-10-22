"""Tests for the evaluation helpers."""

import numpy as np
import pytest

from phishing_detection.config import FEATURES, TARGET_COLUMN
from phishing_detection.evaluate import (
    build_model,
    deduplicate_dataset,
    duplicate_aware_holdout,
    error_analysis,
    evaluate_model,
    find_conflicting_groups,
    standard_cross_validate,
    strict_cross_validate,
)


def test_build_model_defaults():
    model = build_model()
    assert model.n_estimators == 100
    assert model.random_state == 42


def test_find_conflicting_groups(dataset):
    conflicting = find_conflicting_groups(dataset)
    assert len(conflicting) == 64


def test_deduplicate_dataset_removes_ambiguity(dataset):
    cleaned, conflicting = deduplicate_dataset(dataset)

    assert len(conflicting) == 64
    assert len(cleaned) == 5721
    assert TARGET_COLUMN in cleaned.columns

    # No feature vector may appear twice after cleaning.
    assert not cleaned.duplicated(subset=FEATURES).any()

    # No surviving feature vector may carry more than one label.
    label_counts = cleaned.groupby(FEATURES)[TARGET_COLUMN].nunique()
    assert label_counts.max() == 1


def test_evaluate_model_reports_metrics(small_model, dataset):
    from sklearn.model_selection import train_test_split

    X = dataset[FEATURES]
    y = dataset[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    small_model.fit(X_train, y_train)
    results = evaluate_model(small_model, X_test, y_test)

    assert 0.9 < results["accuracy"] < 1.0
    assert results["confusion_matrix"].shape == (2, 2)
    assert results["false_negatives"] >= 0
    assert results["false_positives"] >= 0
    assert "Phishing" in results["report"]


def test_duplicate_aware_holdout(dataset):
    results = duplicate_aware_holdout(dataset, n_estimators=10)

    assert results["rows_before"] == 11055
    assert results["rows_after"] == 5721
    assert results["conflicting_groups_removed"] == 64
    # The strict score must be lower than the optimistic one.
    assert results["accuracy"] < 0.98


def test_strict_cross_validate_is_stable(dataset):
    scores = strict_cross_validate(dataset, n_estimators=10)

    assert len(scores) == 5
    assert scores.mean() > 0.9
    # The strict folds should agree far better than the optimistic ones.
    assert scores.std() < 0.05


def test_standard_cross_validate_shape(dataset):
    scores = standard_cross_validate(dataset, n_estimators=10)
    assert len(scores) == 5
    assert scores.mean() > 0.9


def test_error_analysis_reports_false_negatives(small_model, dataset):
    from sklearn.model_selection import train_test_split

    X = dataset[FEATURES]
    y = dataset[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    small_model.fit(X_train, y_train)
    predictions = small_model.predict(X_test)

    errors = error_analysis(small_model, X_test, y_test, predictions)

    assert errors["false_negative_count"] >= 0
    if errors["false_negative_count"]:
        assert not errors["comparison"].empty
        assert "Difference" in errors["comparison"].columns


def test_error_analysis_handles_no_false_negatives(small_model, dataset):
    X = dataset[FEATURES]
    y = dataset[TARGET_COLUMN]

    # Pretend every phishing row was caught.
    predictions = y.to_numpy()

    errors = error_analysis(small_model, X, y, np.asarray(predictions))
    assert errors["false_negative_count"] == 0
    assert errors["comparison"].empty
