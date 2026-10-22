"""Tests for dataset loading and validation."""

import pandas as pd
import pytest

from phishing_detection.config import FEATURES, TARGET_COLUMN
from phishing_detection.data import (
    load_dataset,
    split_features_target,
    validate_feature_columns,
)


def test_dataset_loads(dataset):
    assert len(dataset) == 11055
    assert len(dataset.columns) == 31


def test_dataset_has_all_features(dataset):
    missing = validate_feature_columns(dataset)
    assert missing == []
    assert len(FEATURES) == 30


def test_dataset_target_values(dataset):
    assert set(dataset[TARGET_COLUMN].unique()) == {-1, 1}
    assert dataset[TARGET_COLUMN].value_counts().to_dict() == {
        1: 6157,
        -1: 4898,
    }


def test_dataset_has_no_missing_values(dataset):
    assert dataset.isna().sum().sum() == 0


def test_feature_values_are_encoded(dataset):
    observed = set(dataset[FEATURES].to_numpy().ravel())
    assert observed <= {-1, 0, 1}


def test_split_features_target(dataset):
    X, y = split_features_target(dataset)
    assert TARGET_COLUMN not in X.columns
    assert list(X.columns) == FEATURES
    assert len(X) == len(y) == len(dataset)


def test_split_requires_target_column():
    frame = pd.DataFrame({feature: [0] for feature in FEATURES})
    with pytest.raises(KeyError):
        split_features_target(frame)


def test_validate_feature_columns_reports_missing():
    frame = pd.DataFrame({feature: [0] for feature in FEATURES[:-1]})
    assert validate_feature_columns(frame) == [FEATURES[-1]]


def test_load_dataset_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "nope.csv")
