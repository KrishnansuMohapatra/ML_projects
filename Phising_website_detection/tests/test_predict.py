"""Tests for model loading and prediction."""

import numpy as np
import pandas as pd
import pytest

from phishing_detection.config import FEATURES
from phishing_detection.predict import load_model, predict_dataframe


def test_load_model_reads_the_shipped_artifact():
    model = load_model()
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")


def test_load_model_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "missing.pkl")


def test_predict_dataframe_labels(small_model):
    frame = pd.DataFrame(
        [[-1, -1, 1, 1, -1, -1, 1, 1, 1, 1,
          0, 1, 1, 1, 1, 1, -1, -1, -1, -1,
          -1, -1, -1, 1, 1, 1, 1, 1, 1, 1]],
        columns=FEATURES,
    )

    result = predict_dataframe(small_model, frame)

    assert "prediction" in result.columns
    assert result["prediction"].iloc[0] in {"Phishing", "Legitimate"}
    assert "phishing_probability" in result.columns
    assert 0.0 <= result["phishing_probability"].iloc[0] <= 1.0


def test_predict_dataframe_requires_all_features(small_model):
    frame = pd.DataFrame({feature: [0] for feature in FEATURES[:-1]})

    with pytest.raises(KeyError):
        predict_dataframe(small_model, frame)


def test_predict_dataframe_rejects_invalid_values(small_model):
    frame = pd.DataFrame(
        [[5] * len(FEATURES)],
        columns=FEATURES,
    )

    with pytest.raises(ValueError):
        predict_dataframe(small_model, frame)


def test_predict_dataframe_rejects_non_numeric(small_model):
    frame = pd.DataFrame(
        [["abc"] * len(FEATURES)],
        columns=FEATURES,
    )

    with pytest.raises(ValueError):
        predict_dataframe(small_model, frame)


def test_predict_dataframe_preserves_extra_columns(small_model):
    frame = pd.DataFrame(
        [[-1] * len(FEATURES)],
        columns=FEATURES,
    )
    frame["result"] = 1

    result = predict_dataframe(small_model, frame)

    assert "result" in result.columns
    assert len(result) == 1
