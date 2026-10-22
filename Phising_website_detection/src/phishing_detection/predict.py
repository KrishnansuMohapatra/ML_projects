"""Model loading and prediction helpers shared by the CLI and the web app."""

from pathlib import Path

import joblib
import pandas as pd

from .config import FEATURES, MODEL_PATH, PHISHING_LABEL
from .data import validate_feature_columns


def load_model(path: str | Path = MODEL_PATH):
    """Load a trained model from disk."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model found at '{path}'. Train one first with: "
            f"uv run phishing-detector train"
        )
    return joblib.load(path)


def predict_dataframe(
    model,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Predict a dataframe of features.

    Returns a copy of `frame` with `prediction` and (when the model supports
    it) `phishing_probability` columns appended.
    """
    missing = validate_feature_columns(frame)
    if missing:
        raise KeyError(f"Missing required feature columns: {missing}")

    X = frame[FEATURES].apply(pd.to_numeric, errors="coerce")

    if X.isna().any().any():
        raise ValueError("Feature values must be numeric and non-empty.")

    invalid_values = sorted(set(X.to_numpy().ravel()) - {-1, 0, 1})
    if invalid_values:
        raise ValueError(
            f"Invalid feature values found: {invalid_values}. "
            f"Only -1, 0, and 1 are allowed."
        )

    X = X.astype(int)

    predictions = model.predict(X)
    result = frame.copy()

    result["prediction"] = [
        "Phishing" if p == PHISHING_LABEL else "Legitimate"
        for p in predictions
    ]

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        class_to_index = {c: i for i, c in enumerate(model.classes_)}
        if PHISHING_LABEL in class_to_index:
            result["phishing_probability"] = probabilities[
                :, class_to_index[PHISHING_LABEL]
            ]

    return result
