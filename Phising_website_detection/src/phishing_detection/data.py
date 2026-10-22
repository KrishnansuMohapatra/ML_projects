"""Dataset loading and validation helpers."""

from pathlib import Path

import pandas as pd

from .config import DATASET_PATH, FEATURES, TARGET_COLUMN


def load_dataset(path: str | Path = DATASET_PATH) -> pd.DataFrame:
    """Load the phishing dataset from disk."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. Expected the UCI Phishing "
            f"Websites dataset (phishing_data.csv)."
        )
    return pd.read_csv(path)


def split_features_target(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a dataset into its feature matrix and target column."""
    if TARGET_COLUMN not in frame.columns:
        raise KeyError(
            f"Target column '{TARGET_COLUMN}' not found in the dataset."
        )
    return frame.drop(columns=[TARGET_COLUMN]), frame[TARGET_COLUMN]


def validate_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return the list of required feature columns missing from `frame`."""
    return [feature for feature in FEATURES if feature not in frame.columns]
