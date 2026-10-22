"""Phishing website detection with a Random Forest classifier.

Public API:

    from phishing_detection import FEATURES, load_dataset, train_model

"""

from .config import (
    BATCH_DIR,
    DATASET_PATH,
    FEATURES,
    LABEL_NAMES,
    LEGITIMATE_LABEL,
    MODEL_PATH,
    PHISHING_LABEL,
    PROJECT_ROOT,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)
from .data import load_dataset, split_features_target, validate_feature_columns
from .evaluate import (
    duplicate_aware_holdout,
    error_analysis,
    evaluate_model,
    strict_cross_validate,
)
from .predict import predict_dataframe, load_model
from .train import save_model, train_model

__all__ = [
    "BATCH_DIR",
    "DATASET_PATH",
    "FEATURES",
    "LABEL_NAMES",
    "LEGITIMATE_LABEL",
    "MODEL_PATH",
    "PHISHING_LABEL",
    "PROJECT_ROOT",
    "RANDOM_STATE",
    "TARGET_COLUMN",
    "TEST_SIZE",
    "duplicate_aware_holdout",
    "error_analysis",
    "evaluate_model",
    "load_dataset",
    "load_model",
    "predict_dataframe",
    "save_model",
    "split_features_target",
    "strict_cross_validate",
    "train_model",
    "validate_feature_columns",
]

__version__ = "0.1.0"
