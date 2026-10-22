"""Shared pytest fixtures."""

from pathlib import Path

import pytest

from phishing_detection.config import PROJECT_ROOT


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def dataset():
    """The full dataset, loaded once for the whole session."""
    from phishing_detection.data import load_dataset

    return load_dataset()


@pytest.fixture(scope="session")
def small_model(dataset):
    """A quickly-trained model, used where the exact accuracy is irrelevant."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    from phishing_detection.config import FEATURES, TARGET_COLUMN

    X = dataset[FEATURES]
    y = dataset[TARGET_COLUMN]

    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    return model
