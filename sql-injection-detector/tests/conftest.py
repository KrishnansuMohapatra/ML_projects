"""Shared pytest fixtures.

The suite runs against the real artifact when one exists (``models/*.pkl``).
If none is present it trains a tiny in-memory model so tests never require a
network download or a checked-in binary. Set ``SQLI_TEST_MODEL=real`` to skip
the stub entirely.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.config import Settings  # noqa: E402
from api.dependencies import get_limiter  # noqa: E402
from api.main import create_app  # noqa: E402

DEFAULT_MODEL = PROJECT_ROOT / "models" / "sql_injection_model_v4.pkl"

TRAIN_TEXTS = [
    "' OR '1'='1",
    "' OR 1=1 --",
    "admin'--",
    "' UNION SELECT NULL --",
    "'; DROP TABLE users; --",
    "' AND SLEEP(5) --",
    "1' AND 1=1 #",
    "%27%20OR%201%3D1--",
    "Please select a product",
    "Select a country",
    "Union membership benefits",
    "Update my profile",
    "Delete my account",
    "SELECT * FROM users WHERE id = 1",
    "The SQL SELECT statement retrieves records",
    "Contact the administrator",
]
TRAIN_LABELS = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]


def _train_stub_model(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    features = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), sublinear_tf=True)
    matrix = features.fit_transform(TRAIN_TEXTS)
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(matrix, TRAIN_LABELS)

    joblib.dump(
        {
            "features": features,
            "model": model,
            "version": "test-stub",
            "labels": {0: "Normal Input", 1: "SQL Injection"},
            "normalization": "url-decode-x2",
        },
        path,
    )
    return path


@pytest.fixture(scope="session")
def model_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Path to a loadable model artifact (real one if available)."""
    if os.getenv("SQLI_TEST_MODEL", "auto") == "real" and not DEFAULT_MODEL.exists():
        pytest.skip("SQLI_TEST_MODEL=real but no model artifact is present")

    if DEFAULT_MODEL.exists():
        return DEFAULT_MODEL

    return _train_stub_model(tmp_path_factory.mktemp("models") / "stub_model.pkl")


@pytest.fixture
def settings(model_path: Path) -> Settings:
    return Settings(
        model_path=model_path,
        environment="test",
        api_keys=[],
        rate_limit_per_minute=0,
        _env_file=None,
    )


@pytest.fixture
def app(settings: Settings):
    get_limiter().reset()
    application = create_app(settings)
    with TestClient(application) as client:
        # Entering the context runs lifespan, which loads the model.
        yield application
        client.close()


@pytest.fixture
def client(app) -> TestClient:
    return TestClient(app)


@pytest.fixture
def api(client: TestClient) -> str:
    return "/api/v1"
