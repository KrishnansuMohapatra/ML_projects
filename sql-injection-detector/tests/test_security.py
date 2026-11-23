"""API-key auth, rate limiting and admin guards."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.config import Settings
from api.dependencies import get_limiter
from api.main import create_app


@pytest.fixture
def secured(model_path, monkeypatch):
    monkeypatch.delenv("SQLI_ALLOW_MODEL_RELOAD", raising=False)
    get_limiter().reset()
    settings = Settings(
        model_path=model_path,
        environment="test",
        api_keys=["secret-key-1", "secret-key-2"],
        rate_limit_per_minute=0,
        _env_file=None,
    )
    application = create_app(settings)
    with TestClient(application) as client:
        yield client
        client.close()


@pytest.fixture
def limited(model_path, monkeypatch):
    monkeypatch.delenv("SQLI_ALLOW_MODEL_RELOAD", raising=False)
    get_limiter().reset()
    settings = Settings(
        model_path=model_path,
        environment="test",
        api_keys=["k1"],
        rate_limit_per_minute=3,
        _env_file=None,
    )
    application = create_app(settings)
    with TestClient(application) as client:
        yield client
        client.close()
    get_limiter().reset()


def test_missing_key_is_rejected(secured: TestClient) -> None:
    response = secured.post("/api/v1/predict", json={"query": "hello"})
    assert response.status_code == 401
    assert response.json()["error"] == "unauthorized"


def test_wrong_key_is_rejected(secured: TestClient) -> None:
    response = secured.post(
        "/api/v1/predict", json={"query": "hello"}, headers={"X-API-Key": "nope"}
    )
    assert response.status_code == 401


def test_valid_key_is_accepted(secured: TestClient) -> None:
    response = secured.post(
        "/api/v1/predict", json={"query": "hello"}, headers={"X-API-Key": "secret-key-1"}
    )
    assert response.status_code == 200


def test_second_configured_key_also_works(secured: TestClient) -> None:
    response = secured.post(
        "/api/v1/predict", json={"query": "hello"}, headers={"X-API-Key": "secret-key-2"}
    )
    assert response.status_code == 200


def test_health_stays_open_for_probes(secured: TestClient) -> None:
    # Orchestrators must be able to probe without credentials.
    assert secured.get("/health").status_code == 200
    assert secured.get("/api/v1/ready").status_code == 200


def test_rate_limit_eventually_blocks(limited: TestClient) -> None:
    headers = {"X-API-Key": "k1"}
    codes = [
        limited.post("/api/v1/predict", json={"query": "hello"}, headers=headers).status_code
        for _ in range(4)
    ]
    assert codes[:3] == [200, 200, 200]
    assert codes[3] == 429

    body = limited.post("/api/v1/predict", json={"query": "hello"}, headers=headers).json()
    assert body["error"] == "rate_limit_exceeded"
    assert "Retry in" in body["message"]


def test_admin_reload_disabled_by_default(secured: TestClient) -> None:
    response = secured.post("/api/v1/admin/reload", headers={"X-API-Key": "secret-key-1"})
    assert response.status_code == 403
    assert response.json()["error"] == "reload_disabled"


def test_admin_reload_requires_key(secured: TestClient, monkeypatch) -> None:
    monkeypatch.setenv("SQLI_ALLOW_MODEL_RELOAD", "true")
    response = secured.post("/api/v1/admin/reload")
    assert response.status_code == 401


def test_admin_reload_works_when_enabled(secured: TestClient, monkeypatch) -> None:
    monkeypatch.setenv("SQLI_ALLOW_MODEL_RELOAD", "true")
    response = secured.post("/api/v1/admin/reload", headers={"X-API-Key": "secret-key-1"})
    assert response.status_code == 200
    assert response.json()["reloaded"] is True


def test_docs_hidden_in_production(model_path) -> None:
    get_limiter().reset()
    application = create_app(
        Settings(model_path=model_path, environment="production", _env_file=None)
    )
    with TestClient(application) as client:
        assert client.get("/docs").status_code == 404
        assert client.get("/openapi.json").status_code == 404
        assert client.get("/health").status_code == 200
