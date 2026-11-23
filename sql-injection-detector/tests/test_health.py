"""Health, readiness and metadata endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_root_lists_entrypoints(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["api"] == "/api/v1"
    assert "version" in body


def test_health_reports_loaded_model(client: TestClient, api: str) -> None:
    response = client.get(f"{api}/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_version"]


def test_root_health_alias(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_ready(client: TestClient, api: str) -> None:
    response = client.get(f"{api}/ready")
    assert response.status_code == 200
    assert response.json()["ready"] is True


def test_model_info_exposes_limits_and_security(client: TestClient, api: str) -> None:
    response = client.get(f"{api}/model")
    assert response.status_code == 200
    body = response.json()
    assert body["model"]["loaded"] is True
    assert body["limits"]["max_batch_size"] >= 1
    assert body["security"]["auth_enabled"] is False


def test_request_id_headers_are_echoed(client: TestClient, api: str) -> None:
    response = client.get(f"{api}/health", headers={"X-Request-ID": "abc123"})
    assert response.headers["X-Request-ID"] == "abc123"
    assert float(response.headers["X-Process-Time-Ms"]) >= 0


def test_request_id_is_generated_when_missing(client: TestClient, api: str) -> None:
    response = client.get(f"{api}/health")
    assert len(response.headers["X-Request-ID"]) >= 8


def test_openapi_schema_is_valid(client: TestClient) -> None:
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert "/api/v1/predict" in schema["paths"]
    assert "/api/v1/predict/batch" in schema["paths"]
    assert "/api/v1/files/analyze" in schema["paths"]
