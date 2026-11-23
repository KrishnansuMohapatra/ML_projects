"""Single and batch prediction behaviour."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

INJECTIONS = ["' OR '1'='1", "' OR 1=1 --", "'; DROP TABLE users; --", "%27%20OR%201%3D1--"]
NORMALS = ["Please select a product", "Union membership benefits"]


@pytest.mark.parametrize("query", INJECTIONS)
def test_obvious_injections_are_flagged(client: TestClient, api: str, query: str) -> None:
    response = client.post(f"{api}/predict", json={"query": query})
    assert response.status_code == 200
    result = response.json()["result"]
    assert result["is_injection"] is True
    assert result["verdict"] == "suspicious"
    assert result["injection_probability"] > 0.5
    assert result["risk_level"] in {"medium", "high"}


@pytest.mark.parametrize("query", NORMALS)
def test_normal_text_is_not_flagged(client: TestClient, api: str, query: str) -> None:
    response = client.post(f"{api}/predict", json={"query": query})
    assert response.status_code == 200
    result = response.json()["result"]
    assert result["is_injection"] is False
    assert result["verdict"] == "normal"


def test_response_envelope_fields(client: TestClient, api: str) -> None:
    body = client.post(f"{api}/predict", json={"query": "hello world"}).json()
    assert set(body) == {
        "model_name",
        "model_version",
        "threshold",
        "processing_time_ms",
        "result",
    }
    assert body["processing_time_ms"] >= 0
    assert 0.0 <= body["threshold"] <= 1.0


def test_url_encoded_payload_is_normalised(client: TestClient, api: str) -> None:
    body = client.post(f"{api}/predict", json={"query": "%27%20OR%201%3D1--"}).json()
    result = body["result"]
    assert result["query"] == "%27%20OR%201%3D1--"
    assert "' OR 1=1--" == result["normalized_query"]
    assert result["is_injection"] is True


def test_whitespace_is_collapsed(client: TestClient, api: str) -> None:
    body = client.post(f"{api}/predict", json={"query": "  select    a   from b  "}).json()
    assert body["result"]["normalized_query"] == "select a from b"


def test_explanation_none_returns_no_signals(client: TestClient, api: str) -> None:
    body = client.post(
        f"{api}/predict", json={"query": "' OR 1=1 --", "explain": "none"}
    ).json()
    assert body["result"]["signals"] == []


def test_explanation_full_returns_heuristic_signals(client: TestClient, api: str) -> None:
    body = client.post(
        f"{api}/predict", json={"query": "' OR 1=1 --", "explain": "full"}
    ).json()
    signals = body["result"]["signals"]
    assert signals, "expected at least one heuristic signal"
    ids = {signal["id"] for signal in signals}
    assert "boolean_tautology" in ids
    for signal in signals:
        assert {"id", "description", "matched_text", "weight"} <= set(signal)


def test_signals_are_ranked_by_weight(client: TestClient, api: str) -> None:
    body = client.post(
        f"{api}/predict", json={"query": "'; DROP TABLE users; --", "explain": "full"}
    ).json()
    weights = [signal["weight"] for signal in body["result"]["signals"]]
    assert weights == sorted(weights, reverse=True)


def test_empty_query_is_rejected(client: TestClient, api: str) -> None:
    response = client.post(f"{api}/predict", json={"query": "   "})
    assert response.status_code == 422
    assert response.json()["error"] == "validation_error"


def test_missing_field_is_rejected(client: TestClient, api: str) -> None:
    response = client.post(f"{api}/predict", json={})
    assert response.status_code == 422


def test_unknown_field_is_rejected(client: TestClient, api: str) -> None:
    response = client.post(f"{api}/predict", json={"query": "x", "bogus": 1})
    assert response.status_code == 422


def test_oversized_query_is_rejected(client: TestClient, api: str, app) -> None:
    limit = app.state.settings.max_query_length
    response = client.post(f"{api}/predict", json={"query": "a" * (limit + 1)})
    assert response.status_code == 422
    assert response.json()["error"] == "invalid_input"


# ---------------------------------------------------------------- batch


def test_batch_preserves_order_and_counts(client: TestClient, api: str) -> None:
    queries = INJECTIONS[:2] + NORMALS
    body = client.post(f"{api}/predict/batch", json={"queries": queries}).json()

    assert body["count"] == len(queries)
    assert [item["query"] for item in body["results"]] == queries
    assert body["flagged_count"] == 2
    assert body["results"][0]["is_injection"] is True
    assert body["results"][-1]["is_injection"] is False


def test_batch_rejects_empty_list(client: TestClient, api: str) -> None:
    response = client.post(f"{api}/predict/batch", json={"queries": []})
    assert response.status_code == 422


def test_batch_respects_size_limit(client: TestClient, api: str, app) -> None:
    limit = app.state.settings.max_batch_size
    response = client.post(f"{api}/predict/batch", json={"queries": ["x"] * (limit + 1)})
    assert response.status_code == 413
    assert response.json()["error"] == "payload_too_large"


# ------------------------------------------------------- path-style GET


def test_get_predict_path_style(client: TestClient, api: str) -> None:
    response = client.get(f"{api}/predict/hello")
    assert response.status_code == 200
    assert response.json()["result"]["query"] == "hello"


def test_get_explain_returns_bare_prediction(client: TestClient, api: str) -> None:
    response = client.get(f"{api}/predict/{'%27%20OR%201%3D1'} /explain".replace(" ", ""))
    assert response.status_code == 200
    body = response.json()
    assert body["is_injection"] is True
    assert "signals" in body
