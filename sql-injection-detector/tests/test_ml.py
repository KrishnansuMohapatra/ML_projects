"""Unit tests for preprocessing, heuristics, service logic and the registry."""

from __future__ import annotations

import pytest

from api.config import Settings
from api.ml.heuristics import extract_signals, rules_score
from api.ml.preprocessing import normalize_text
from api.ml.registry import ModelRegistry, _load_from_artifact
from api.schemas import Explanation
from api.services import DetectionService, risk_level


# ------------------------------------------------------- preprocessing


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("%27%20OR%201%3D1--", "' OR 1=1--"),
        ("  a   b  ", "a b"),
        ("line\nbreak", "line break"),
        ("café", "café"),
        ("％27OR％201=1", "'OR 1=1"),  # full-width percent/space normalised by NFKC
        ("\x00\x01hello", "hello"),
        ("", ""),
    ],
)
def test_normalize_text(raw: str, expected: str) -> None:
    assert normalize_text(raw) == expected


def test_normalize_text_is_idempotent() -> None:
    once = normalize_text("%2527%2520OR%25201%253D1")
    assert normalize_text(once) == once


def test_normalize_text_handles_non_strings() -> None:
    assert normalize_text(42) == "42"
    assert normalize_text(None) == "None"


# ----------------------------------------------------------- heuristics


def test_boolean_tautology_detected() -> None:
    signals = extract_signals("' OR '1'='1")
    assert any(signal["id"] == "boolean_tautology" for signal in signals)


def test_union_select_detected() -> None:
    ids = {signal["id"] for signal in extract_signals("' UNION SELECT NULL --")}
    assert {"union_select", "sql_comment"} <= ids


def test_stacked_query_detected() -> None:
    ids = {signal["id"] for signal in extract_signals("'; DROP TABLE users; --")}
    assert {"stacked_query", "destructive_statement"} <= ids


def test_time_based_detected() -> None:
    ids = {signal["id"] for signal in extract_signals("' AND SLEEP(5) --")}
    assert "time_based" in ids


def test_encoded_payload_detected() -> None:
    ids = {signal["id"] for signal in extract_signals("%27%20OR%201%3D1--")}
    assert "encoded_payload" in ids


def test_plain_english_has_no_signals() -> None:
    assert extract_signals("Please select a product") == []


def test_signals_respect_limit() -> None:
    payload = "'; DROP TABLE users; -- UNION SELECT NULL FROM information_schema"
    assert len(extract_signals(payload, limit=2)) == 2


def test_rules_score_is_bounded() -> None:
    assert rules_score([]) == 0.0
    assert 0.0 < rules_score(extract_signals("' OR 1=1 --")) <= 1.0


# -------------------------------------------------------------- risk


@pytest.mark.parametrize(
    ("probability", "threshold", "expected"),
    [
        (0.99, 0.5, "high"),
        (0.95, 0.9, "high"),
        (0.6, 0.5, "medium"),
        (0.5, 0.5, "medium"),
        (0.1, 0.5, "low"),
        (0.0, 0.5, "low"),
    ],
)
def test_risk_level(probability: float, threshold: float, expected: str) -> None:
    assert risk_level(probability, threshold).value == expected


# ------------------------------------------------------ artifact shapes


def test_legacy_dict_artifact(model_path) -> None:
    import joblib

    model = _load_from_artifact(joblib.load(model_path), model_path)
    assert model.version
    assert model.labels[1] == "SQL Injection"
    described = model.describe()
    assert described["estimator"] == "LogisticRegression"


def test_unsupported_dict_artifact_raises() -> None:
    with pytest.raises(ValueError):
        _load_from_artifact({"labels": {0: "a"}}, None)


# ------------------------------------------------------------- service


def test_service_respects_custom_threshold(model_path) -> None:
    settings = Settings(model_path=model_path, decision_threshold=1.0, _env_file=None)
    registry = ModelRegistry(settings)
    registry.load()
    service = DetectionService(registry, settings)

    response = service.predict_one("' OR 1=1 --", explain=Explanation.none)
    # Nothing can reach a 1.0 threshold, so everything is "normal".
    assert response.result.is_injection is False
    assert response.result.injection_probability < 1.0


def test_service_confidence_is_for_predicted_class(model_path) -> None:
    settings = Settings(model_path=model_path, _env_file=None)
    registry = ModelRegistry(settings)
    registry.load()
    service = DetectionService(registry, settings)

    result = service.predict_one("' OR 1=1 --", explain=Explanation.none).result
    expected = (
        result.injection_probability if result.is_injection else 1 - result.injection_probability
    )
    assert result.confidence == pytest.approx(expected, abs=1e-6)


def test_service_explain_levels(model_path) -> None:
    settings = Settings(model_path=model_path, _env_file=None)
    registry = ModelRegistry(settings)
    registry.load()
    service = DetectionService(registry, settings)

    none = service.predict_one("' OR 1=1 --", explain=Explanation.none).result
    basic = service.predict_one("' OR 1=1 --", explain=Explanation.basic).result
    full = service.predict_one("' OR 1=1 --", explain=Explanation.full).result

    assert none.signals == []
    assert 0 < len(basic.signals) <= 2
    assert len(full.signals) >= len(basic.signals)


def test_service_report_contains_summary(model_path) -> None:
    settings = Settings(model_path=model_path, _env_file=None)
    registry = ModelRegistry(settings)
    registry.load()
    service = DetectionService(registry, settings)

    results = service.predict_many(["' OR 1=1 --", "hello there"], explain=Explanation.basic)
    report = service.build_report(results, filename="sample.csv")
    assert "Total inputs analyzed: 2" in report
    assert "Source file: sample.csv" in report


def test_registry_raises_when_model_missing(tmp_path) -> None:
    settings = Settings(model_path=tmp_path / "nope.pkl", _env_file=None)
    registry = ModelRegistry(settings)
    with pytest.raises(FileNotFoundError):
        registry.load()
    assert registry.is_loaded is False
    assert "not found" in (registry.load_error or "")
