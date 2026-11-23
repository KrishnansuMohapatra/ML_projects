"""Business logic between HTTP handlers and the model.

Routers stay thin: they validate input, call :class:`DetectionService`, and
serialise the result. Everything reusable (batching, thresholds, risk
bucketing, signals, reporting) lives here so it can be unit-tested without a
web server.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone

from api.config import Settings
from api.ml.heuristics import extract_signals, rules_score
from api.ml.preprocessing import normalize_text
from api.ml.registry import ModelRegistry, RawPrediction
from api.schemas import (
    BatchPredictResponse,
    Explanation,
    FilePredictionSummary,
    Prediction,
    PredictResponse,
    RiskLevel,
    Signal,
    Verdict,
)

logger = logging.getLogger(__name__)


@dataclass
class ServicePrediction:
    """A prediction plus the columns needed for exports."""

    prediction: Prediction
    raw: RawPrediction
    input_index: int


class DetectionService:
    """Stateless-ish facade over the model registry and settings."""

    def __init__(self, registry: ModelRegistry, settings: Settings) -> None:
        self.registry = registry
        self.settings = settings

    # -- single -------------------------------------------------------

    def predict_one(self, query: str, explain: Explanation = Explanation.basic) -> PredictResponse:
        started = time.perf_counter()
        predictions = self.predict_many([query], explain=explain)
        return PredictResponse(
            model_name=self.settings.model_name,
            model_version=self._version,
            threshold=self.settings.decision_threshold,
            processing_time_ms=self._elapsed(started),
            result=predictions[0].prediction,
        )

    # -- batch --------------------------------------------------------

    def predict_many(
        self,
        queries: list[str],
        explain: Explanation = Explanation.basic,
    ) -> list[ServicePrediction]:
        normalized = [normalize_text(query) for query in queries]
        raw_predictions = self.registry.predict(normalized)

        results: list[ServicePrediction] = []
        for index, (original, text, raw) in enumerate(zip(queries, normalized, raw_predictions)):
            results.append(
                ServicePrediction(
                    prediction=self._build_prediction(
                        original=original,
                        normalized=text,
                        raw=raw,
                        explain=explain,
                    ),
                    raw=raw,
                    input_index=index,
                )
            )
        return results

    def predict_batch(self, queries: list[str], explain: Explanation = Explanation.basic) -> BatchPredictResponse:
        started = time.perf_counter()
        results = self.predict_many(queries, explain=explain)
        return BatchPredictResponse(
            model_name=self.settings.model_name,
            model_version=self._version,
            threshold=self.settings.decision_threshold,
            processing_time_ms=self._elapsed(started),
            count=len(results),
            flagged_count=sum(1 for item in results if item.raw.is_injection),
            results=[item.prediction for item in results],
        )

    # -- helpers ------------------------------------------------------

    @property
    def _version(self) -> str:
        model = self.registry.model
        return model.version if model else "unavailable"

    @staticmethod
    def _elapsed(started: float) -> float:
        return round((time.perf_counter() - started) * 1000, 3)

    def _build_prediction(
        self,
        *,
        original: str,
        normalized: str,
        raw: RawPrediction,
        explain: Explanation,
    ) -> Prediction:
        probability = raw.probability
        confidence = probability if raw.is_injection else 1.0 - probability

        signals: list[Signal] = []
        if explain is not Explanation.none:
            payload = f"{original} {normalized}"
            limit = 5 if explain is Explanation.full else 2
            for signal in extract_signals(payload, limit=limit):
                signals.append(Signal(**signal))  # type: ignore[arg-type]

        return Prediction(
            query=original,
            normalized_query=normalized,
            verdict=Verdict.suspicious if raw.is_injection else Verdict.normal,
            is_injection=raw.is_injection,
            label=raw.label_name,
            injection_probability=probability,
            confidence=round(confidence, 6),
            risk_level=risk_level(probability, self.settings.decision_threshold),
            signals=signals,
        )

    # -- reporting ----------------------------------------------------

    def summarize(self, results: list[ServicePrediction]) -> FilePredictionSummary:
        total = len(results)
        flagged = sum(1 for item in results if item.raw.is_injection)
        return FilePredictionSummary(
            total=total,
            flagged=flagged,
            normal=total - flagged,
            flagged_percentage=round(flagged / total * 100, 2) if total else 0.0,
        )

    def build_report(
        self,
        results: list[ServicePrediction],
        *,
        filename: str | None = None,
    ) -> str:
        """Plain-text report in the same spirit as the Streamlit app's."""
        summary = self.summarize(results)
        generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        lines = [
            "SQL INJECTION DETECTOR — ANALYSIS REPORT",
            "=" * 48,
            f"Generated: {generated}",
            f"Model: {self.settings.model_name} (v{self._version})",
            f"Threshold: {self.settings.decision_threshold:.2f}",
            "Method: TF-IDF + Logistic Regression",
        ]
        if filename:
            lines.append(f"Source file: {filename}")

        lines += [
            "",
            "SUMMARY",
            "-" * 48,
            f"Total inputs analyzed: {summary.total}",
            f"Flagged as SQL injection: {summary.flagged}",
            f"Classified as normal: {summary.normal}",
            f"Flagged percentage: {summary.flagged_percentage:.2f}%",
            "",
            "FINDINGS",
            "-" * 48,
        ]

        for item in results:
            prediction = item.prediction
            lines.append(f"Input #{item.input_index + 1}: {prediction.query}")
            lines.append(f"Normalized: {prediction.normalized_query}")
            lines.append(f"Verdict: {prediction.verdict.value} ({prediction.label})")
            lines.append(f"Injection probability: {prediction.injection_probability:.4f}")
            lines.append(f"Model confidence: {prediction.confidence:.2%}")
            lines.append(f"Risk level: {prediction.risk_level.value}")
            if prediction.signals:
                joined = "; ".join(signal.id for signal in prediction.signals)
                lines.append(f"Signals: {joined}")
            lines.append("")

        lines += [
            "LIMITATIONS",
            "-" * 48,
            "Predictions come from a machine learning text classifier.",
            "Confidence is not a guarantee of correctness.",
            "A normal prediction does not prove an input is safe.",
            "A flagged prediction does not prove malicious intent.",
            "Always use parameterized queries as the actual control.",
            "",
        ]
        return "\n".join(lines)


def risk_level(probability: float, threshold: float) -> RiskLevel:
    """Map a probability to a coarse bucket relative to the threshold.

    The bands are threshold-aware so raising ``SQLI_DECISION_THRESHOLD`` keeps
    the labels meaningful.
    """
    if probability >= max(threshold, 0.75):
        return RiskLevel.high
    if probability >= max(threshold * 0.6, 0.3):
        return RiskLevel.medium
    return RiskLevel.low


__all__ = ["DetectionService", "ServicePrediction", "risk_level", "rules_score"]
