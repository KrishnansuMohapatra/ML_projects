"""Pydantic request/response contracts.

Keeping these in one module means the OpenAPI schema, the routers and the
tests all agree on the wire format.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------- enums


class Verdict(str, Enum):
    """Coarse outcome of a detection."""

    normal = "normal"
    suspicious = "suspicious"


class RiskLevel(str, Enum):
    """Bucketed risk, derived from the injection probability."""

    low = "low"
    medium = "medium"
    high = "high"


class Explanation(str, Enum):
    """How much detail a caller wants back."""

    none = "none"
    basic = "basic"
    full = "full"


# ----------------------------------------------------------- requests


class PredictRequest(BaseModel):
    """A single input to classify."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {"query": "' OR '1'='1", "explain": "full"},
                {"query": "Please select a product", "explain": "basic"},
            ]
        },
    )

    query: Annotated[str, Field(min_length=1, description="Text or SQL fragment to analyse.")]
    explain: Explanation = Field(
        default=Explanation.basic,
        description="Amount of explanation detail to include.",
    )

    @field_validator("query")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must contain at least one non-whitespace character")
        return value


class BatchPredictRequest(BaseModel):
    """Up to ``SQLI_MAX_BATCH_SIZE`` inputs in one round trip."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "queries": ["' OR 1=1 --", "SELECT * FROM users WHERE id = 1"],
                    "explain": "basic",
                }
            ]
        }
    )

    queries: Annotated[
        list[str],
        Field(min_length=1, description="Inputs to classify; order is preserved in the response."),
    ]
    explain: Explanation = Field(default=Explanation.basic)


# ---------------------------------------------------------- responses


class Signal(BaseModel):
    """A single heuristic rule that matched."""

    id: str
    description: str
    matched_text: str
    weight: float


class Prediction(BaseModel):
    """Result for one input."""

    model_config = ConfigDict(protected_namespaces=())

    query: str = Field(description="The input exactly as received.")
    normalized_query: str = Field(description="Input after URL-decoding and whitespace cleanup.")
    verdict: Verdict
    is_injection: bool
    label: str = Field(description="Human-readable class name from the model artifact.")
    injection_probability: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0, description="Probability of the predicted class.")
    risk_level: RiskLevel
    signals: list[Signal] = Field(default_factory=list)


class PredictResponse(BaseModel):
    """Envelope for a single prediction."""

    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    model_version: str
    threshold: float
    processing_time_ms: float
    result: Prediction


class BatchPredictResponse(BaseModel):
    """Envelope for a batch prediction."""

    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    model_version: str
    threshold: float
    processing_time_ms: float
    count: int
    flagged_count: int
    results: list[Prediction]


class FilePredictionSummary(BaseModel):
    """Aggregate statistics for an uploaded file."""

    total: int
    flagged: int
    normal: int
    flagged_percentage: float


class BatchFileMetadata(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    filename: str
    size_bytes: int
    column: str
    model_name: str
    model_version: str
    summary: FilePredictionSummary


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    environment: str
    model_loaded: bool
    model_name: str
    model_version: str | None = None
    uptime_seconds: float
    timestamp: datetime


class ModelInfoResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model: dict
    threshold: float
    limits: dict
    security: dict


class ErrorResponse(BaseModel):
    """Shape returned by :class:`api.errors.APIError` handlers."""

    error: str
    message: str
    request_id: str | None = None


def utcnow() -> datetime:
    return datetime.now(timezone.utc)
