"""Prediction endpoints: single, batch, threshold sweep and explanations."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query, status

from api.config import Settings, get_settings
from api.dependencies import (
    DetectionService,
    enforce_batch_limits,
    get_detection_service,
    rate_limiter,
    validate_query_length,
)
from api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    Explanation,
    Prediction,
    PredictRequest,
    PredictResponse,
)

router = APIRouter(
    prefix="/predict",
    tags=["Prediction"],
    dependencies=[Depends(rate_limiter)],
)

SettingsDep = Annotated[Settings, Depends(get_settings)]
ServiceDep = Annotated[DetectionService, Depends(get_detection_service)]


@router.post(
    "",
    response_model=PredictResponse,
    summary="Classify a single input",
    description=(
        "Normalises the input (URL-decode up to twice, collapse whitespace), "
        "runs the TF-IDF + Logistic Regression model, and returns a verdict, "
        "probability and optional heuristic signals."
    ),
    responses={422: {"description": "Empty or oversized input"}},
)
def predict(
    payload: PredictRequest,
    service: ServiceDep,
    settings: SettingsDep,
) -> PredictResponse:
    validate_query_length(payload.query, settings)
    return service.predict_one(payload.query, explain=payload.explain)


@router.post(
    "/batch",
    response_model=BatchPredictResponse,
    summary="Classify up to SQLI_MAX_BATCH_SIZE inputs",
    description="Results preserve input order; summary counts span the whole batch.",
    responses={
        413: {"description": "Batch larger than SQLI_MAX_BATCH_SIZE"},
        422: {"description": "Empty batch or oversized item"},
    },
)
def predict_batch(
    payload: BatchPredictRequest,
    service: ServiceDep,
    settings: SettingsDep,
) -> BatchPredictResponse:
    enforce_batch_limits(payload.queries, settings)
    return service.predict_batch(payload.queries, explain=payload.explain)


@router.get(
    "/{query}",
    response_model=PredictResponse,
    summary="Classify input supplied as a path segment",
    description=(
        "Convenience endpoint for shell one-liners and quick checks. Prefer the "
        "POST endpoint for anything containing spaces or slashes."
    ),
    status_code=status.HTTP_200_OK,
)
def predict_path(
    query: str,
    service: ServiceDep,
    settings: SettingsDep,
    explain: Annotated[Explanation, Query(description="none | basic | full")] = Explanation.basic,
) -> PredictResponse:
    validate_query_length(query, settings)
    return service.predict_one(query, explain=explain)


@router.get(
    "/{query}/explain",
    response_model=Prediction,
    summary="Explain a verdict",
    description=(
        "Returns the per-signal breakdown only, leaving out the response "
        "envelope. Useful for dashboards that render 'why was this flagged?'."
    ),
)
def explain(
    query: str,
    service: ServiceDep,
    settings: SettingsDep,
) -> Prediction:
    validate_query_length(query, settings)
    return service.predict_one(query, explain=Explanation.full).result
