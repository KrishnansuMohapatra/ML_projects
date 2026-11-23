"""Health and readiness endpoints.

``/health`` is a liveness probe (the process is up). ``/ready`` is a
readiness probe (the process can actually serve predictions), so orchestration
can hold traffic back until the model is loaded.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, Response, status

from api.config import Settings, get_settings
from api.dependencies import get_registry
from api.ml.registry import ModelRegistry
from api.schemas import HealthResponse, ModelInfoResponse, utcnow

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
    description="Returns 200 while the process is running. Does not require a model.",
)
def health(
    registry: ModelRegistry = Depends(get_registry),
    settings: Settings = Depends(get_settings),
) -> HealthResponse:
    started = getattr(registry, "started_at", None) or time.time()
    model = registry.model
    return HealthResponse(
        status="ok" if registry.is_loaded else "degraded",
        environment=settings.environment,
        model_loaded=registry.is_loaded,
        model_name=settings.model_name,
        model_version=model.version if model else None,
        uptime_seconds=round(time.time() - started, 2),
        timestamp=utcnow(),
    )


@router.get(
    "/ready",
    summary="Readiness probe",
    description="Returns 200 only when a model is loaded and can serve predictions.",
    responses={503: {"description": "Model not loaded yet"}},
)
def ready(
    response: Response,
    registry: ModelRegistry = Depends(get_registry),
    settings: Settings = Depends(get_settings),
) -> dict[str, object]:
    if not registry.is_loaded:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {
            "ready": False,
            "reason": registry.load_error or "model not loaded",
            "model_path": str(settings.model_path),
        }
    return {"ready": True, "model_name": settings.model_name}


@router.get(
    "/model",
    response_model=ModelInfoResponse,
    summary="Describe the served model",
    description="Artifact details plus the active threshold, limits and security posture.",
)
def model_info(
    registry: ModelRegistry = Depends(get_registry),
    settings: Settings = Depends(get_settings),
) -> ModelInfoResponse:
    model = registry.model
    details = model.describe() if model else {"loaded": False}
    details["loaded"] = registry.is_loaded

    return ModelInfoResponse(
        model=details,
        threshold=settings.decision_threshold,
        limits={
            "max_query_length": settings.max_query_length,
            "max_batch_size": settings.max_batch_size,
            "max_file_size_mb": settings.max_file_size_mb,
            "max_file_rows": settings.max_file_rows,
        },
        security={
            "auth_enabled": settings.auth_enabled,
            "rate_limit_per_minute": settings.rate_limit_per_minute,
            "cors_origins": settings.cors_origins,
        },
    )
