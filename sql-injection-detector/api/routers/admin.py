"""Operational endpoints for the model registry.

Reloading is disabled unless ``SQLI_ALLOW_MODEL_RELOAD=true`` and an API key
is configured, so a public deployment cannot be prodded into re-reading
arbitrary files.
"""

from __future__ import annotations

import os

from fastapi import APIRouter, Depends

from api.config import Settings, get_settings
from api.dependencies import get_registry, require_api_key
from api.errors import APIError
from api.ml.registry import ModelRegistry

router = APIRouter(prefix="/admin", tags=["Admin"])


class ReloadDisabledError(APIError):
    status_code = 403
    error_code = "reload_disabled"


@router.post(
    "/reload",
    summary="Reload the model artifact from disk",
    description=(
        "Requires an API key and `SQLI_ALLOW_MODEL_RELOAD=true`. Useful after "
        "retraining without restarting the process."
    ),
)
def reload_model(
    registry: ModelRegistry = Depends(get_registry),
    settings: Settings = Depends(get_settings),
    _: str = Depends(require_api_key),
) -> dict[str, object]:
    if os.getenv("SQLI_ALLOW_MODEL_RELOAD", "false").lower() not in {"1", "true", "yes"}:
        raise ReloadDisabledError(
            "Model reload is disabled. Set SQLI_ALLOW_MODEL_RELOAD=true to enable it."
        )

    model = registry.reload()
    return {
        "reloaded": True,
        "model_name": settings.model_name,
        "version": model.version,
        "path": str(settings.model_path),
    }
