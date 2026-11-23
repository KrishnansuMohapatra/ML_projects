"""Domain exceptions and their HTTP representation.

Routers raise small, explicit exceptions; :func:`register_exception_handlers`
turns them into consistent JSON. That keeps error shapes stable for clients
instead of leaking FastAPI's default ``{"detail": ...}`` in some places and
custom payloads in others.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from api.ml.registry import ModelNotLoadedError

logger = logging.getLogger(__name__)


class APIError(Exception):
    """Base class for errors that map cleanly to an HTTP status code."""

    status_code: int = status.HTTP_400_BAD_REQUEST
    error_code: str = "bad_request"

    def __init__(self, message: str, *, error_code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        if error_code:
            self.error_code = error_code


class InvalidInputError(APIError):
    status_code = 422
    error_code = "invalid_input"


class PayloadTooLargeError(APIError):
    status_code = 413
    error_code = "payload_too_large"


class UnsupportedMediaTypeError(APIError):
    status_code = 415
    error_code = "unsupported_media_type"


class ModelUnavailableError(APIError):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    error_code = "model_unavailable"


def _request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


def register_exception_handlers(app: FastAPI) -> None:
    """Attach JSON handlers for domain, validation and HTTP errors."""

    @app.exception_handler(APIError)
    async def _handle_api_error(request: Request, exc: APIError) -> JSONResponse:
        if exc.status_code >= 500:
            logger.error("%s on %s: %s", exc.error_code, request.url.path, exc.message)
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.error_code,
                "message": exc.message,
                "request_id": _request_id(request),
            },
        )

    @app.exception_handler(ModelNotLoadedError)
    async def _handle_model_missing(request: Request, exc: ModelNotLoadedError) -> JSONResponse:
        logger.error("Inference requested without a loaded model: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "error": "model_unavailable",
                "message": str(exc),
                "request_id": _request_id(request),
            },
        )

    @app.exception_handler(RequestValidationError)
    async def _handle_validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error": "validation_error",
                "message": "Request validation failed.",
                "details": jsonable_encoder(exc.errors()),
                "request_id": _request_id(request),
            },
        )

    @app.exception_handler(StarletteHTTPException)
    async def _handle_http(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": f"http_{exc.status_code}",
                "message": str(exc.detail),
                "request_id": _request_id(request),
            },
            headers=getattr(exc, "headers", None),
        )
