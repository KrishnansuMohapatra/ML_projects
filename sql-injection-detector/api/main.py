"""FastAPI application factory.

Run locally with::

    uv run uvicorn api.main:app --reload

Interactive docs: ``/docs`` (Swagger) and ``/redoc`` (ReDoc). Both are
disabled automatically when ``SQLI_ENVIRONMENT=production``.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator, Callable

from fastapi import Depends, FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from api.config import Settings, get_settings
from api.dependencies import get_registry
from api.errors import register_exception_handlers
from api.ml.registry import ModelRegistry
from api.routers import admin, files, health, predict
from api.schemas import HealthResponse

logger = logging.getLogger("api")

DESCRIPTION = """
Machine-learning SQL injection detection over HTTP.

The service wraps the project's **TF-IDF + Logistic Regression** model and
adds the pieces a REST client needs: request validation, batching, file
analysis with downloadable reports, heuristic explanations, health/readiness
probes and optional API-key auth with rate limiting.

**Typical flow**

1. `POST /api/v1/predict` with `{"query": "' OR '1'='1"}` for one input.
2. `POST /api/v1/predict/batch` for up to `SQLI_MAX_BATCH_SIZE` inputs.
3. `POST /api/v1/files/analyze` to scan a CSV/TXT export.

> This is a text classifier, not a database firewall. It does not execute the
> inputs it receives. Use parameterized queries as the actual control.
"""

TAGS_METADATA = [
    {"name": "Prediction", "description": "Classify one input or a batch of inputs."},
    {"name": "File Analysis", "description": "Scan uploaded CSV/TXT files and export reports."},
    {"name": "Health", "description": "Liveness, readiness and model metadata."},
    {"name": "Admin", "description": "Operational endpoints (guarded)."},
]


def configure_logging(settings: Settings) -> None:
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the model on start-up and release references on shutdown."""
    settings: Settings = app.state.settings
    registry: ModelRegistry = app.state.registry

    try:
        registry.load()
        registry.warmup()
    except FileNotFoundError as exc:
        # Start anyway so /health and /model stay reachable and explain the
        # problem; /predict returns 503 with the same message.
        logger.error("Model unavailable at start-up: %s", exc)
    except Exception as exc:  # pragma: no cover - corrupt artifact
        logger.exception("Failed to load the model: %s", exc)

    logger.info(
        "%s v%s ready (env=%s, model_loaded=%s)",
        settings.app_name,
        settings.app_version,
        settings.environment,
        registry.is_loaded,
    )
    try:
        yield
    finally:
        app.state.registry = None
        logger.info("Shutting down %s", settings.app_name)


def _add_request_context(app: FastAPI) -> None:
    """Tag every request with an ID and echo it plus timing in the response."""

    @app.middleware("http")
    async def request_context(request: Request, call_next: Callable):  # type: ignore[no-untyped-def]
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        request.state.request_id = request_id
        started = time.perf_counter()

        response = await call_next(request)

        elapsed_ms = (time.perf_counter() - started) * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
        if elapsed_ms > 1000:
            logger.warning(
                "Slow request %s %s took %.0fms",
                request.method,
                request.url.path,
                elapsed_ms,
            )
        return response


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build a fully wired application instance."""
    settings = settings or get_settings()
    configure_logging(settings)

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=DESCRIPTION,
        openapi_tags=TAGS_METADATA,
        lifespan=lifespan,
        docs_url="/docs" if settings.docs_enabled else None,
        redoc_url="/redoc" if settings.docs_enabled else None,
        openapi_url="/openapi.json" if settings.docs_enabled else None,
        contact={"name": "Krishnansu Mohapatra", "url": "https://github.com/KrishnansuMohapatra"},
        license_info={"name": "MIT"},
    )

    registry = ModelRegistry(settings)
    registry.started_at = time.time()  # type: ignore[attr-defined]
    app.state.settings = settings
    app.state.registry = registry

    # Bind the injected Settings to this app instance. Without this, routers
    # would read the module-level cached settings and silently ignore the
    # configuration the app was actually built with.
    app.dependency_overrides[get_settings] = lambda: settings

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=False,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["*"],
            expose_headers=["X-Request-ID", "X-Process-Time-Ms"],
        )

    _add_request_context(app)
    register_exception_handlers(app)

    app.include_router(health.router, prefix=settings.api_v1_prefix)
    app.include_router(predict.router, prefix=settings.api_v1_prefix)
    app.include_router(files.router, prefix=settings.api_v1_prefix)
    app.include_router(admin.router, prefix=settings.api_v1_prefix)

    # Health probes are conventionally unversioned, so expose liveness and
    # readiness at the root as well (orchestrators should not need the prefix).
    @app.get("/health", response_model=HealthResponse, tags=["Health"], include_in_schema=False)
    def root_health(
        registry: ModelRegistry = Depends(get_registry),
        settings: Settings = Depends(get_settings),
    ) -> HealthResponse:
        return health.health(registry=registry, settings=settings)

    @app.get("/ready", tags=["Health"], include_in_schema=False)
    def root_ready(
        response: Response,
        registry: ModelRegistry = Depends(get_registry),
        settings: Settings = Depends(get_settings),
    ) -> dict[str, object]:
        return health.ready(response, registry=registry, settings=settings)

    @app.get("/", include_in_schema=False)
    def root() -> dict[str, str]:
        payload = {
            "service": settings.app_name,
            "version": settings.app_version,
            "docs": "/docs" if settings.docs_enabled else "disabled",
            "api": settings.api_v1_prefix,
            "health": "/health",
        }
        return payload

    if settings.docs_enabled:

        @app.get("/docs-redirect", include_in_schema=False)
        def docs_redirect() -> RedirectResponse:  # pragma: no cover - trivial
            return RedirectResponse("/docs")

    return app


app = create_app()


def main() -> None:  # pragma: no cover - CLI convenience
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":  # pragma: no cover
    main()
