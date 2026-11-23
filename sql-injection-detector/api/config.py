"""Application configuration.

Every value can be overridden with an environment variable prefixed with
``SQLI_`` (for example ``SQLI_DECISION_THRESHOLD=0.7``) or through a local
``.env`` file. List-like values accept either a JSON array or a plain
comma-separated string, so both of these work::

    SQLI_CORS_ORIGINS=["https://app.example.com"]
    SQLI_CORS_ORIGINS=https://app.example.com,https://admin.example.com
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "sql_injection_model_v4.pkl"


class Settings(BaseSettings):
    """Runtime configuration for the detection API."""

    model_config = SettingsConfigDict(
        env_prefix="SQLI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Application -------------------------------------------------
    app_name: str = "SQL Injection Detector API"
    app_version: str = "1.0.0"
    environment: str = Field(
        default="development",
        description="Free-form environment label, e.g. development/staging/production.",
    )
    api_v1_prefix: str = "/api/v1"
    log_level: str = "INFO"

    # --- Model -------------------------------------------------------
    model_path: Path = DEFAULT_MODEL_PATH
    model_name: str = "sql-injection-detector-v4"
    decision_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Probability of SQL injection above which an input is flagged.",
    )

    # --- Limits ------------------------------------------------------
    max_query_length: int = Field(default=10_000, ge=1)
    max_batch_size: int = Field(default=500, ge=1)
    max_file_size_mb: int = Field(default=10, ge=1)
    max_file_rows: int = Field(default=5_000, ge=1)

    # --- Security ----------------------------------------------------
    api_keys: list[str] = Field(
        default_factory=list,
        description="When non-empty, callers must send a matching X-API-Key header.",
    )
    rate_limit_per_minute: int = Field(
        default=0,
        ge=0,
        description="Per-key request budget per minute. 0 disables rate limiting.",
    )
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    @field_validator("api_keys", "cors_origins", mode="before")
    @classmethod
    def _split_comma_separated(cls, value: object) -> object:
        """Allow ``A,B`` in addition to a JSON array from the environment."""
        if isinstance(value, str):
            raw = value.strip()
            if not raw:
                return []
            if raw.startswith("["):
                return value  # let pydantic handle JSON
            return [item.strip() for item in raw.split(",") if item.strip()]
        return value

    @field_validator("model_path", mode="before")
    @classmethod
    def _resolve_model_path(cls, value: object) -> object:
        """Resolve relative paths against the project root, not the CWD."""
        if isinstance(value, (str, Path)):
            path = Path(value).expanduser()
            return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()
        return value

    @property
    def auth_enabled(self) -> bool:
        return bool(self.api_keys)

    @property
    def rate_limit_enabled(self) -> bool:
        return self.rate_limit_per_minute > 0

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024

    @property
    def docs_enabled(self) -> bool:
        """Interactive docs are handy in development, closed in production."""
        return self.environment.lower() != "production"


@lru_cache
def get_settings() -> Settings:
    """Cached settings accessor (also usable as a FastAPI dependency)."""
    return Settings()
