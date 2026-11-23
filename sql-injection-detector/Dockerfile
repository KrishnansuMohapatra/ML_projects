# syntax=docker/dockerfile:1

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first for better layer caching.
COPY pyproject.toml ./
RUN pip install --upgrade pip && \
    pip install "fastapi>=0.115" "uvicorn[standard]>=0.30" "python-multipart>=0.0.9" \
                "pydantic-settings>=2.4" "joblib>=1.3" "scikit-learn>=1.3" "pandas>=2.0"

COPY api ./api
COPY src ./src
COPY scripts ./scripts
COPY models ./models

# Fail the build loudly if no model artifact was provided — the API is useless
# without one. Supply models/ via the build context or a mounted volume.
RUN python -c "import pathlib,sys; p=pathlib.Path('models/sql_injection_model_v4.pkl'); \
    sys.exit(0 if p.exists() else 'ERROR: missing models/sql_injection_model_v4.pkl')"

ENV SQLI_ENVIRONMENT=production \
    SQLI_MODEL_PATH=models/sql_injection_model_v4.pkl

EXPOSE 8000

# Non-root runtime user.
RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status==200 else 1)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
