"""Minimal single-file SQL Injection Detector API (~130 lines).

Same model as the full `api/` package, stripped to the REST essentials so it
is easy to read and revise from. Covers: pydantic models, request validation,
response_model, status codes, batching, file upload and error handling.

Run:   uv run uvicorn api_minimal:app --reload
Docs:  http://localhost:8000/docs
"""

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).parent / "models" / "sql_injection_model_v4.pkl"
SAVED = joblib.load(MODEL_PATH)          # V4 artifact: {"features", "model", ...}
FEATURES, MODEL = SAVED["features"], SAVED["model"]

app = FastAPI(title="SQL Injection Detector API", version="1.0.0")


# ------------------------------------------------------------------ schemas
class Query(BaseModel):
    """Request body for one prediction."""

    query: str = Field(min_length=1, max_length=10_000, examples=["' OR '1'='1"])
    explain: bool = False


class BatchQuery(BaseModel):
    queries: list[str] = Field(min_length=1, max_length=500)


class Prediction(BaseModel):
    query: str
    is_injection: bool
    label: str
    probability: float
    confidence: float


class BatchResult(BaseModel):
    count: int
    flagged: int
    results: list[Prediction]


# ------------------------------------------------------------- core inference
def score_one(text: str) -> Prediction:
    """Run the model on one string and shape the result."""
    vector = FEATURES.transform([text])
    prediction = int(MODEL.predict(vector)[0])
    probability = float(MODEL.predict_proba(vector)[0][1])

    return Prediction(
        query=text,
        is_injection=bool(prediction),
        label=SAVED["labels"][prediction],
        probability=round(probability, 4),
        # Confidence = probability of the class we actually predicted.
        confidence=round(probability if prediction else 1 - probability, 4),
    )


# ------------------------------------------------------------------ endpoints
@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": SAVED.get("version", "unknown")}


@app.post("/predict", response_model=Prediction)
def predict(payload: Query) -> Prediction:
    result = score_one(payload.query)
    if payload.explain and result.is_injection:
        print(f"flagged: {payload.query!r} p={result.probability}")
    return result


@app.post("/predict/batch", response_model=BatchResult)
def predict_batch(payload: BatchQuery) -> BatchResult:
    results = [score_one(query) for query in payload.queries]
    return BatchResult(
        count=len(results),
        flagged=sum(r.is_injection for r in results),
        results=results,
    )


@app.post("/predict/file", response_model=BatchResult)
async def predict_file(file: UploadFile = File(...)) -> BatchResult:
    """Analyse a CSV (query column) or TXT (one query per line)."""
    if not file.filename.lower().endswith((".csv", ".txt")):
        raise HTTPException(415, "Upload a .csv or .txt file")

    raw = await file.read()
    if len(raw) > 10 * 1024 * 1024:
        raise HTTPException(413, "File exceeds 10 MB")

    if file.filename.lower().endswith(".txt"):
        lines = [line.strip() for line in raw.decode("utf-8").splitlines()]
        queries = [line for line in lines if line]
    else:
        frame = pd.read_csv(pd.io.common.BytesIO(raw))
        if frame.empty:
            raise HTTPException(422, "CSV is empty")
        # Prefer a query-like column, else fall back to the first one.
        preferred = {"sentence", "query", "sql", "text", "input", "payload"}
        column = next(
            (c for c in frame.columns if str(c).lower() in preferred), frame.columns[0]
        )
        queries = frame[column].fillna("").astype(str).str.strip().tolist()
        queries = [q for q in queries if q]

    if not queries:
        raise HTTPException(422, "No queries found in file")
    if len(queries) > 5_000:
        raise HTTPException(413, "Too many rows (max 5,000)")

    results = [score_one(query) for query in queries]
    return BatchResult(
        count=len(results),
        flagged=sum(r.is_injection for r in results),
        results=results,
    )
