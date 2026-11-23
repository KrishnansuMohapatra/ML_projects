"""Batch analysis of uploaded CSV/TXT files.

Mirrors the Streamlit "File Analysis" tab but over HTTP: upload a file, pick a
column, and get back either JSON, a results CSV or a plain-text report.
"""

from __future__ import annotations

import csv
import io
import uuid
from typing import Annotated, Literal

import pandas as pd
from fastapi import APIRouter, Depends, File, Query, UploadFile, status
from fastapi.responses import Response, StreamingResponse

from api.config import Settings, get_settings
from api.dependencies import (
    DetectionService,
    get_detection_service,
    rate_limiter,
)
from api.errors import (
    InvalidInputError,
    PayloadTooLargeError,
    UnsupportedMediaTypeError,
)
from api.schemas import (
    BatchFileMetadata,
    Explanation,
    FilePredictionSummary,
    Prediction,
)

router = APIRouter(
    prefix="/files",
    tags=["File Analysis"],
    dependencies=[Depends(rate_limiter)],
)

# Columns auto-detected as "the query column", matching the Streamlit app.
SUGGESTED_COLUMNS = {"sentence", "query", "sql", "text", "input", "payload"}

OutputFormat = Literal["json", "csv", "txt"]


def _decode(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "utf-16", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise InvalidInputError("Could not decode the file as text. Upload UTF-8 or UTF-16.")


def _read_upload(
    raw: bytes,
    filename: str,
    settings: Settings,
) -> tuple[list[str], str]:
    """Return ``(queries, column_name)`` for a CSV or TXT payload."""
    if len(raw) > settings.max_file_size_bytes:
        raise PayloadTooLargeError(
            f"File is {len(raw) / 1024 / 1024:.1f} MB; the limit is "
            f"{settings.max_file_size_mb} MB."
        )

    suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if suffix == "txt":
        lines = [line.strip() for line in _decode(raw).splitlines()]
        queries = [line for line in lines if line]
        if not queries:
            raise InvalidInputError("The text file contains no non-empty lines.")
        column = "input"
    elif suffix == "csv":
        try:
            frame = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")
        except Exception as exc:
            raise InvalidInputError(f"Could not parse the CSV: {exc}") from exc

        if frame.empty or frame.shape[1] == 0:
            raise InvalidInputError("The CSV has no rows or no columns.")
        if frame.columns.duplicated().any():
            raise InvalidInputError("The CSV contains duplicate column names.")

        column = next(
            (
                str(name)
                for name in frame.columns
                if str(name).strip().lower() in SUGGESTED_COLUMNS
            ),
            str(frame.columns[0]),
        )
        series = frame[column].fillna("").astype(str).str.strip()
        queries = [value for value in series.tolist() if value]
        if not queries:
            raise InvalidInputError(f"Column '{column}' contains no non-empty values.")
    else:
        raise UnsupportedMediaTypeError("Upload a .csv or .txt file.")

    if len(queries) > settings.max_file_rows:
        raise PayloadTooLargeError(
            f"File contains {len(queries)} queries; the limit is "
            f"{settings.max_file_rows}. Raise SQLI_MAX_FILE_ROWS to allow more."
        )

    longest = max(len(query) for query in queries)
    if longest > settings.max_query_length:
        raise InvalidInputError(
            f"A query in the file is {longest} characters long; the per-query "
            f"limit is {settings.max_query_length}."
        )

    return queries, column


def _results_csv(predictions: list[Prediction]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        [
            "input",
            "normalized_input",
            "verdict",
            "label",
            "injection_probability",
            "confidence",
            "risk_level",
            "signals",
        ]
    )
    for prediction in predictions:
        writer.writerow(
            [
                prediction.query,
                prediction.normalized_query,
                prediction.verdict.value,
                prediction.label,
                f"{prediction.injection_probability:.6f}",
                f"{prediction.confidence:.6f}",
                prediction.risk_level.value,
                "|".join(signal.id for signal in prediction.signals),
            ]
        )
    # BOM keeps Excel happy with non-ASCII payloads.
    return "\ufeff" + buffer.getvalue()


@router.post(
    "/analyze",
    summary="Analyse an uploaded CSV or TXT file",
    description=(
        "Upload a CSV with a query-like column (sentence/query/sql/text/input/) "
        "payload) or a TXT file with one query per line.\n\n"
        "* `format=json` (default) — full results plus summary statistics\n"
        "* `format=csv` — downloadable results CSV\n"
        "* `format=txt` — downloadable plain-text report\n"
    ),
    responses={
        200: {"description": "Analysis results; body type depends on `format`"},
        413: {"description": "File exceeds the size or row limit"},
        415: {"description": "File is not CSV or TXT"},
    },
)
async def analyze_file(
    file: Annotated[UploadFile, File(description="CSV or TXT file to analyse")],
    service: Annotated[DetectionService, Depends(get_detection_service)],
    settings: Annotated[Settings, Depends(get_settings)],
    format: Annotated[OutputFormat, Query(description="json | csv | txt")] = "json",
    explain: Annotated[Explanation, Query()] = Explanation.basic,
    column: Annotated[
        str | None,
        Query(description="CSV column to analyse; auto-detected when omitted."),
    ] = None,
):
    filename = file.filename or "upload"
    raw = await file.read()

    queries, detected_column = _read_upload(raw, filename, settings)
    if column:
        # Explicit override: re-read only the requested column.
        if filename.lower().endswith(".txt"):
            raise InvalidInputError("The 'column' option only applies to CSV files.")
        frame = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")
        if column not in frame.columns:
            raise InvalidInputError(
                f"Column '{column}' not found. Available: {list(frame.columns)}"
            )
        queries = [
            value
            for value in frame[column].fillna("").astype(str).str.strip().tolist()
            if value
        ][: settings.max_file_rows]
        detected_column = column

    results = service.predict_many(queries, explain=explain)
    predictions = [item.prediction for item in results]
    summary: FilePredictionSummary = service.summarize(results)

    if format == "csv":
        return Response(
            content=_results_csv(predictions).encode("utf-8"),
            media_type="text/csv",
            headers={
                "Content-Disposition": f'attachment; filename="sql_injection_results_{uuid.uuid4().hex[:8]}.csv"'
            },
        )

    if format == "txt":
        report = service.build_report(results, filename=filename)
        return Response(
            content=report.encode("utf-8"),
            media_type="text/plain",
            headers={
                "Content-Disposition": f'attachment; filename="sql_injection_report_{uuid.uuid4().hex[:8]}.txt"'
            },
        )

    metadata = BatchFileMetadata(
        filename=filename,
        size_bytes=len(raw),
        column=detected_column,
        model_name=settings.model_name,
        model_version=service._version,  # noqa: SLF001 - same package
        summary=summary,
    )
    return {
        "metadata": metadata.model_dump(mode="json"),
        "results": [prediction.model_dump(mode="json") for prediction in predictions],
    }


@router.post(
    "/report",
    summary="Download a plain-text report for a CSV or TXT file",
    description="Convenience wrapper around `/files/analyze?format=txt`.",
    response_class=StreamingResponse,
)
async def report_file(
    file: Annotated[UploadFile, File(description="CSV or TXT file to analyse")],
    service: Annotated[DetectionService, Depends(get_detection_service)],
    settings: Annotated[Settings, Depends(get_settings)],
):
    filename = file.filename or "upload"
    queries, _ = _read_upload(await file.read(), filename, settings)
    results = service.predict_many(queries, explain=Explanation.basic)
    report = service.build_report(results, filename=filename)

    return Response(
        content=report.encode("utf-8"),
        media_type="text/plain",
        status_code=status.HTTP_200_OK,
        headers={"Content-Disposition": 'attachment; filename="sql_injection_report.txt"'},
    )
