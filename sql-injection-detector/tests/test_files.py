"""CSV/TXT upload analysis and report downloads."""

from __future__ import annotations

import csv
import io

import pytest
from fastapi.testclient import TestClient

CSV_BODY = (
    "id,sentence\n"
    "1,\"' OR '1'='1\"\n"
    "2,Please select a product\n"
    "3,' UNION SELECT NULL --\n"
)

TXT_BODY = "' OR 1=1 --\nPlease select a country\n\nDROP TABLE users\n"


def _upload(content: str, name: str) -> dict[str, tuple[str, bytes, str]]:
    media = "text/csv" if name.endswith(".csv") else "text/plain"
    return {"file": (name, content.encode("utf-8"), media)}


def test_csv_analysis_json(client: TestClient, api: str) -> None:
    response = client.post(
        f"{api}/files/analyze", files=_upload(CSV_BODY, "queries.csv")
    )
    assert response.status_code == 200
    body = response.json()

    assert body["metadata"]["column"] == "sentence"
    assert body["metadata"]["summary"]["total"] == 3
    assert body["metadata"]["summary"]["flagged"] == 2
    assert body["metadata"]["summary"]["flagged_percentage"] == pytest.approx(66.67, abs=0.1)
    assert len(body["results"]) == 3


def test_csv_column_autodetection_prefers_query_column(client: TestClient, api: str) -> None:
    body = (
        "notes,sql\n"
        "\"some note\",\"' OR 1=1 --\"\n"
        "\"other note\",\"normal text please\"\n"
    )
    response = client.post(f"{api}/files/analyze", files=_upload(body, "x.csv"))
    assert response.json()["metadata"]["column"] == "sql"


def test_csv_explicit_column_override(client: TestClient, api: str) -> None:
    body = "notes,sql\n\"a\",\"' OR 1=1 --\"\n\"b\",\"hello there\"\n"
    response = client.post(
        f"{api}/files/analyze?column=notes", files=_upload(body, "x.csv")
    )
    assert response.status_code == 200
    assert response.json()["metadata"]["column"] == "notes"
    assert response.json()["metadata"]["summary"]["total"] == 2


def test_unknown_column_is_rejected(client: TestClient, api: str) -> None:
    response = client.post(
        f"{api}/files/analyze?column=missing", files=_upload(CSV_BODY, "x.csv")
    )
    assert response.status_code == 422
    assert "not found" in response.json()["message"]


def test_txt_analysis_skips_blank_lines(client: TestClient, api: str) -> None:
    response = client.post(f"{api}/files/analyze", files=_upload(TXT_BODY, "q.txt"))
    assert response.status_code == 200
    body = response.json()
    assert body["metadata"]["summary"]["total"] == 3
    assert body["metadata"]["summary"]["flagged"] == 2


def test_csv_download(client: TestClient, api: str) -> None:
    response = client.post(
        f"{api}/files/analyze?format=csv", files=_upload(CSV_BODY, "q.csv")
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert "attachment" in response.headers["content-disposition"]

    text = response.content.decode("utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(text)))
    assert len(rows) == 3
    assert set(rows[0]) >= {"input", "verdict", "injection_probability", "risk_level"}


def test_txt_report_download(client: TestClient, api: str) -> None:
    response = client.post(
        f"{api}/files/analyze?format=txt", files=_upload(CSV_BODY, "q.csv")
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    report = response.text
    assert "SQL INJECTION DETECTOR" in report
    assert "Total inputs analyzed: 3" in report
    assert "LIMITATIONS" in report


def test_report_endpoint(client: TestClient, api: str) -> None:
    response = client.post(f"{api}/files/report", files=_upload(TXT_BODY, "q.txt"))
    assert response.status_code == 200
    assert "Source file: q.txt" in response.text


def test_unsupported_extension(client: TestClient, api: str) -> None:
    files = {"file": ("notes.pdf", b"%PDF-1.4", "application/pdf")}
    response = client.post(f"{api}/files/analyze", files=files)
    assert response.status_code == 415
    assert response.json()["error"] == "unsupported_media_type"


def test_missing_file_field(client: TestClient, api: str) -> None:
    response = client.post(f"{api}/files/analyze")
    assert response.status_code == 422


def test_empty_txt_is_rejected(client: TestClient, api: str) -> None:
    response = client.post(f"{api}/files/analyze", files=_upload("\n\n\n", "e.txt"))
    assert response.status_code == 422
    assert "no non-empty lines" in response.json()["message"]


def test_file_size_limit(client: TestClient, api: str, app) -> None:
    limit_mb = app.state.settings.max_file_size_mb
    # A single oversized line trips the size guard before row limits.
    payload = "x" * (limit_mb * 1024 * 1024 + 10)
    response = client.post(f"{api}/files/analyze", files=_upload(payload, "big.txt"))
    assert response.status_code == 413
    assert response.json()["error"] == "payload_too_large"


def test_row_limit(client: TestClient, api: str, app) -> None:
    rows = app.state.settings.max_file_rows + 1
    response = client.post(
        f"{api}/files/analyze", files=_upload("\n".join(["hello"] * rows), "many.txt")
    )
    assert response.status_code == 413
