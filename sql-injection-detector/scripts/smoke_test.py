"""Tiny CLI client for the SQL Injection Detector API.

Useful for shell smoke tests and for demoing the service without the docs UI.

Examples::

    uv run python scripts/smoke_test.py                     # against localhost:8000
    uv run python scripts/smoke_test.py --url http://host:8000 --api-key secret
    uv run python scripts/smoke_test.py --file queries.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx

EXAMPLES = [
    "' OR '1'='1",
    "' OR 1=1 --",
    "'; DROP TABLE users; --",
    "' UNION SELECT username, password FROM users --",
    "%27%20OR%201%3D1--",
    "' AND SLEEP(5) --",
    "Please select a product",
    "Union membership benefits",
    "SELECT * FROM users WHERE id = 1",
]


def _headers(api_key: str | None) -> dict[str, str]:
    return {"X-API-Key": api_key} if api_key else {}


def check(base: str, api_key: str | None, timeout: float) -> int:
    headers = _headers(api_key)
    failures = 0

    with httpx.Client(base_url=base, timeout=timeout, headers=headers) as client:
        health = client.get("/health")
        print(f"[{health.status_code}] GET /health -> {health.json().get('status')}")
        if health.status_code != 200:
            return 1

        for query in EXAMPLES:
            response = client.post("/api/v1/predict", json={"query": query})
            if response.status_code != 200:
                print(f"  FAIL {query!r}: HTTP {response.status_code} {response.text[:120]}")
                failures += 1
                continue
            result = response.json()["result"]
            marker = "FLAGGED" if result["is_injection"] else "normal "
            print(
                f"  {marker} p={result['injection_probability']:.3f} "
                f"risk={result['risk_level']:<6} {query!r}"
            )

    return 1 if failures else 0


def analyze_file(base: str, api_key: str | None, path: Path, timeout: float) -> int:
    with path.open("rb") as handle:
        response = httpx.post(
            f"{base}/api/v1/files/analyze",
            files={"file": (path.name, handle)},
            headers=_headers(api_key),
            timeout=timeout,
        )
    if response.status_code != 200:
        print(f"FAIL: HTTP {response.status_code} {response.text[:300]}")
        return 1

    body = response.json()
    summary = body["metadata"]["summary"]
    print(json.dumps(summary, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--api-key", default=None, help="X-API-Key value, if auth is enabled")
    parser.add_argument("--file", type=Path, default=None, help="Analyse a CSV/TXT instead of the canned examples")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    base = args.url.rstrip("/")
    if args.file:
        return analyze_file(base, args.api_key, args.file, args.timeout)
    return check(base, args.api_key, args.timeout)


if __name__ == "__main__":
    sys.exit(main())
