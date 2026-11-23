"""Cheap, explainable heuristics layered on top of the ML prediction.

The classifier answers "does this look like SQL injection?". These rules add
*why* it looks that way, in language a human reviewer can act on. They never
override the model — they are advisory signals returned alongside it.
"""

from __future__ import annotations

import re
from typing import Iterable

# (identifier, human-readable description, compiled pattern, weight)
_PATTERNS: tuple[tuple[str, str, re.Pattern[str], float], ...] = (
    (
        "boolean_tautology",
        "Always-true boolean expression (classic ' OR '1'='1')",
        re.compile(
            r"(?:\bor\b|\band\b)\s+\(*\s*(?:'[^']*'?|\d+)\s*\)*\s*=\s*\(*\s*(?:'[^']*'?|\d+)\s*\)*(?=\s|$|--|#|;|/\*)",
            re.IGNORECASE,
        ),
        0.9,
    ),
    (
        "sql_comment",
        "SQL comment sequence used to truncate a query",
        re.compile(r"(?:--|#|/\*).*$", re.MULTILINE),
        0.5,
    ),
    (
        "union_select",
        "UNION-based data extraction",
        re.compile(r"\bunion\b\s+(?:all\s+)?\bselect\b", re.IGNORECASE),
        0.9,
    ),
    (
        "stacked_query",
        "Stacked/terminated statement (; followed by another SQL keyword)",
        re.compile(
            r";\s*(?:select|insert|update|delete|drop|alter|create|truncate|exec)\b",
            re.IGNORECASE,
        ),
        0.8,
    ),
    (
        "destructive_statement",
        "Destructive DDL/DML keyword",
        re.compile(
            r"\b(?:drop|truncate|alter)\s+(?:table|database|schema|user)\b",
            re.IGNORECASE,
        ),
        0.8,
    ),
    (
        "time_based",
        "Time-based blind injection function",
        re.compile(r"\b(?:sleep|pg_sleep|benchmark|waitfor\s+delay)\s*\(", re.IGNORECASE),
        0.9,
    ),
    (
        "error_based",
        "Error-based extraction helper",
        re.compile(
            r"\b(?:extractvalue|updatexml|load_file|into\s+outfile|into\s+dumpfile)\b",
            re.IGNORECASE,
        ),
        0.7,
    ),
    (
        "information_schema",
        "Reads database metadata",
        re.compile(r"\b(?:information_schema|pg_catalog|sysobjects|sqlite_master)\b", re.IGNORECASE),
        0.6,
    ),
    (
        "quote_breakout",
        "Quote followed by SQL syntax (attempt to break out of a string literal)",
        re.compile(r"['\"]\s*(?:or|and|union|;|--|#|/\*)", re.IGNORECASE),
        0.7,
    ),
    (
        "encoded_payload",
        "Percent-encoded SQL punctuation",
        re.compile(r"%(?:27|22|3b|2d|2d|20|3d)", re.IGNORECASE),
        0.4,
    ),
    (
        "system_command",
        "Shell/OS command execution attempt",
        re.compile(r"\b(?:xp_cmdshell|exec(?:ute)?\s+xp_|sp_oacreate|system)\b", re.IGNORECASE),
        0.8,
    ),
)


def extract_signals(raw_text: str, *, limit: int = 5) -> list[dict[str, object]]:
    """Return matching heuristics sorted by weight (strongest first)."""
    matches: list[dict[str, object]] = []

    for identifier, description, pattern, weight in _PATTERNS:
        found = pattern.search(raw_text)
        if found is None:
            continue
        matches.append(
            {
                "id": identifier,
                "description": description,
                "matched_text": found.group(0)[:80],
                "weight": weight,
            }
        )

    matches.sort(key=lambda item: item["weight"], reverse=True)  # type: ignore[arg-type]
    return matches[:limit]


def rules_score(signals: Iterable[dict[str, object]]) -> float:
    """Combined heuristic confidence in ``[0, 1]`` (noisy-OR, capped)."""
    score = 0.0
    for signal in signals:
        score = 1.0 - (1.0 - score) * (1.0 - float(signal["weight"]))
    return round(score, 4)
