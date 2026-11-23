"""Text normalisation shared between training and inference.

``src/train_v4.py`` normalises every sentence before fitting (URL-decode up
to twice, then collapse whitespace). Inference *must* apply the exact same
transform or the vectorizer sees out-of-distribution text. The Streamlit app
skips this step, which is why encoded payloads such as
``%27%20OR%201%3D1--`` behave inconsistently there — the API fixes that.
"""

from __future__ import annotations

import re
import unicodedata
from urllib.parse import unquote

_WHITESPACE_RE = re.compile(r"\s+")
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def normalize_text(value: object) -> str:
    """Mirror the training-time normalisation, plus defensive cleanup.

    Steps:
    1. Strip control characters that are meaningless for a text classifier.
    2. Normalise unicode first so full-width ``％27`` becomes ``%27``.
    3. Decode URL-encoded characters at most twice (``%27`` -> ``'``).
    4. Collapse runs of whitespace without touching SQL punctuation.
    """
    text = str(value)
    text = _CONTROL_CHARS_RE.sub(" ", text)

    # NFKC first: full-width "％" must become "%" *before* percent-decoding,
    # otherwise obfuscated payloads slip past the decoder unnoticed.
    text = unicodedata.normalize("NFKC", text)

    for _ in range(2):
        decoded = unquote(text)
        if decoded == text:
            break
        text = decoded

    return _WHITESPACE_RE.sub(" ", text).strip()
