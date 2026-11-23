"""Model loading and in-process registry.

Trained artifacts come in three shapes across the project's history:

* V1  — a bare ``sklearn`` ``Pipeline`` (``src/train.py``)
* V2/V3 — a ``Pipeline`` whose first step is a TF-IDF ``FeatureUnion``
* V4  — a ``dict`` with ``features``, ``model``, ``labels`` and ``version``

:class:`ModelRegistry` hides those differences behind one ``predict`` call so
the routers never need to care which artifact was loaded.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np

from api.config import Settings

logger = logging.getLogger(__name__)

DEFAULT_LABELS = {0: "Normal Input", 1: "SQL Injection"}


class ModelNotLoadedError(RuntimeError):
    """Raised when inference is attempted before a model is available."""


@dataclass
class RawPrediction:
    """One raw model output, before thresholding."""

    label: int
    label_name: str
    probability: float
    is_injection: bool


@dataclass
class LoadedModel:
    """Normalised view over any supported artifact format."""

    version: str
    labels: dict[int, str]
    path: Path | None = None
    normalization: str = "url-decode-x2; collapse whitespace"
    feature_extractor: Any | None = None
    estimator: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # -- public API ---------------------------------------------------

    def predict(self, texts: Sequence[str], threshold: float) -> list[RawPrediction]:
        """Classify a batch of already-normalised texts."""
        matrix = self.transform(texts)
        probabilities = self.estimator.predict_proba(matrix)  # type: ignore[union-attr]

        # Column index of the positive class; classes_ may be [0, 1] or ["0", "1"].
        classes = [int(c) for c in getattr(self.estimator, "classes_", [0, 1])]
        positive_index = classes.index(1) if 1 in classes else len(classes) - 1
        positive_probs = np.asarray(probabilities, dtype=float)[:, positive_index]

        results: list[RawPrediction] = []
        for probability in positive_probs:
            probability = float(probability)
            is_injection = probability >= threshold
            label = 1 if is_injection else 0
            results.append(
                RawPrediction(
                    label=label,
                    label_name=self.labels.get(label, DEFAULT_LABELS[label]),
                    probability=round(probability, 6),
                    is_injection=is_injection,
                )
            )
        return results

    def transform(self, texts: Sequence[str]) -> Any:
        """Apply the artifact's own feature extraction step."""
        if self.feature_extractor is not None:
            return self.feature_extractor.transform(list(texts))
        return list(texts)

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.metadata.get("name"),
            "version": self.version,
            "labels": {str(key): value for key, value in self.labels.items()},
            "estimator": type(self.estimator).__name__,
            "feature_extractor": type(self.feature_extractor).__name__
            if self.feature_extractor is not None
            else None,
            "normalization": self.normalization,
            "path": str(self.path) if self.path else None,
        }


def _load_from_artifact(artifact: Any, path: Path | None) -> LoadedModel:
    """Translate any supported artifact shape into a :class:`LoadedModel`."""
    if isinstance(artifact, dict):
        estimator = artifact.get("model", artifact.get("estimator"))
        if estimator is None:
            raise ValueError("Artifact dict is missing a 'model' entry.")
        labels = {
            int(key): str(value)
            for key, value in dict(artifact.get("labels", DEFAULT_LABELS)).items()
        }
        return LoadedModel(
            version=str(artifact.get("version", "unknown")),
            labels=labels,
            path=path,
            normalization=str(artifact.get("normalization", "url-decode-x2")),
            feature_extractor=artifact.get("features"),
            estimator=estimator,
        )

    # A Pipeline may already bundle the vectorizer, so only split it when the
    # last step looks like an estimator and an earlier step looks like a
    # vectorizer. Otherwise treat the whole object as the estimator.
    steps = getattr(artifact, "steps", None)
    if steps and len(steps) >= 2:
        estimator = steps[-1][1]
        extractor_parts = [step[1] for step in steps[:-1]]
        extractor = extractor_parts[0] if len(extractor_parts) == 1 else artifact[:-1]
        return LoadedModel(
            version="legacy-pipeline",
            labels=dict(DEFAULT_LABELS),
            path=path,
            feature_extractor=extractor,
            estimator=estimator,
        )

    return LoadedModel(
        version="legacy",
        labels=dict(DEFAULT_LABELS),
        path=path,
        feature_extractor=getattr(artifact, "named_steps", {}).get("tfidf"),
        estimator=artifact,
    )


class ModelRegistry:
    """Thread-safe holder for the currently served model.

    Loading is eager during application start-up (so a broken artifact fails
    fast) but can be repeated via :meth:`reload` — handy after retraining or
    when the model file is swapped by a deployment.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._lock = threading.RLock()
        self._model: LoadedModel | None = None
        self._loaded_at: float | None = None
        self._load_error: str | None = None

    # -- lifecycle ----------------------------------------------------

    def load(self) -> LoadedModel:
        """Load (or reload) the artifact from disk."""
        path = self._settings.model_path
        with self._lock:
            if not path.exists():
                self._load_error = f"Model file not found: {path}"
                self._model = None
                raise FileNotFoundError(self._load_error)

            started = time.perf_counter()
            artifact = joblib.load(path)
            model = _load_from_artifact(artifact, path)
            model.metadata.setdefault("name", self._settings.model_name)

            self._model = model
            self._loaded_at = time.time()
            self._load_error = None

            logger.info(
                "Loaded model %s (%s) in %.2fs from %s",
                self._settings.model_name,
                model.version,
                time.perf_counter() - started,
                path,
            )
            return model

    def reload(self) -> LoadedModel:
        return self.load()

    def warmup(self, sample: str = "test query") -> bool:
        """Run a throwaway prediction so the first real request is fast."""
        try:
            self.predict([sample], threshold=self._settings.decision_threshold)
            return True
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Model warmup failed: %s", exc)
            return False

    # -- access -------------------------------------------------------

    @property
    def model(self) -> LoadedModel | None:
        return self._model

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    @property
    def loaded_at(self) -> float | None:
        return self._loaded_at

    def predict(self, texts: Sequence[str], *, threshold: float | None = None) -> list[RawPrediction]:
        model = self._model
        if model is None:
            raise ModelNotLoadedError(
                "No model is loaded. Train one with `uv run src/train_v4.py` "
                f"or point SQLI_MODEL_PATH at an existing artifact "
                f"(current: {self._settings.model_path})."
            )
        return model.predict(
            texts,
            threshold=self._settings.decision_threshold if threshold is None else threshold,
        )
