from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import joblib
import numpy as np

from app.features import extract_features

DEFAULT_ARTIFACT_PATH = (
    Path(__file__).resolve().parent.parent / "model_artifact" / "anomaly_model.joblib"
)


class AnomalyModel:
    """Thin wrapper around the trained IsolationForest + scaling params.

    Instantiated ONCE at API startup (see app/main.py's lifespan handler),
    not per-request, per the task brief.
    """

    def __init__(self, artifact_path: Path = DEFAULT_ARTIFACT_PATH):
        artifact = joblib.load(artifact_path)
        self._model = artifact["model"]
        self.window_size: int = artifact["window_size"]
        self._medians: Mapping[str, float] = artifact["medians"]
        self._score_min: float = artifact["score_min"]
        self._score_max: float = artifact["score_max"]
        self.threshold: float = artifact["threshold"]

    def _to_unit_score(self, raw_score: float) -> float:
        scaled = (raw_score - self._score_min) / max(self._score_max - self._score_min, 1e-9)
        return float(np.clip(scaled, 0.0, 1.0))

    def score(self, readings: Sequence[Mapping[str, float]]) -> tuple[float, bool]:
        """Score one window of readings. Returns (anomaly_score, is_anomaly)."""
        features = extract_features(readings, self._medians).reshape(1, -1)
        # decision_function: higher = more normal, so flip sign -> higher = more anomalous.
        raw_score = float(-self._model.decision_function(features)[0])
        unit_score = self._to_unit_score(raw_score)
        return unit_score, unit_score > self.threshold
