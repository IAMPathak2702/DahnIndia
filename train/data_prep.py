"""
Shared data loading / windowing / splitting for anything that trains or
evaluates a model on AI4I 2020 -- the production pipeline
(train/train_model.py) and the algorithm-comparison experiments
(experiments/compare_models.py) both import this so every algorithm sees
IDENTICAL windows and IDENTICAL train/val/test splits. Without this, a
"model comparison" would really be comparing different random data splits,
not different algorithms.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from app.features import (
    SENSOR_CHANNELS,
    build_sliding_windows,
    extract_features,
    sliding_window_sums,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "ai4i2020.csv"

WINDOW_SIZE = 10
STRIDE = 1
RANDOM_STATE = 42

# (train, val, test) fractions of the normal windows; anomaly (failure-
# containing) windows are never trained on, so they only need a val/test cut.
NORMAL_SPLIT = (0.6, 0.2, 0.2)
ANOMALY_SPLIT = (0.5, 0.5)

COLUMN_MAP = {
    "Air temperature [K]": "air_temperature",
    "Process temperature [K]": "process_temperature",
    "Rotational speed [rpm]": "rotational_speed",
    "Torque [Nm]": "torque",
    "Tool wear [min]": "tool_wear",
    "Machine failure": "machine_failure",
}


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns=COLUMN_MAP)
    # UDI is already 1..10000 in order; treat that order as the sensor stream.
    df = df.sort_values("UDI").reset_index(drop=True)
    return df


def build_windows(df: pd.DataFrame) -> tuple[list[Any], list[Any], dict[str, float]]:
    """Return (normal_windows, anomaly_windows, medians). Anomaly windows are
    any WINDOW_SIZE-row window touching >=1 failure row -- see
    app/features.py::sliding_window_sums for the O(n) prefix-sum used to
    classify them instead of an O(n * window_size) per-window scan."""
    medians = {ch: float(df[ch].median()) for ch in SENSOR_CHANNELS}
    raw_rows = df[list(SENSOR_CHANNELS) + ["machine_failure"]].to_dict(orient="records")
    rows = cast(list[dict[str, float]], raw_rows)
    windows = build_sliding_windows(rows, window_size=WINDOW_SIZE, stride=STRIDE)
    failure_counts = sliding_window_sums(
        df["machine_failure"].to_numpy(dtype=float).tolist(), window_size=WINDOW_SIZE, stride=STRIDE
    )
    normal = [w for w, count in zip(windows, failure_counts, strict=True) if count == 0]
    anomaly = [w for w, count in zip(windows, failure_counts, strict=True) if count > 0]
    return normal, anomaly, medians


def three_way_split(
    items: list[Any], fractions: tuple[float, ...], rng: np.random.RandomState
) -> list[list[Any]]:
    """Shuffle `items` and cut it into len(fractions) disjoint chunks sized by
    `fractions` (which should sum to ~1.0). Used instead of sklearn's
    train_test_split, which doesn't cleanly support an N-way split in one call."""
    order = rng.permutation(len(items))
    sizes = [int(round(f * len(items))) for f in fractions]
    sizes[-1] = len(items) - sum(sizes[:-1])  # last chunk absorbs any rounding
    chunks, start = [], 0
    for size in sizes:
        idx = order[start : start + size]
        chunks.append([items[i] for i in idx])
        start += size
    return chunks


def stack_features(windows: list[Any], medians: dict[str, float]) -> np.ndarray:
    return np.stack([extract_features(w, medians) for w in windows])


class Splits:
    """The one canonical train/val/test split every model in this repo is
    fit and scored against, so comparisons across algorithms are apples-to-
    apples. `train_normal` is the only data any model is ever fit on for the
    unsupervised algorithms; `y_val`/`y_test` are only used for threshold
    selection / evaluation / the supervised XGBoost comparison baseline,
    never for fitting the unsupervised models."""

    def __init__(self, random_state: int = RANDOM_STATE) -> None:
        df = load_dataset()
        normal_windows, anomaly_windows, self.medians = build_windows(df)
        self.n_normal_total = len(normal_windows)
        self.n_anomaly_total = len(anomaly_windows)

        rng = np.random.RandomState(random_state)
        train_n, val_n, test_n = three_way_split(normal_windows, NORMAL_SPLIT, rng)
        val_a, test_a = three_way_split(anomaly_windows, ANOMALY_SPLIT, rng)
        self.train_n, self.val_n, self.test_n = train_n, val_n, test_n
        self.val_a, self.test_a = val_a, test_a

        self.X_train = stack_features(train_n, self.medians)
        self.X_val = np.concatenate(
            [stack_features(val_n, self.medians), stack_features(val_a, self.medians)]
        )
        self.y_val = np.concatenate([np.zeros(len(val_n)), np.ones(len(val_a))])
        self.X_test = np.concatenate(
            [stack_features(test_n, self.medians), stack_features(test_a, self.medians)]
        )
        self.y_test = np.concatenate([np.zeros(len(test_n)), np.ones(len(test_a))])
