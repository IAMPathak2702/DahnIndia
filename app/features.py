"""
Shared feature engineering for the IoT anomaly detection service.

This module is imported by BOTH the training script (train/train_model.py)
and the serving API (app/model.py) so that windows are turned into feature
vectors the exact same way at train time and at inference time. Skew between
these two code paths is the single most common way anomaly-detection
services silently break in production, so there's deliberately only one
implementation of this logic in the whole repo.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

# The continuous sensor channels we build features from. Names match the
# JSON keys the API expects in each reading (see app/schemas.py).
SENSOR_CHANNELS: tuple[str, ...] = (
    "air_temperature",
    "process_temperature",
    "rotational_speed",
    "torque",
    "tool_wear",
)

# Per-channel aggregate stats computed over a window.
# "trend" is last - first, a cheap proxy for drift within the window (e.g.
# tool wear climbing).
# "max_dev" is max(|reading - window median|): unlike mean/std, a robust
# median barely moves when 1 of 10 readings is an outlier, so this stays
# sensitive to a single spike row that mean/std/min/max "average away" when
# the other 9 readings in the window are normal (see README §2).
_STATS = ("mean", "std", "min", "max", "trend", "max_dev")

# Final feature vector = 5 channels x 6 stats + 1 window-length feature = 31 dims.
FEATURE_NAMES: list[str] = [f"{ch}_{stat}" for ch in SENSOR_CHANNELS for stat in _STATS] + [
    "window_length"
]


def _channel_array(
    readings: Sequence[Mapping[str, float]],
    channel: str,
    medians: Mapping[str, float],
) -> np.ndarray:
    """Pull one channel out of a window of readings, imputing missing/None
    values with the training-set median for that channel. This is the
    "handling missing or irregular readings" requirement from the brief:
    a dropped sensor reading doesn't crash the request, it gets a sane
    fallback value instead."""
    fallback = medians[channel]
    values = []
    for r in readings:
        v = r.get(channel, None)
        if v is None:
            values.append(fallback)
        else:
            try:
                values.append(float(v))
            except (TypeError, ValueError):
                values.append(fallback)
    return np.asarray(values, dtype=float)


def extract_features(
    readings: Sequence[Mapping[str, float]],
    medians: Mapping[str, float],
) -> np.ndarray:
    """Turn a window (list of sensor-reading dicts) into a fixed-length
    feature vector, regardless of how many readings are in the window.

    Works for:
      - the canonical training window size (10 rows)
      - shorter/longer windows sent by a client (irregular reading counts)
      - windows with missing values in individual readings
      - a window of length 1 (std/trend/max_dev degrade to 0 rather than NaN)
    """
    if len(readings) == 0:
        raise ValueError("A prediction window must contain at least one reading.")

    feats: list[float] = []
    for channel in SENSOR_CHANNELS:
        arr = _channel_array(readings, channel, medians)
        mean = float(np.mean(arr))
        std = float(np.std(arr)) if len(arr) > 1 else 0.0
        vmin = float(np.min(arr))
        vmax = float(np.max(arr))
        trend = float(arr[-1] - arr[0]) if len(arr) > 1 else 0.0
        max_dev = float(np.max(np.abs(arr - np.median(arr)))) if len(arr) > 1 else 0.0
        feats.extend([mean, std, vmin, vmax, trend, max_dev])

    feats.append(float(len(readings)))
    return np.asarray(feats, dtype=float)


def build_sliding_windows(
    rows: Sequence[Mapping[str, float]],
    window_size: int,
    stride: int = 1,
) -> list[list[Mapping[str, float]]]:
    """Slide a fixed-size window over an ordered sequence of rows.

    AI4I 2020 is tabular (one row = one machine snapshot, no timestamp), so
    per the task brief we treat row order (UDI ascending) as a proxy sensor
    stream and window over it, rather than treating each row as an
    independent classification example.
    """
    windows = []
    for start in range(0, len(rows) - window_size + 1, stride):
        windows.append(list(rows[start : start + window_size]))
    return windows


def sliding_window_sums(values: Sequence[float], window_size: int, stride: int = 1) -> list[float]:
    """Sum of `values` in each sliding window, in O(n) total via a prefix-sum
    array, instead of the naive O(n * window_size) approach of re-summing
    every window from scratch.

    Used by the training script to classify each window as "pure normal" vs
    "contains a failure row" (sum of a 0/1 flag over the window) without
    scanning every row of every window twice (once for `all(...)`, once for
    `any(...)`).
    """
    if window_size <= 0:
        raise ValueError("window_size must be positive.")
    prefix = np.concatenate(([0.0], np.cumsum(values, dtype=float)))
    starts = range(0, len(values) - window_size + 1, stride)
    return [float(prefix[start + window_size] - prefix[start]) for start in starts]
