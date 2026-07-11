"""Unit tests for the shared feature-engineering helpers in app/features.py."""

from __future__ import annotations

import numpy as np

from app.features import FEATURE_NAMES, build_sliding_windows, extract_features, sliding_window_sums


def test_sliding_window_sums_matches_naive_per_window_sum() -> None:
    values = [0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    window_size, stride = 3, 1

    fast = sliding_window_sums(values, window_size=window_size, stride=stride)
    naive = [
        sum(values[start : start + window_size])
        for start in range(0, len(values) - window_size + 1, stride)
    ]

    assert fast == naive


def test_sliding_window_sums_respects_stride() -> None:
    values = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    result = sliding_window_sums(values, window_size=2, stride=2)
    assert result == [1.0, 1.0, 1.0]


def test_build_sliding_windows_and_sliding_window_sums_agree_on_length() -> None:
    rows = [{"machine_failure": float(i % 2)} for i in range(20)]
    windows = build_sliding_windows(rows, window_size=4, stride=1)
    sums = sliding_window_sums([r["machine_failure"] for r in rows], window_size=4, stride=1)
    assert len(windows) == len(sums)


def test_extract_features_handles_missing_and_short_windows() -> None:
    medians = {
        "air_temperature": 300.0,
        "process_temperature": 310.0,
        "rotational_speed": 1500.0,
        "torque": 40.0,
        "tool_wear": 100.0,
    }
    single_reading = [{"air_temperature": 298.1}]
    feats = extract_features(single_reading, medians)

    assert feats.shape == (len(FEATURE_NAMES),) == (31,)
    # std/trend/max_dev degrade to 0 for a window of length 1 rather than NaN.
    assert not np.isnan(feats).any()


def test_extract_features_max_dev_catches_a_single_spike_row() -> None:
    """mean/std/min/max are all diluted when 9 of 10 readings are normal and
    1 is a spike -- max_dev (max |reading - window median|) is designed to
    stay sensitive to exactly that case (see README §2)."""
    medians = {
        "air_temperature": 300.0,
        "process_temperature": 310.0,
        "rotational_speed": 1500.0,
        "torque": 40.0,
        "tool_wear": 100.0,
    }
    normal_torque = 40.0
    spike_torque = 400.0
    window = [{"torque": normal_torque}] * 9 + [{"torque": spike_torque}]

    feats = extract_features(window, medians)
    torque_max_dev = feats[FEATURE_NAMES.index("torque_max_dev")]
    torque_std = feats[FEATURE_NAMES.index("torque_std")]

    assert torque_max_dev == abs(spike_torque - normal_torque)
    # The spike moves std, but max_dev captures the full deviation directly
    # rather than a variance-diluted proxy for it.
    assert torque_max_dev > torque_std
