"""Threshold selection and metrics helpers shared by train_model.py and
experiments/compare_models.py, so every algorithm is scored the same way."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)


def best_threshold_by_f1(
    y_true: np.ndarray, scores: np.ndarray
) -> tuple[float, float, float, float]:
    """Sweep the precision-recall curve and return the (threshold, f1,
    precision, recall) that maximizes F1. Falls back to the 95th percentile
    of normal-window scores if the curve yields no usable threshold (e.g. a
    degenerate split with zero positives)."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, scores)
    if len(thresholds) == 0:
        threshold = float(np.percentile(scores[y_true == 0], 95.0))
        return threshold, 0.0, 0.0, 0.0
    f1s = 2 * precisions[:-1] * recalls[:-1] / np.clip(precisions[:-1] + recalls[:-1], 1e-9, None)
    best = int(np.argmax(f1s))
    return float(thresholds[best]), float(f1s[best]), float(precisions[best]), float(recalls[best])


def low_fpr_threshold(y_true: np.ndarray, scores: np.ndarray, fpr_budget: float = 0.05) -> float:
    """The score threshold that flags roughly `fpr_budget` of the *normal*
    (y_true == 0) examples -- a fixed false-alarm budget, independent of how
    many positives happen to be in this split."""
    normal_scores = scores[y_true == 0]
    return float(np.percentile(normal_scores, 100.0 * (1.0 - fpr_budget)))


def metrics_at_threshold(
    y_true: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[str, Any]:
    """Precision/recall/F1/false-positive-rate/confusion matrix for one
    threshold, so different operating points can be reported side by side."""
    pred = (scores > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "threshold": threshold,
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "false_positive_rate": float(fp / max(fp + tn, 1)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
