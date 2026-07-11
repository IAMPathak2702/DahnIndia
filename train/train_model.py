"""
Train the anomaly detection model on the AI4I 2020 Predictive Maintenance
dataset and save a single joblib artifact that the API loads at startup.

Usage:
    python train/train_model.py

Reads:  data/ai4i2020.csv
Writes: model_artifact/anomaly_model.joblib
        model_artifact/metadata.json   (human-readable summary)

See experiments/compare_models.py for how IsolationForest was chosen over
XGBoost / One-Class SVM / Local Outlier Factor on the exact same data split,
and README §1/§2/§6 for the full writeup of the methodology below.

Evaluation methodology:
IsolationForest has no gradient-descent loss to "optimize" -- it's built from
random splits, not minimized error. The two real levers for detection quality
are (a) which features it sees and (b) which of its hyperparameters control
how the isolation trees are built (n_estimators / max_samples / max_features;
`contamination` only shifts an internal offset_ this script doesn't use, so
it's not worth tuning). This script does both:
  1. Splits the "pure normal" windows 60/20/20 into train/val/test, and the
     held-out failure-containing windows 50/50 into val/test (train/data_prep.py).
     Windows only ever appear in exactly one split -- no leakage between
     fitting, hyperparameter selection, threshold selection, and reporting.
  2. Grid-searches a few (n_estimators, max_samples, max_features)
     combinations, fit on train, scored by PR-AUC on val -- PR-AUC (not
     accuracy) because failure-containing windows are the rare class in a
     real deployment (see the threshold-choice comment below for why they're
     NOT rare in this particular evaluation split).
  3. Picks two threshold operating points on val: F1-optimal, and a fixed
     ~5% false-positive budget. Ships the latter -- see the comment at
     `low_fpr_threshold` below for why.
  4. Reports precision / recall / F1 / ROC-AUC / PR-AUC / confusion matrix at
     both operating points on test -- a split untouched by steps 2-3, so
     these numbers are an honest estimate of generalization.
  5. Refits the chosen hyperparameters on ALL normal windows (train+val+test)
     for the artifact that actually ships, since more training data is free
     once the design decisions above are already validated on held-out data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.features import SENSOR_CHANNELS  # noqa: E402
from train.data_prep import RANDOM_STATE, STRIDE, WINDOW_SIZE, Splits, stack_features  # noqa: E402
from train.evaluation import (  # noqa: E402
    best_threshold_by_f1,
    low_fpr_threshold,
    metrics_at_threshold,
)

ARTIFACT_DIR = ROOT / "model_artifact"
ARTIFACT_PATH = ARTIFACT_DIR / "anomaly_model.joblib"
METADATA_PATH = ARTIFACT_DIR / "metadata.json"

# n_estimators/max_samples/max_features actually change what the isolation
# trees look like (unlike `contamination`, which only shifts an offset_ this
# service doesn't use -- see module docstring). Kept small and CPU-cheap.
HYPERPARAM_GRID: list[dict[str, Any]] = [
    {"n_estimators": 200, "max_samples": "auto", "max_features": 1.0},
    {"n_estimators": 200, "max_samples": 256, "max_features": 0.75},
    {"n_estimators": 300, "max_samples": "auto", "max_features": 1.0},
    {"n_estimators": 300, "max_samples": 512, "max_features": 0.75},
]


def main() -> None:
    ARTIFACT_DIR.mkdir(exist_ok=True)
    splits = Splits()
    print(
        f"{splits.n_normal_total} pure-normal windows, "
        f"{splits.n_anomaly_total} windows contain at least one failure row "
        "(never used for fitting -- only for validation/test evaluation below)"
    )
    print(
        f"Split: {len(splits.train_n)} train (normal only) | "
        f"{len(splits.val_n)} val-normal + {len(splits.val_a)} val-anomaly | "
        f"{len(splits.test_n)} test-normal + {len(splits.test_a)} test-anomaly"
    )

    # --- Hyperparameter search: fit on train, rank by PR-AUC on val. -------
    best_params: dict[str, Any] | None = None
    best_pr_auc = -1.0
    best_eval_model: IsolationForest | None = None
    for params in HYPERPARAM_GRID:
        candidate = IsolationForest(random_state=RANDOM_STATE, n_jobs=-1, **params)
        candidate.fit(splits.X_train)
        val_scores = -candidate.decision_function(splits.X_val)
        pr_auc = float(average_precision_score(splits.y_val, val_scores))
        print(f"  {params} -> val PR-AUC={pr_auc:.3f}")
        if pr_auc > best_pr_auc:
            best_pr_auc, best_params, best_eval_model = pr_auc, params, candidate

    assert best_params is not None and best_eval_model is not None
    print(f"Selected hyperparameters {best_params} (val PR-AUC={best_pr_auc:.3f})")

    # --- Score scaling + threshold, calibrated on val only. -----------------
    train_raw_scores = -best_eval_model.decision_function(splits.X_train)
    score_min = float(np.percentile(train_raw_scores, 0.5))
    score_max = float(np.percentile(train_raw_scores, 99.5))

    def to_unit_score(raw: np.ndarray) -> np.ndarray:
        scaled = (raw - score_min) / max(score_max - score_min, 1e-9)
        return np.clip(scaled, 0.0, 1.0)

    val_unit_scores = to_unit_score(-best_eval_model.decision_function(splits.X_val))
    f1_optimal_threshold, val_f1, val_precision, val_recall = best_threshold_by_f1(
        splits.y_val, val_unit_scores
    )
    print(
        f"F1-optimal threshold={f1_optimal_threshold:.3f} from val "
        f"(F1={val_f1:.3f}, precision={val_precision:.3f}, recall={val_recall:.3f})"
    )

    # A second, more conservative operating point: fix the false-positive
    # budget at ~5% of normal windows (like the original single-threshold
    # design) instead of maximizing F1.
    #
    # This -- not the F1-optimal one above -- is what actually ships. Why:
    # "anomaly" windows here are any 10-row window touching >=1 failure row,
    # so with stride=1 a single failure row contaminates 10 overlapping
    # windows. That makes ~25% of ALL windows in this dataset "anomalous"
    # (n_anomaly_total / n_normal_total below) -- nothing like a real
    # deployment, where the overwhelming majority of traffic is normal and
    # true failures are rare. A threshold tuned for F1 on a near-balanced
    # 55/45 val set chases recall at a 44% false-positive rate, which would
    # mean crying wolf on roughly half of all normal operation once deployed
    # -- alert fatigue, and it even flags this repo's own "normal" example
    # (sample_request.json). The ~5%-FPR threshold preserves the actual
    # design goal (quiet on normal data, flag real deviations) at a real, if
    # smaller, recall gain over the original aggregate-only features. Both
    # operating points are reported below so the trade-off is visible, not
    # hidden behind one number.
    fpr_threshold = low_fpr_threshold(splits.y_val, val_unit_scores, fpr_budget=0.05)
    threshold = fpr_threshold  # <- deployed operating point

    # --- Final, honest evaluation on test -- untouched until now. -----------
    test_unit_scores = to_unit_score(-best_eval_model.decision_function(splits.X_test))
    f1_optimal_metrics = metrics_at_threshold(splits.y_test, test_unit_scores, f1_optimal_threshold)
    low_fpr_metrics = metrics_at_threshold(splits.y_test, test_unit_scores, fpr_threshold)
    roc_auc = float(roc_auc_score(splits.y_test, test_unit_scores))
    pr_auc_test = float(average_precision_score(splits.y_test, test_unit_scores))
    print(
        f"TEST @ F1-optimal threshold={f1_optimal_threshold:.3f} -> "
        f"precision={f1_optimal_metrics['precision']:.3f} "
        f"recall={f1_optimal_metrics['recall']:.3f} "
        f"f1={f1_optimal_metrics['f1']:.3f} fpr={f1_optimal_metrics['false_positive_rate']:.3f}"
    )
    print(
        f"TEST @ ~5%-FPR threshold={fpr_threshold:.3f} (DEPLOYED) -> "
        f"precision={low_fpr_metrics['precision']:.3f} recall={low_fpr_metrics['recall']:.3f} "
        f"f1={low_fpr_metrics['f1']:.3f} fpr={low_fpr_metrics['false_positive_rate']:.3f}"
    )
    print(f"TEST ROC-AUC={roc_auc:.3f} PR-AUC={pr_auc_test:.3f} (threshold-independent)")

    # --- Refit on ALL normal windows for the artifact that actually ships. --
    # The design (features, hyperparameters, threshold) was already validated
    # on held-out data above; folding val+test's normal windows back in just
    # gives the shipped model strictly more "normal" data to learn from.
    final_model = IsolationForest(random_state=RANDOM_STATE, n_jobs=-1, **best_params)
    X_all_normal = np.concatenate(
        [
            splits.X_train,
            stack_features(splits.val_n, splits.medians),
            stack_features(splits.test_n, splits.medians),
        ]
    )
    final_model.fit(X_all_normal)
    final_raw_scores = -final_model.decision_function(X_all_normal)
    final_score_min = float(np.percentile(final_raw_scores, 0.5))
    final_score_max = float(np.percentile(final_raw_scores, 99.5))

    artifact = {
        "model": final_model,
        "window_size": WINDOW_SIZE,
        "sensor_channels": SENSOR_CHANNELS,
        "medians": splits.medians,
        "score_min": final_score_min,
        "score_max": final_score_max,
        "threshold": threshold,
    }
    joblib.dump(artifact, ARTIFACT_PATH)
    print(f"Saved model artifact -> {ARTIFACT_PATH}")

    metadata = {
        "model_type": "IsolationForest (scikit-learn)",
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "hyperparameters": best_params,
        "sensor_channels": list(SENSOR_CHANNELS),
        "feature_medians_used_for_missing_values": splits.medians,
        "score_min_raw": final_score_min,
        "score_max_raw": final_score_max,
        "is_anomaly_threshold": threshold,
        "deployed_threshold_strategy": "low_fpr_5pct",
        "n_normal_windows_total": splits.n_normal_total,
        "n_holdout_failure_windows_total": splits.n_anomaly_total,
        "split_sizes": {
            "train_normal": len(splits.train_n),
            "val_normal": len(splits.val_n),
            "val_anomaly": len(splits.val_a),
            "test_normal": len(splits.test_n),
            "test_anomaly": len(splits.test_a),
        },
        "validation_metrics_at_f1_optimal_threshold": {
            "threshold": f1_optimal_threshold,
            "f1": val_f1,
            "precision": val_precision,
            "recall": val_recall,
            "pr_auc": best_pr_auc,
        },
        "held_out_test_metrics": {
            "roc_auc": roc_auc,
            "pr_auc": pr_auc_test,
            "at_f1_optimal_threshold_NOT_DEPLOYED": f1_optimal_metrics,
            "at_low_fpr_threshold_DEPLOYED": low_fpr_metrics,
        },
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    print(f"Saved metadata -> {METADATA_PATH}")


if __name__ == "__main__":
    main()
