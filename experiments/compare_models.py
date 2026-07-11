"""
Compare IsolationForest (the deployed model) against three alternative
anomaly-detection approaches on the IDENTICAL data split (train/data_prep.py)
and the IDENTICAL metrics (train/evaluation.py) used by train/train_model.py,
so the comparison in README §1a is apples-to-apples, not four different
scripts with four different setups.

Models compared:
  - IsolationForest      unsupervised, tree-based (production model)
  - One-Class SVM        unsupervised, kernel-based, on standardized features
  - Local Outlier Factor  unsupervised, density-based, on standardized features
  - XGBoost               SUPERVISED -- trained directly on failure labels.
      Included because it's a natural "how good could we do if we cheated
      and used the labels" reference point, not because it's deployable:
      the task brief requires fitting on normal data only (no failure labels
      at fit time), which XGBoost-as-classifier violates by construction, and
      a real deployment won't have reliable failure labels for new machines
      to retrain a classifier against. See the "why not deploy it" note in
      the printed/saved comparison table.

Usage:
    python experiments/compare_models.py

Writes: experiments/results.md   (the table reproduced in README §1a)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from train.data_prep import RANDOM_STATE, Splits, stack_features  # noqa: E402
from train.evaluation import (  # noqa: E402
    best_threshold_by_f1,
    low_fpr_threshold,
    metrics_at_threshold,
)

RESULTS_PATH = ROOT / "experiments" / "results.md"

# Same hyperparameters train/train_model.py's grid search selected, so this
# comparison uses the exact model that ships, not a re-tuned stand-in.
ISOLATION_FOREST_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "max_samples": 512,
    "max_features": 0.75,
}


def _unit_score(raw: np.ndarray, ref_raw: np.ndarray) -> np.ndarray:
    """Min-max scale `raw` using the 0.5th/99.5th percentile of `ref_raw`
    (always the training-fold scores) -- same calibration approach as
    app/model.py, applied identically for every algorithm here."""
    lo = np.percentile(ref_raw, 0.5)
    hi = np.percentile(ref_raw, 99.5)
    return np.asarray(np.clip((raw - lo) / max(hi - lo, 1e-9), 0.0, 1.0))


def evaluate(
    name: str, val_scores: np.ndarray, test_scores: np.ndarray, splits: Splits
) -> dict[str, Any]:
    """Threshold selection on val, honest metrics on test -- identical
    methodology to train/train_model.py, applied to every model here."""
    f1_thr, val_f1, val_p, val_r = best_threshold_by_f1(splits.y_val, val_scores)
    fpr_thr = low_fpr_threshold(splits.y_val, val_scores, fpr_budget=0.05)
    return {
        "name": name,
        "roc_auc": float(roc_auc_score(splits.y_test, test_scores)),
        "pr_auc": float(average_precision_score(splits.y_test, test_scores)),
        "val_f1_optimal": {"threshold": f1_thr, "f1": val_f1, "precision": val_p, "recall": val_r},
        "test_at_f1_optimal": metrics_at_threshold(splits.y_test, test_scores, f1_thr),
        "test_at_low_fpr": metrics_at_threshold(splits.y_test, test_scores, fpr_thr),
    }


def run_isolation_forest(splits: Splits) -> dict[str, Any]:
    start = time.perf_counter()
    model = IsolationForest(random_state=RANDOM_STATE, n_jobs=-1, **ISOLATION_FOREST_PARAMS)
    model.fit(splits.X_train)
    fit_seconds = time.perf_counter() - start

    train_raw = -model.decision_function(splits.X_train)
    val_scores = _unit_score(-model.decision_function(splits.X_val), train_raw)
    test_scores = _unit_score(-model.decision_function(splits.X_test), train_raw)

    result = evaluate("IsolationForest", val_scores, test_scores, splits)
    result["fit_seconds"] = fit_seconds
    result["supervised"] = False
    result["satisfies_brief"] = True
    result["notes"] = "Tree-based; no scaling needed; deployed model."
    return result


def run_one_class_svm(splits: Splits) -> dict[str, Any]:
    scaler = StandardScaler().fit(splits.X_train)
    X_train = scaler.transform(splits.X_train)
    X_val = scaler.transform(splits.X_val)
    X_test = scaler.transform(splits.X_test)

    start = time.perf_counter()
    # nu ~= expected fraction of outliers in the training set; there are none
    # by construction (train is normal-only), so this is just the standard
    # small-nu default for a one-class boundary that isn't too tight.
    model = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale")
    model.fit(X_train)
    fit_seconds = time.perf_counter() - start

    train_raw = -model.decision_function(X_train)
    val_scores = _unit_score(-model.decision_function(X_val), train_raw)
    test_scores = _unit_score(-model.decision_function(X_test), train_raw)

    result = evaluate("One-Class SVM", val_scores, test_scores, splits)
    result["fit_seconds"] = fit_seconds
    result["supervised"] = False
    result["satisfies_brief"] = True
    result["notes"] = (
        "Kernel-based; needs StandardScaler bundled into the artifact. Fit time is O(n^2)-ish "
        "in theory, but at this data scale (~4.5k train rows) it's actually the FASTEST fit "
        "here, not the slowest -- see fit time column."
    )
    return result


def run_local_outlier_factor(splits: Splits) -> dict[str, Any]:
    scaler = StandardScaler().fit(splits.X_train)
    X_train = scaler.transform(splits.X_train)
    X_val = scaler.transform(splits.X_val)
    X_test = scaler.transform(splits.X_test)

    start = time.perf_counter()
    model = LocalOutlierFactor(n_neighbors=20, novelty=True)
    model.fit(X_train)
    fit_seconds = time.perf_counter() - start

    train_raw = -model.decision_function(X_train)
    val_scores = _unit_score(-model.decision_function(X_val), train_raw)
    test_scores = _unit_score(-model.decision_function(X_test), train_raw)

    result = evaluate("Local Outlier Factor", val_scores, test_scores, splits)
    result["fit_seconds"] = fit_seconds
    result["supervised"] = False
    result["satisfies_brief"] = True
    result["notes"] = (
        "Density-based; needs StandardScaler; scores are LOCAL to each window's neighbors."
    )
    return result


def run_xgboost(splits: Splits) -> dict[str, Any]:
    # SUPERVISED reference point only -- see module docstring. Trained on
    # train-normal (label 0) + val-anomaly (label 1); val-normal is held back
    # so threshold selection below still has an unseen-by-fitting negative
    # class, though val-anomaly's label was seen at fit time (unlike every
    # other model here). Test is untouched by fitting either way.
    X_fit = np.concatenate([splits.X_train, stack_features(splits.val_a, splits.medians)])
    y_fit = np.concatenate([np.zeros(len(splits.X_train)), np.ones(len(splits.val_a))])

    start = time.perf_counter()
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_fit, y_fit)
    fit_seconds = time.perf_counter() - start

    val_scores = model.predict_proba(splits.X_val)[:, 1]
    test_scores = model.predict_proba(splits.X_test)[:, 1]

    result = evaluate("XGBoost (supervised)", val_scores, test_scores, splits)
    result["fit_seconds"] = fit_seconds
    result["supervised"] = True
    result["satisfies_brief"] = False
    result["notes"] = (
        "Trained directly on failure labels (val-anomaly) -- violates the brief's "
        "'fit on normal data only' requirement and needs labeled failures for new "
        "machines that a real deployment won't reliably have. Included as an upper-"
        "bound reference, not a deployment candidate."
    )
    return result


def _fmt_metrics(m: dict[str, Any]) -> str:
    return (
        f"{m['precision']:.3f} / {m['recall']:.3f} / {m['f1']:.3f} / {m['false_positive_rate']:.3f}"
    )


def _fmt_row(r: dict[str, Any]) -> str:
    brief = "yes" if r["satisfies_brief"] else "no (supervised)"
    return (
        f"| {r['name']} | {r['roc_auc']:.3f} | {r['pr_auc']:.3f} | "
        f"{_fmt_metrics(r['test_at_f1_optimal'])} | {_fmt_metrics(r['test_at_low_fpr'])} | "
        f"{r['fit_seconds']:.2f}s | {brief} |"
    )


def robustness_check(seeds: list[int]) -> str:
    """IsolationForest and One-Class SVM came out closest on the seed=42
    split (see README §1a) -- close enough that a single random split isn't
    enough to call a winner. Refit both across several independent
    train/val/test splits and report mean +/- std of PR-AUC and the deployed
    (~5%-FPR) F1, so the comparison reflects split-to-split variance instead
    of one lucky/unlucky draw."""
    pr_aucs: dict[str, list[float]] = {"IsolationForest": [], "One-Class SVM": []}
    f1s: dict[str, list[float]] = {"IsolationForest": [], "One-Class SVM": []}
    for seed in seeds:
        splits = Splits(random_state=seed)
        for result in (run_isolation_forest(splits), run_one_class_svm(splits)):
            pr_aucs[result["name"]].append(result["pr_auc"])
            f1s[result["name"]].append(result["test_at_low_fpr"]["f1"])

    lines = [
        "| Model | PR-AUC (mean +/- std) | Deployed-threshold F1 (mean +/- std) |",
        "| --- | --- | --- |",
    ]
    for name in pr_aucs:
        pr = np.array(pr_aucs[name])
        f1 = np.array(f1s[name])
        lines.append(
            f"| {name} | {pr.mean():.3f} +/- {pr.std():.3f} | {f1.mean():.3f} +/- {f1.std():.3f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    splits = Splits()
    val_n, val_a = len(splits.val_n), len(splits.val_a)
    test_n, test_a = len(splits.test_n), len(splits.test_a)
    print(
        f"Loaded splits: {len(splits.train_n)} train-normal | "
        f"val {val_n}n/{val_a}a | test {test_n}n/{test_a}a"
    )

    results = [
        run_isolation_forest(splits),
        run_one_class_svm(splits),
        run_local_outlier_factor(splits),
        run_xgboost(splits),
    ]

    header = (
        "| Model | ROC-AUC | PR-AUC | TEST @ F1-optimal (P/R/F1/FPR) | "
        "TEST @ ~5%-FPR (P/R/F1/FPR) | Fit time | Satisfies brief? |\n"
        "| --- | --- | --- | --- | --- | --- | --- |"
    )
    rows = "\n".join(_fmt_row(r) for r in results)
    table = f"{header}\n{rows}\n"
    print()
    print(table)

    print("Robustness check: IsolationForest vs One-Class SVM across 5 random splits...")
    robustness_table = robustness_check(seeds=[1, 2, 3, 4, 5])
    print(robustness_table)

    RESULTS_PATH.write_text(
        "# Model comparison (generated by experiments/compare_models.py)\n\n"
        "All four models are fit/scored on the identical train/val/test split from "
        "`train/data_prep.py`. P/R/F1/FPR = precision/recall/F1/false-positive-rate on "
        "the held-out test split, never touched during fitting or threshold selection "
        "(except XGBoost's own labels -- see notes).\n\n"
        + table
        + "\n## Notes\n\n"
        + "\n".join(f"- **{r['name']}**: {r['notes']}" for r in results)
        + "\n\n## Robustness check (IsolationForest vs One-Class SVM, 5 random splits)\n\n"
        + "IsolationForest and One-Class SVM came out closest above -- refit both on 5 "
        "independent seeds to see if that gap is a real effect or one split's noise.\n\n"
        + robustness_table
    )
    print(f"Saved -> {RESULTS_PATH}")


if __name__ == "__main__":
    main()
