# IoT Anomaly Detection Service

A small REST API that scores a window of machine sensor readings and flags
whether it looks abnormal — an early signal of possible failure. Trained on
the AI4I 2020 Predictive Maintenance dataset (UCI, synthetic industrial
telemetry, 10,000 rows).

```text
POST /predict  ->  {"anomaly_score": 0.52, "is_anomaly": false}
GET  /health   ->  {"status": "ok", "model_loaded": true}
```

## Contents

1. [Requirements checklist](#0-requirements-checklist)
2. [Model choice & why](#1-model-choice--why)
3. [Model performance & algorithm comparison](#2-model-performance--algorithm-comparison)
4. [Preprocessing](#3-preprocessing)
5. [API](#4-api)
6. [Build & run](#5-build--run)
7. [Code quality: types, lint, tests](#6-code-quality-types-lint-tests)
8. [Complexity notes](#7-complexity-notes)
9. [Trade-offs & what I'd change next](#8-trade-offs--what-id-change-next)
10. [Repo layout](#9-repo-layout)

## 0. Requirements checklist

Mapped directly against the task brief, for reviewer convenience:

| Requirement | Where |
| --- | --- |
| Model fit on "normal" data, scores new windows, no GPU | `train/train_model.py` — `IsolationForest` fit only on failure-free windows, CPU only (§1) |
| `POST /predict` → `{"anomaly_score": ..., "is_anomaly": ...}` | `app/main.py`, `app/schemas.py` (§4) |
| `GET /health` | `app/main.py` (§4) |
| Dockerfile, build+run in two commands | `Dockerfile` (§5) |
| Model loaded once at startup, not per request | `app/main.py` `lifespan` handler loads `AnomalyModel` once into `model_state`; every request reads the same instance (§5) |
| README: model choice & why | §1, §2 |
| README: preprocessing — windowing, normalization, missing/irregular readings | §3 |
| README: exact build/run commands | §5 |
| README: example curl request + response | §5 |
| README: trade-offs & next steps, incl. Azure | §8 |
| Sliding windows over tabular rows, not row-wise classification | `app/features.py::build_sliding_windows`, used by both training and (implicitly, per-request) serving (§3) |

## 1. Model choice & why

**IsolationForest (scikit-learn), fit only on windows of "normal" data.**

- The brief asks for an anomaly detector, not a failure classifier: "fit on
  normal data, score new windows." IsolationForest is built for exactly
  that — unsupervised, no labels needed at fit time, isolates points that
  are easy to separate from the rest of the data with few random splits.
- It's cheap: trains on 4,513 windows in a fraction of a second, no GPU, no
  heavy dependency, and the resulting artifact is a small joblib file, not a
  multi-hundred-MB checkpoint.
- It's scale-invariant (tree splits on raw feature values), so the shipped
  artifact doesn't need to carry and version a fitted `StandardScaler`
  alongside the model — one less stateful object that train/serve could
  silently drift apart on.

This wasn't just an assumption going in — §2 below backs it with an actual
four-way benchmark (IsolationForest vs XGBoost vs One-Class SVM vs Local
Outlier Factor) on identical data, including a result that contradicts what
I expected: One-Class SVM measurably edges IsolationForest out on raw
metrics here. Read §2 for the honest comparison and why IsolationForest is
still what ships.

**Why not X (short version — see §2 for the measured version):**

- *Autoencoder (reconstruction error)* — not benchmarked here (see §8); more
  moving parts (architecture, training loop) for a ~10k-row dataset where a
  tree ensemble already gets a reasonable signal.
- *Supervised classifier on `Machine failure`* — explicitly not what the
  brief asked for (it wants anomaly scoring fit to normal data, not
  row-wise supervised classification), and it wouldn't generalize to a real
  deployment where you don't have failure labels for new machines yet. §2
  benchmarks this anyway (as XGBoost) to show what a labeled upper bound
  looks like, and to make the "why not" concrete instead of hand-wavy.

## 2. Model performance & algorithm comparison

### Methodology

`train/train_model.py` and `experiments/compare_models.py` share one
train/val/test split (`train/data_prep.py`), built so every number below is
honest, not cherry-picked:

- **Pure-normal windows** (7,521 of them — zero failure rows) are split
  60/20/20 into **train / val / test**. Only `train` (4,513 windows) is ever
  used to fit an unsupervised model.
- **Held-out failure-containing windows** (2,470 of them — never used for
  fitting any unsupervised model) are split 50/50 into **val / test**.
- **val** is used only for hyperparameter selection (ranked by PR-AUC) and
  threshold selection. **test** is touched exactly once, after every other
  decision is locked in, to report final numbers — so these are an honest
  estimate of generalization, not a threshold chosen to look good.
- IsolationForest itself is a small grid search over `n_estimators` /
  `max_samples` / `max_features` (the parameters that actually change what
  the isolation trees look like — `contamination` only shifts an internal
  `offset_` this service doesn't use, since it derives its own threshold
  from validation data instead of `model.predict()`).

### Two threshold operating points — and why the "obviously better" one didn't ship

For every model below, two thresholds are reported:

- **F1-optimal** — sweeps the precision/recall curve on val, picks the
  threshold that maximizes F1.
- **~5%-FPR** — the 95th percentile of *normal*-window scores on val (fixed
  false-positive budget, independent of how many positives happen to be in
  a split).

IsolationForest's **F1-optimal** threshold looked great in isolation
(precision 0.603, recall 0.807, F1 0.690) — until I checked what it actually
does: it flags **43.7%** of normal windows as anomalous. That number is an
artifact of how "anomaly window" is defined here: with `stride=1`, one
failure row contaminates 10 overlapping windows, so ~25% of *all* windows in
this dataset are "anomalous" — nothing like a real deployment, where the
overwhelming majority of traffic is normal. A threshold tuned for F1 on that
near-balanced evaluation set chases recall at the cost of crying wolf on
almost half of normal operation — enough that it flagged this repo's own
"normal" example (`sample_request.json`) as anomalous when I first shipped
it. **The ~5%-FPR threshold is what's actually deployed**, because it
preserves the actual design goal (quiet on normal data, flag real
deviations). Both operating points are reported below so the trade-off is
visible, not hidden behind one convenient number.

### Headline result: IsolationForest (deployed)

| Split | ROC-AUC | PR-AUC |
| --- | --- | --- |
| test (1,504 normal + 1,235 anomalous, never touched until this number) | 0.769 | 0.740 |

| Threshold | Precision | Recall | F1 | False-positive rate | Confusion matrix (TN/FP/FN/TP) |
| --- | --- | --- | --- | --- | --- |
| **~5%-FPR (deployed, τ=0.714)** | 0.846 | 0.325 | 0.469 | 0.049 | 1431 / 73 / 834 / 401 |
| F1-optimal (reference only, τ=0.268) | 0.603 | 0.807 | 0.690 | 0.437 | 847 / 657 / 238 / 997 |

For comparison, the pre-tuning baseline (aggregate-only features: mean/std/
min/max/trend, no hyperparameter search, single in-sample 95th-percentile
threshold) caught ~29% of failure-containing windows at a ~5% false-positive
rate. The deployed model above catches **32.5%** at essentially the same FPR
— a real if modest gain, mostly from the `max_dev` spike feature (§3) rather
than the hyperparameter search, which mattered more for PR-AUC than for this
specific operating point.

### Full comparison: IsolationForest vs XGBoost vs One-Class SVM vs Local Outlier Factor

Run it yourself: `pip install -r experiments/requirements.txt && python
experiments/compare_models.py`. All four models are fit/scored on the exact
same split above; full output (including the notes column) is saved to
[`experiments/results.md`](experiments/results.md).

| Model | ROC-AUC | PR-AUC | Test @ F1-optimal (P / R / F1 / FPR) | Test @ ~5%-FPR (P / R / F1 / FPR) | Fit time | Satisfies brief? |
| --- | --- | --- | --- | --- | --- | --- |
| **IsolationForest (deployed)** | 0.769 | 0.740 | .603 / .807 / .690 / .437 | .846 / .325 / .469 / .049 | 0.30s | yes |
| One-Class SVM | 0.763 | **0.758** | .582 / .810 / .678 / .477 | .870 / .370 / **.519** / .045 | **0.05s** | yes |
| Local Outlier Factor | 0.724 | 0.731 | .521 / .806 / .633 / .609 | .866 / .340 / .488 / .043 | 1.54s | yes |
| XGBoost (supervised) | **0.987** | **0.985** | **.977 / .883 / .928** / .017 | .950 / .921 / .935 / .040 | 1.87s | **no** |

**XGBoost dominates every metric — and is disqualified anyway.** It was
trained directly on failure labels (see `experiments/compare_models.py` for
exactly which windows), which is precisely what the brief rules out ("fit on
normal data, score new windows," not row-wise supervised classification) and
what a real deployment can't do reliably for a new machine with no failure
history yet. It's included as an honest upper bound — "how good could this
be if we cheated" — not a candidate.

**One-Class SVM measurably edges out IsolationForest**, and it isn't noise:
refitting both across 5 independent random splits confirms it —

| Model | PR-AUC (mean ± std, 5 splits) | Deployed-threshold F1 (mean ± std) |
| --- | --- | --- |
| IsolationForest | 0.747 ± 0.004 | 0.487 ± 0.017 |
| One-Class SVM | 0.761 ± 0.007 | 0.515 ± 0.004 |

OCSVM is also *faster* to fit at this data scale (~4.5k rows) — its
`O(n²)`-ish training cost only starts to bite at scales this project doesn't
reach. My original assumption (still visible in git history) was that OCSVM
would be "more sensitive to feature scaling and kernel choice, slower... not
obviously better." Measured against real data, that was wrong about speed
and wrong about "not obviously better" — a good reminder to check
assumptions like this rather than ship them.

**So why does IsolationForest still ship, not OCSVM?** Honestly: the margin
is real but modest (~1.4pp PR-AUC, ~3pp F1), and I weighed it against two
non-metric factors — (1) OCSVM requires bundling a fitted `StandardScaler`
into the artifact and applying it identically at serving time, a second
stateful object that has to stay version-locked with the model (the kind of
train/serve skew risk `app/features.py` was explicitly written to avoid
elsewhere in this repo); (2) this comparison is a single stratified split
design validated across 5 seeds, not full k-fold cross-validation or a
sweep over OCSVM's `nu`/`gamma`, so I'm not confident the gap is fully
optimized on either side. **If I had another hour, switching the default to
One-Class SVM would be the first thing I'd validate properly** — the
evidence for it is real, not hand-waved. See §8.

## 3. Preprocessing

**The core problem:** AI4I 2020 is tabular — one row per machine snapshot,
no timestamp, no "this row follows that row on the same machine" structure.
Per the task note, it's treated as a proxy sensor stream by sorting on
`UDI` (the dataset's row order, 1..10000) and sliding a window over it. This
is a documented modeling assumption, not a claim that these rows are a real
time series — see the caveat in §8.

**Windowing:** `window_size=10`, `stride=1` (`train/data_prep.py`) → 9,991
overlapping windows (`app/features.py::build_sliding_windows`, a linear scan
over the row sequence — see §7). Windows made entirely of non-failure rows
are used to fit the model (7,521 of them, further split per §2); windows
that contain at least one failure row are held out and used only for
validation/test evaluation — never for fitting.

**Feature extraction (`app/features.py`, shared by training and serving):**
for each of the 5 sensor channels (air temperature, process temperature,
rotational speed, torque, tool wear), compute `mean`, `std`, `min`, `max`,
`trend` (last − first), and `max_dev` over the window, plus one
`window_length` feature. **31 features total.** `max_dev` — max
`|reading − window median|` — was added specifically because the other
aggregate stats dilute a single bad reading: a window is 10 rows, and if
only 1 is the actual failure, the other 9 normal rows pull the mean/std back
toward "normal" while a robust median barely moves, so `max_dev` stays
sensitive to exactly that spike (`tests/test_features.py` has a unit test
that demonstrates this directly). Using fixed aggregate stats instead of raw
sequences also means the same code handles a 10-reading training window and
a 3-reading or 15-reading window from a client with no shape mismatch.

**Normalization:** IsolationForest splits on raw feature values (it's
tree-based, not distance-based), so no `StandardScaler` is needed for the
model itself (see §2 for why that matters when comparing against
kernel-/distance-based alternatives). What *is* normalized is the **output
score**: raw `-decision_function()` values are min-max scaled using the
0.5th/99.5th percentile of the model's own training scores, then clipped to
`[0, 1]`, so the API always returns a score you can reason about instead of
an arbitrary IsolationForest number that could be anywhere from -0.5 to 0.5.

**Missing / irregular readings:** any reading can omit any channel (all
sensor fields in `SensorReading` are `Optional`, see `app/schemas.py`).
Missing values are imputed with the training-set per-channel median
(`model_artifact/anomaly_model.joblib` stores these) before feature
extraction. Windows of any length ≥ 1 are accepted — `std`/`trend`/`max_dev`
degrade to `0` for a single reading rather than `NaN`. This is a deliberate
trade-off: silently imputing keeps the endpoint robust to real-world sensor
dropouts, at the cost of not being explicit per-request about which fields
were imputed (see §8 for the fix).

**Train/serve parity:** `app/features.py` is the *only* place feature
vectors are built, imported by both `train/data_prep.py` and `app/model.py`.
Training and inference can never silently drift apart in how a window
becomes a feature vector — the single most common way anomaly-detection
services quietly break in production.

## 4. API

### `POST /predict`

```json
{
  "readings": [
    {"air_temperature": 298.1, "process_temperature": 308.6, "rotational_speed": 1551, "torque": 42.8, "tool_wear": 0, "type": "M"},
    { "...": "9 more readings, oldest to newest" }
  ]
}
```

Returns:

```json
{"anomaly_score": 0.52, "is_anomaly": false}
```

`type` (product quality variant L/M/H) is accepted but not currently used by
the model — see §8. Every sensor field is optional per-reading (a dropped
sensor value is imputed, not rejected); an empty `readings` list is
rejected with `422`.

### `GET /health`

```json
{"status": "ok", "model_loaded": true}
```

## 5. Build & run

```bash
docker build -t iot-anomaly-service .
docker run -p 8000:8000 iot-anomaly-service
```

**Normal window** — a real 10-row window from the training data, included
as `sample_request.json`:

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

Response:

```json
{"anomaly_score":0.6589200828103985,"is_anomaly":false}
```

**Anomalous window** — a real 10-row window that contains a failed
machine row, included as `sample_anomaly_request.json`, scores noticeably
higher and crosses the threshold:

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @sample_anomaly_request.json
```

Response:

```json
{"anomaly_score":0.908588068082555,"is_anomaly":true}
```

**Health check:**

```bash
curl -s http://localhost:8000/health
```

```json
{"status":"ok","model_loaded":true}
```

**Local dev without Docker:**

```bash
pip install -r requirements.txt
python train/train_model.py      # regenerates model_artifact/ from data/ai4i2020.csv
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 6. Code quality: types, lint, tests

Dev tooling is kept out of the runtime image on purpose — install it
separately:

```bash
pip install -r requirements-dev.txt   # pytest, httpx, mypy, ruff, pandas-stubs
```

```bash
pytest -q                          # tests/ — API smoke tests + feature-engineering unit tests
mypy app train tests               # strict mode (see [tool.mypy] in pyproject.toml)
ruff check .                       # lint
ruff format --check .              # formatting
```

All four are clean on this repo (10 tests passing, 0 mypy errors in strict
mode across `app/`, `train/`, and `tests/`, 0 ruff findings). `pyproject.toml`
holds config for both tools; the only type-checking gaps are
`ignore_missing_imports` overrides for `joblib`, `sklearn`, and `xgboost`,
none of which ship type stubs — not a gap in this repo's own typing.

`tests/test_api.py` covers the API surface (health, a normal window, a real
failure-containing window that must cross the threshold, missing fields, a
rejected empty window). `tests/test_features.py` covers the shared
feature-engineering module directly: the prefix-sum windowing helper against
a naive reference implementation, stride handling, the missing-value/
short-window edge cases in `extract_features`, and a targeted test proving
`max_dev` catches a single-row spike that `std` dilutes.

`experiments/` (see §2) has its own `requirements.txt` — `xgboost` is a
large native-code dependency that routine dev work (API, tests, lint,
type-checks) never needs, so it's kept out of `requirements-dev.txt`.

## 7. Complexity notes

Windowing and feature extraction are linear in the data, not accidentally
quadratic:

- `build_sliding_windows`: **O(n)** — one pass over the row sequence,
  `n_windows = (n_rows - window_size) / stride + 1`.
- `extract_features`: **O(w)** per window (w = window size) — a window must
  be read at least once to compute any aggregate over it, so this is
  already optimal; total cost across training is O(n · w).
- Classifying each window as "pure normal" vs "contains a failure"
  (`train/data_prep.py::build_windows`): originally two full `all(...)` /
  `any(...)` scans per window — O(n · w) with a constant factor of 2.
  Replaced with `sliding_window_sums` (`app/features.py`), a **prefix-sum**
  over the 0/1 failure flags: O(n) to build the prefix array, then O(1) per
  window to read off its failure count via `prefix[end] - prefix[start]`.
  Verified behavior-preserving — retraining after the change reproduces the
  exact same 7,521 normal / 2,470 held-out window split as before.
- Inference: one `IsolationForest.decision_function` call per request over
  300 trees, each O(log(max_samples)) deep — effectively O(1) relative to
  request size, dominated by the O(w) feature extraction for whatever
  window the client sent.

At n=10,000 rows none of this is a measured bottleneck (training runs in
well under a second either way) — the prefix-sum change is included because
it's the correct tool for a "sum over every sliding window" query and costs
nothing extra to write correctly, not because O(n·w) was ever slow here.

## 8. Trade-offs & what I'd change next

**Honesty about model quality first:** at the deployed ~5%-false-positive
operating point, the model catches 32.5% of failure-containing windows
(§2) — a real limitation, not a hidden one. A window is 10 rows, and if only
1 is the actual failure row, the other 9 normal rows still shape most of the
aggregate features; `max_dev` (§3) helps but doesn't fully solve this.
Reporting the honest number instead of picking a threshold that looks good
in isolation (§2's F1-optimal-threshold trap) is deliberate.

**What I'd do next, roughly in priority order:**

1. **Validate switching the default to One-Class SVM.** §2's comparison
   shows it beating IsolationForest on PR-AUC and deployed-threshold F1,
   consistently across 5 random splits — not noise. Before switching:
   k-fold cross-validate properly (this repo only checked 5 single splits),
   sweep `nu`/`gamma` the way `HYPERPARAM_GRID` sweeps IsolationForest's
   params, and bundle a versioned `StandardScaler` into the artifact.
2. **Feature engineering over aggregation:** `max_dev` (§3) was the first
   spike-sensitive feature; a per-channel "rate of change of max_dev" or a
   rolling z-score against a longer machine history could sharpen this
   further without changing the modeling approach.
3. **Weight by product `type`:** the dataset's own failure logic
   (overstrain threshold) depends on `type` (L/M/H); a per-type baseline —
   or even three separate small models — would likely sharpen detection
   instead of pooling all types together.
4. **True time-series windowing:** the biggest caveat in this whole
   exercise — AI4I rows aren't really sequential per-machine. Given a real
   device stream, I'd window per physical machine ID and by actual
   timestamp, not by CSV row order.
5. **Autoencoder / deep methods:** not benchmarked here (§2 only compares
   classical methods). With more data — or a per-machine deployment that
   accumulates history over weeks — a small reconstruction-error
   autoencoder or Deep SVDD is worth adding to the comparison; a library
   survey via [PyOD](https://pyod.readthedocs.io/) would be the fast way to
   try several at once.
6. **Proper k-fold hyperparameter search:** `HYPERPARAM_GRID` in
   `train/train_model.py` is 4 combinations scored on one val split;
   k-fold CV over a wider grid (and the same for OCSVM's `nu`/`gamma`)
   would give more confidence the chosen config isn't a lucky split.
7. **Confidence/explainability in the response:** return which channel(s)
   contributed most to the score (e.g. per-channel feature importance from
   the forest) so a technician gets an actionable "torque looks off" rather
   than a bare number.
8. **Explicit imputation flag:** currently missing fields are silently
   imputed; I'd add an `imputed_fields` list to the response so a caller
   knows a reading was incomplete rather than trusting a score computed on
   partially fabricated data.
9. **Cost-sensitive threshold:** the ~5%-FPR threshold is a reasonable
   default, but the right operating point genuinely depends on the ratio of
   "cost of a missed failure" to "cost of a false alarm" for a specific
   deployment — worth exposing as a configurable knob rather than baking in
   one number.

**For real device streams on Azure**, specifically:

- Swap the CSV-driven training script for a scheduled retrain job (Azure ML
  pipeline or a simple Azure Function on a timer) reading from wherever the
  device telemetry actually lands (Azure IoT Hub → Event Hub / ADLS), rather
  than a one-off local CSV.
- Put the service behind Azure Container Apps or AKS instead of `docker
  run` directly; use a readiness probe against `/health` so the model is
  fully loaded (not just "container started") before traffic is routed to
  it — the lifespan-loaded model already makes this a non-issue for cold
  requests, but the orchestrator still needs the health check.
- Real streams need actual per-device windowing with timestamps and
  gap-handling (a device that drops offline for 20 minutes shouldn't have
  its next reading treated as a `trend` of 20 minutes of drift) — this repo
  windows over CSV row order as a stand-in for that.
- Model artifact versioning: store artifacts in Azure Blob Storage / Azure
  ML Model Registry with a version tag, and have the API pull the latest
  approved version at startup instead of a `COPY` baked into the image, so
  retraining doesn't require a rebuild+redeploy of the whole service.
- Add basic auth / API key (e.g. Azure API Management in front) since this
  version has none — deliberately out of scope for the "not a production
  system" brief, but the first thing to add before it touched a real
  device fleet.

## 9. Repo layout

```text
app/
  main.py                     FastAPI app, /predict and /health, model loaded once at startup (lifespan)
  model.py                    AnomalyModel wrapper: loads the joblib artifact, scores one window
  features.py                 Shared feature engineering (used by training, serving, AND experiments -- no train/serve skew)
  schemas.py                  Pydantic request/response models
train/
  data_prep.py                Shared data loading / windowing / leakage-free train-val-test split
  evaluation.py                Shared threshold selection (F1-optimal, fixed-FPR) + precision/recall/F1 metrics
  train_model.py               Production pipeline: hyperparameter search, threshold selection, honest test-set evaluation, joblib artifact
experiments/
  compare_models.py            IsolationForest vs XGBoost vs One-Class SVM vs Local Outlier Factor on the identical split (see README §2)
  results.md                   Generated output of the above (comparison table + 5-seed robustness check)
  requirements.txt             xgboost, kept separate from requirements-dev.txt (large native dependency, experiments-only)
model_artifact/
  anomaly_model.joblib          Trained model + scaling params + threshold (loaded by the API)
  metadata.json                 Human-readable summary, incl. held-out precision/recall/F1/ROC-AUC/PR-AUC
data/
  ai4i2020.csv                  AI4I 2020 dataset (CC BY 4.0, UCI ML Repository / S. Matzka)
tests/
  test_api.py                   API smoke tests (health, normal window, real anomalous window, missing fields, rejected empty window)
  test_features.py              Unit tests for windowing / prefix-sum / feature extraction / the max_dev spike feature
sample_request.json             A real 10-row "normal" window from the dataset, used in the curl example above
sample_anomaly_request.json     A real 10-row window containing a failure row, used in the curl example above
pyproject.toml                  ruff + mypy + pytest configuration
requirements.txt                Runtime dependencies (what ships in the Docker image)
requirements-dev.txt            + pytest/httpx/mypy/ruff/pandas-stubs, for local development
Dockerfile
```
