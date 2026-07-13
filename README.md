# IoT Anomaly Detection Service

A small REST API that looks at a window of machine sensor readings and
decides whether it looks abnormal — basically an early warning before a
machine actually fails. It's trained on the AI4I 2020 Predictive
Maintenance dataset (UCI, synthetic industrial telemetry, 10,000 rows).

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

The task brief had a specific set of asks, so here's where each one is
actually satisfied in the code, for anyone reviewing this:

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

I went with **IsolationForest (scikit-learn), fit only on windows of
"normal" data.**

The brief is pretty clear that this should be an anomaly detector, not a
failure classifier — fit on normal data, then score new windows. That's
exactly what IsolationForest is built for: it doesn't need labels at fit
time, and it works by isolating points that are easy to separate from the
rest of the data with just a few random splits.

It's also just cheap to run. Training on 4,513 windows takes a fraction of
a second, no GPU needed, and the artifact that comes out is a small joblib
file rather than a multi-hundred-MB checkpoint. And because it's
tree-based rather than distance-based, it doesn't care about feature
scale — so I don't have to ship and version a fitted `StandardScaler`
alongside the model, which is one less stateful thing that training and
serving could quietly drift apart on.

I didn't want to just assert this was the right call, though — §2 below
has an actual four-way benchmark against XGBoost, One-Class SVM, and Local
Outlier Factor on identical data. And honestly, one of the results
surprised me: One-Class SVM edges IsolationForest out on the raw numbers.
§2 walks through that and explains why IsolationForest still ships anyway.

A couple of things I considered and ruled out early:

- **Autoencoder (reconstruction error)** — I didn't benchmark this one
  (see §8). It's a lot more moving parts — architecture, training loop —
  for a ~10k-row dataset where a tree ensemble already picks up a
  reasonable signal.
- **Supervised classifier on `Machine failure`** — this is explicitly not
  what the brief is asking for, and it wouldn't generalize to a real
  deployment anyway, since you don't have failure labels for a brand-new
  machine. I still benchmark it in §2 (as XGBoost) just to show what a
  labeled upper bound looks like, so the "why not" isn't just hand-waving.

## 2. Model performance & algorithm comparison

### Methodology

`train/train_model.py` and `experiments/compare_models.py` share one
train/val/test split (`train/data_prep.py`), so every number below is
honest rather than cherry-picked:

- **Pure-normal windows** (7,521 of them — zero failure rows) get split
  60/20/20 into **train / val / test**. Only `train` (4,513 windows) is
  ever used to fit an unsupervised model.
- **Held-out failure-containing windows** (2,470 of them, never used to
  fit anything unsupervised) are split 50/50 into **val / test**.
- **val** is used only for picking hyperparameters (ranked by PR-AUC) and
  the threshold. **test** gets touched exactly once, after every other
  decision is already locked in — so the numbers below are a real estimate
  of generalization, not a threshold I tuned to look good.
- IsolationForest itself gets a small grid search over `n_estimators` /
  `max_samples` / `max_features` — the parameters that actually change
  what the isolation trees look like. (`contamination` only shifts an
  internal `offset_` that this service doesn't even use, since it derives
  its own threshold from validation data instead of `model.predict()`.)

### Two thresholds — and why the "obviously better" one isn't what's deployed

For every model below I report two thresholds:

- **F1-optimal** — sweeps the precision/recall curve on val and picks
  whatever maximizes F1.
- **~5%-FPR** — the 95th percentile of *normal*-window scores on val, so
  it's a fixed false-positive budget instead of depending on how many
  positives happen to land in a given split.

IsolationForest's F1-optimal threshold looks great in isolation
(precision 0.603, recall 0.807, F1 0.690) — until you check what it
actually does in practice: it flags **43.7%** of normal windows as
anomalous. That number comes from how "anomalous window" is defined here —
with `stride=1`, a single failure row contaminates 10 overlapping windows,
so roughly a quarter of *all* windows in this dataset end up labeled
"anomalous." That's nothing like a real deployment, where almost all
traffic is normal. A threshold tuned for F1 on that near-balanced
evaluation set ends up chasing recall at the cost of crying wolf on nearly
half of normal operation — bad enough that it flagged this repo's own
"normal" example (`sample_request.json`) as anomalous the first time I
shipped it. So **the ~5%-FPR threshold is what's actually deployed**,
because it matches the actual goal: stay quiet on normal data, flag real
deviations. Both thresholds are reported below so the trade-off is visible
instead of buried behind one convenient number.

### Headline result: IsolationForest (deployed)

| Split | ROC-AUC | PR-AUC |
| --- | --- | --- |
| test (1,504 normal + 1,235 anomalous, never touched until this number) | 0.769 | 0.740 |

| Threshold | Precision | Recall | F1 | False-positive rate | Confusion matrix (TN/FP/FN/TP) |
| --- | --- | --- | --- | --- | --- |
| **~5%-FPR (deployed, τ=0.714)** | 0.846 | 0.325 | 0.469 | 0.049 | 1431 / 73 / 834 / 401 |
| F1-optimal (reference only, τ=0.268) | 0.603 | 0.807 | 0.690 | 0.437 | 847 / 657 / 238 / 997 |

For comparison, a pre-tuning baseline (just aggregate features — mean/
std/min/max/trend, no hyperparameter search, single in-sample 95th
percentile threshold) caught about 29% of failure-containing windows at a
~5% false-positive rate. The deployed model above catches **32.5%** at
basically the same FPR — a real if modest improvement, and mostly thanks
to the `max_dev` spike feature (§3) rather than the hyperparameter search,
which actually helped PR-AUC more than it helped this specific operating
point.

### Full comparison: IsolationForest vs XGBoost vs One-Class SVM vs Local Outlier Factor

You can run this yourself: `pip install -r experiments/requirements.txt
&& python experiments/compare_models.py`. All four models are fit/scored
on the exact same split above; full output (including the notes column)
is saved to [`experiments/results.md`](experiments/results.md).

| Model | ROC-AUC | PR-AUC | Test @ F1-optimal (P / R / F1 / FPR) | Test @ ~5%-FPR (P / R / F1 / FPR) | Fit time | Satisfies brief? |
| --- | --- | --- | --- | --- | --- | --- |
| **IsolationForest (deployed)** | 0.769 | 0.740 | .603 / .807 / .690 / .437 | .846 / .325 / .469 / .049 | 0.30s | yes |
| One-Class SVM | 0.763 | **0.758** | .582 / .810 / .678 / .477 | .870 / .370 / **.519** / .045 | **0.05s** | yes |
| Local Outlier Factor | 0.724 | 0.731 | .521 / .806 / .633 / .609 | .866 / .340 / .488 / .043 | 1.54s | yes |
| XGBoost (supervised) | **0.987** | **0.985** | **.977 / .883 / .928** / .017 | .950 / .921 / .935 / .040 | 1.87s | **no** |

**XGBoost wins on every metric — and gets disqualified anyway.** It was
trained directly on failure labels (see `experiments/compare_models.py`
for exactly which windows), which is exactly what the brief rules out —
"fit on normal data, score new windows," not row-wise supervised
classification. It also just isn't realistic: a real deployment usually
doesn't have failure history for a brand-new machine yet. I kept it in the
table as an honest upper bound — "how good could this be if we cheated" —
not as a real candidate.

**One-Class SVM measurably beats IsolationForest**, and it's not noise —
refitting both across 5 independent random splits confirms it:

| Model | PR-AUC (mean ± std, 5 splits) | Deployed-threshold F1 (mean ± std) |
| --- | --- | --- |
| IsolationForest | 0.747 ± 0.004 | 0.487 ± 0.017 |
| One-Class SVM | 0.761 ± 0.007 | 0.515 ± 0.004 |

It's also *faster* to fit at this scale (~4.5k rows) — its `O(n²)`-ish
training cost only starts to hurt at scales this project never reaches.
Going in, I'd assumed (still visible in git history) that OCSVM would be
"more sensitive to feature scaling and kernel choice, slower... not
obviously better." Turns out that was wrong on both counts — a good
reminder to actually check assumptions like that instead of shipping them
as fact.

**So why does IsolationForest still ship instead of OCSVM?** Honestly,
the margin is real but modest — about 1.4pp PR-AUC, 3pp F1 — and I weighed
it against two things that aren't in the metrics: (1) OCSVM needs a fitted
`StandardScaler` bundled into the artifact and applied identically at
serving time, which is a second stateful object that has to stay
version-locked with the model — exactly the kind of train/serve skew risk
`app/features.py` was written to avoid elsewhere in this repo; (2) this
comparison is a single stratified split validated across 5 seeds, not
full k-fold cross-validation, and I didn't sweep OCSVM's `nu`/`gamma` the
way I swept IsolationForest's params — so I'm not fully confident the gap
is optimized on either side. **If I had another hour, switching the
default to One-Class SVM would be the first thing I'd go validate
properly** — the evidence is real, not hand-waved. See §8.

## 3. Preprocessing

**The core problem:** AI4I 2020 is tabular — one row per machine
snapshot, no timestamp, nothing that says "this row follows that row on
the same machine." Per the task note, I treat it as a proxy sensor stream
by sorting on `UDI` (the dataset's row order, 1..10000) and sliding a
window over it. That's a modeling assumption I'm making explicitly, not a
claim that these rows are actually a real time series — see the caveat in
§8.

**Windowing:** `window_size=10`, `stride=1` (`train/data_prep.py`), which
gives 9,991 overlapping windows
(`app/features.py::build_sliding_windows`, a linear scan over the row
sequence — see §7). Windows made entirely of non-failure rows are used to
fit the model (7,521 of them, further split per §2); windows that contain
at least one failure row are held out and only ever used for validation/
test evaluation, never for fitting.

**Feature extraction (`app/features.py`, shared by training and
serving):** for each of the 5 sensor channels (air temperature, process
temperature, rotational speed, torque, tool wear) I compute `mean`,
`std`, `min`, `max`, `trend` (last − first), and `max_dev` over the
window, plus one `window_length` feature — 31 features total. I added
`max_dev` — the max `|reading − window median|` — specifically because
the other aggregate stats dilute a single bad reading: a window is 10
rows, and if only 1 of them is the actual failure, the other 9 normal
rows pull mean/std back toward "normal," while a robust median barely
moves — so `max_dev` stays sensitive to exactly that spike
(`tests/test_features.py` has a unit test that demonstrates this
directly). Using fixed aggregate stats instead of raw sequences also
means the same code handles a 10-reading training window and a 3-reading
or 15-reading window from a client without any shape mismatch.

**Normalization:** IsolationForest splits on raw feature values — it's
tree-based, not distance-based — so no `StandardScaler` is needed for the
model itself (see §2 for why that matters when comparing against kernel-/
distance-based alternatives). What *is* normalized is the **output
score**: raw `-decision_function()` values get min-max scaled using the
0.5th/99.5th percentile of the model's own training scores, then clipped
to `[0, 1]`, so the API always returns a score you can reason about
instead of a raw IsolationForest number that could land anywhere from
-0.5 to 0.5.

**Missing / irregular readings:** any reading can leave out any channel
(all sensor fields in `SensorReading` are `Optional`, see
`app/schemas.py`). Missing values get imputed with the training-set
per-channel median (stored in `model_artifact/anomaly_model.joblib`)
before feature extraction. Windows of any length ≥ 1 are accepted —
`std`/`trend`/`max_dev` degrade to `0` for a single reading instead of
`NaN`. This is a deliberate trade-off: silently imputing keeps the
endpoint robust to real-world sensor dropouts, at the cost of not telling
the caller which fields were imputed (see §8 for the fix I'd make).

**Train/serve parity:** `app/features.py` is the *only* place feature
vectors get built, imported by both `train/data_prep.py` and
`app/model.py`. That means training and inference can't silently drift
apart in how a window turns into a feature vector — which is probably the
single most common way anomaly-detection services quietly break in
production.

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

`type` (the product quality variant, L/M/H) is accepted but not currently
used by the model — see §8. Every sensor field is optional per-reading (a
dropped sensor value gets imputed, not rejected); an empty `readings`
list is rejected with `422`.

### `GET /health`

```json
{"status": "ok", "model_loaded": true}
```

## 5. Build & run

```bash
docker build -t iot-anomaly-service .
docker run -p 8000:8000 iot-anomaly-service
```

Once it's running, FastAPI gives you interactive docs for free at
[http://localhost:8000/docs](http://localhost:8000/docs) — handy for
poking at the API without curl.

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
machine row, included as `sample_anomaly_request.json`. It scores
noticeably higher and crosses the threshold:

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

Dev tooling is deliberately kept out of the runtime image — install it
separately if you need it:

```bash
pip install -r requirements-dev.txt   # pytest, httpx, mypy, ruff, pandas-stubs
```

```bash
pytest -q                          # tests/ — API smoke tests + feature-engineering unit tests
mypy app train tests               # strict mode (see [tool.mypy] in pyproject.toml)
ruff check .                       # lint
ruff format --check .              # formatting
```

All four are clean on this repo right now — 10 tests passing, 0 mypy
errors in strict mode across `app/`, `train/`, and `tests/`, 0 ruff
findings. `pyproject.toml` holds config for both tools; the only
type-checking gaps are `ignore_missing_imports` overrides for `joblib`,
`sklearn`, and `xgboost`, none of which ship type stubs — that's on them,
not a gap in this repo's own typing.

`tests/test_api.py` covers the API surface: health, a normal window, a
real failure-containing window that has to cross the threshold, missing
fields, and a rejected empty window. `tests/test_features.py` covers the
shared feature-engineering module directly — the prefix-sum windowing
helper checked against a naive reference implementation, stride handling,
the missing-value/short-window edge cases in `extract_features`, and a
targeted test proving `max_dev` catches a single-row spike that `std`
dilutes.

`experiments/` (see §2) has its own `requirements.txt` — `xgboost` is a
fairly large native-code dependency that routine dev work (API, tests,
lint, type-checks) never actually needs, so I kept it out of
`requirements-dev.txt`.

## 7. Complexity notes

Windowing and feature extraction are linear in the data, not accidentally
quadratic:

- `build_sliding_windows`: **O(n)** — one pass over the row sequence,
  `n_windows = (n_rows - window_size) / stride + 1`.
- `extract_features`: **O(w)** per window (w = window size) — you have to
  read a window at least once to compute any aggregate over it, so this
  is already about as good as it gets; total cost across training is
  O(n · w).
- Classifying each window as "pure normal" vs "contains a failure"
  (`train/data_prep.py::build_windows`) originally did two full
  `all(...)` / `any(...)` scans per window — O(n · w) with a constant
  factor of 2. I replaced it with `sliding_window_sums`
  (`app/features.py`), a **prefix-sum** over the 0/1 failure flags: O(n)
  to build the prefix array, then O(1) per window to read off its failure
  count via `prefix[end] - prefix[start]`. I verified this is
  behavior-preserving — retraining after the change reproduces the exact
  same 7,521 normal / 2,470 held-out window split as before.
- Inference: one `IsolationForest.decision_function` call per request
  over 300 trees, each O(log(max_samples)) deep — effectively O(1)
  relative to request size, dominated by the O(w) feature extraction for
  whatever window the client sent.

At n=10,000 rows none of this is a measured bottleneck (training runs in
well under a second either way) — I made the prefix-sum change because
it's the right tool for a "sum over every sliding window" query and costs
nothing extra to write correctly, not because O(n·w) was ever actually
slow here.

## 8. Trade-offs & what I'd change next

**Being honest about model quality first:** at the deployed ~5%-false-
positive operating point, the model catches 32.5% of failure-containing
windows (§2) — that's a real limitation, not a hidden one. A window is 10
rows, and if only 1 is the actual failure row, the other 9 normal rows
still shape most of the aggregate features; `max_dev` (§3) helps but
doesn't fully solve this. I'd rather report the honest number than pick a
threshold that looks good in isolation (see the F1-optimal-threshold trap
in §2).

**What I'd do next, roughly in priority order:**

1. **Actually validate switching the default to One-Class SVM.** §2's
   comparison shows it beating IsolationForest on PR-AUC and
   deployed-threshold F1, consistently across 5 random splits — not
   noise. Before switching, I'd want to k-fold cross-validate properly
   (this repo only checked 5 single splits), sweep `nu`/`gamma` the way
   `HYPERPARAM_GRID` sweeps IsolationForest's params, and bundle a
   versioned `StandardScaler` into the artifact.
2. **Better feature engineering over aggregation.** `max_dev` (§3) was
   the first spike-sensitive feature; a per-channel "rate of change of
   max_dev," or a rolling z-score against a longer machine history, could
   sharpen this further without changing the modeling approach.
3. **Weight by product `type`.** The dataset's own failure logic
   (overstrain threshold) actually depends on `type` (L/M/H); a per-type
   baseline — or even three separate small models — would probably
   sharpen detection instead of pooling all types together.
4. **True time-series windowing.** This is the biggest caveat in the
   whole exercise — AI4I rows aren't really sequential per-machine. Given
   a real device stream, I'd window per physical machine ID and by actual
   timestamp, not by CSV row order.
5. **Autoencoder / deep methods.** Not benchmarked here (§2 only compares
   classical methods). With more data — or a per-machine deployment that
   accumulates history over weeks — a small reconstruction-error
   autoencoder or Deep SVDD is worth adding to the comparison; a library
   survey via [PyOD](https://pyod.readthedocs.io/) would be the fast way
   to try several at once.
6. **Proper k-fold hyperparameter search.** `HYPERPARAM_GRID` in
   `train/train_model.py` is just 4 combinations scored on one val split;
   k-fold CV over a wider grid (and the same for OCSVM's `nu`/`gamma`)
   would give more confidence the chosen config isn't just a lucky split.
7. **Confidence/explainability in the response.** Return which channel(s)
   contributed most to the score (e.g. per-channel feature importance
   from the forest) so a technician gets an actionable "torque looks off"
   instead of a bare number.
8. **Explicit imputation flag.** Right now missing fields are silently
   imputed; I'd add an `imputed_fields` list to the response so a caller
   knows a reading was incomplete instead of trusting a score computed on
   partially fabricated data.
9. **Cost-sensitive threshold.** The ~5%-FPR threshold is a reasonable
   default, but the right operating point genuinely depends on the ratio
   of "cost of a missed failure" to "cost of a false alarm" for a
   specific deployment — worth exposing as a configurable knob rather
   than baking in one number.

**For real device streams on Azure**, specifically:

- Swap the CSV-driven training script for a scheduled retrain job (an
  Azure ML pipeline, or just an Azure Function on a timer) reading from
  wherever the device telemetry actually lands (Azure IoT Hub → Event Hub
  / ADLS), instead of a one-off local CSV.
- Put the service behind Azure Container Apps or AKS instead of `docker
  run` directly, with a readiness probe against `/health` so the model is
  fully loaded — not just "container started" — before traffic gets
  routed to it. The lifespan-loaded model already makes this a non-issue
  for cold requests, but the orchestrator still needs the health check.
- Real streams need actual per-device windowing with timestamps and
  gap-handling — a device that drops offline for 20 minutes shouldn't
  have its next reading treated as a `trend` of 20 minutes of drift. This
  repo windows over CSV row order as a stand-in for that.
- Model artifact versioning: store artifacts in Azure Blob Storage /
  Azure ML Model Registry with a version tag, and have the API pull the
  latest approved version at startup instead of a `COPY` baked into the
  image, so retraining doesn't require rebuilding and redeploying the
  whole service.
- Add basic auth / an API key (e.g. Azure API Management in front) —
  there's none right now, which is fine for a "not a production system"
  brief, but it'd be the first thing I'd add before this touched a real
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
