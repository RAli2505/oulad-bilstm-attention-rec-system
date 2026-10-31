# OULAD Dropout Prediction v2 — Static-Fusion Redesign (Design Spec)

**Piece D of MARS v2**, contribution §3.3. Supersedes the clicks-only v1 design (`2026-06-21-oulad-pred-design.md`). v1's leak-free early-warning model (clicks `date < first_deadline` only) topped out at `dropout_auc ≈ 0.63` — diagnosed as **signal poverty** (within-presentation re-split still only 0.68), not a model defect. v2 keeps the same leak-free early cutoff but adds the features that actually carry dropout signal: student demographics and early-assessment behaviour. Target `dropout_auc ≥ 0.85`.

## Goal

Predict student dropout at the `first_deadline` cutoff from (a) early VLE click sequences, (b) static demographics, and (c) early-assessment submission/score. A shared LSTM encodes clicks; a static feature vector is fused (late fusion) before two heads: binary dropout classifier (ROC-AUC) and Cox proportional-hazards risk (C-index). The clicks-only v1 result is preserved as an ablation baseline.

## Scope

- OULAD only. XES `agents/prediction_agent.py` NOT touched.
- Modify existing `models/pred_oulad.py` + `tests/test_pred_oulad.py` (no new module — extends v1 in place).
- Branch `feat/oulad-pred` (continues v1 work).
- Late-fusion multi-task model. No new third-party dependency (Cox loss + C-index already hand-rolled; one-hot/standardize by hand with numpy/pandas).

## Why static features (measured, not assumed)

Leak-free signal confirmed on real OULAD before design:
- **Early assessment submission** (`date_submitted < first_deadline`): present → 26.5% dropout, absent → 49.5% dropout (n=9832 vs 5757).
- **Early score**: non-dropout mean 81.5 vs dropout 73.9.
- Demographics already known OULAD predictors: `num_of_prev_attempts`, `highest_education`, `imd_band`, `studied_credits`.

## Data sources

- v1 artifacts (unchanged): `data/oulad/processed/{daily_vle,student_target}.parquet`, `data/oulad/splits/{train,val,test}.parquet`.
- `student_target.parquet` **already carries demographics** (piece A joined `studentInfo`): `gender, region, highest_education, imd_band, age_band, num_of_prev_attempts, studied_credits, disability, date_registration`. No new read for demographics.
- NEW raw reads for early-assessment features: `studentAssessment.csv` (`id_student, id_assessment, date_submitted, is_banked, score`) + `assessments.csv` (`id_assessment, code_module, code_presentation, assessment_type, date`). Path via `--data-dir` (default `../../датасет/oulad`), same pattern as pieces B/C.

## Leak-free static features (cutoff = `first_deadline`, identical to v1 click cutoff)

Build a per-enrollment static vector `s ∈ R^S`:

**Demographics** (from `student_target`):
- Categorical → one-hot, vocab pinned from the train pool: `gender, region, highest_education, imd_band, age_band, disability`. Unseen/NaN category → all-zero row + a per-field `*_missing` flag (imd_band has NaNs).
- Numeric → standardized (train-pool mean/std): `num_of_prev_attempts, studied_credits, date_registration`. NaN → 0 after standardization (i.e. impute to train mean) + `date_registration_missing` flag.

**Early-assessment** (join `studentAssessment`+`assessments`, keep only `date_submitted < first_deadline`):
- `submitted_any` (0/1) — strongest single feature.
- `n_early_submitted` (count, standardized).
- `mean_early_score` (standardized; 0 + `score_missing` flag when no early submission).
- `banked_any` (0/1, from `is_banked`).

Normalization stats and one-hot vocabularies are **fit on the train pool only**, stored on the `OULADPred` instance, and applied unchanged to the test split (no leakage). `S` is determined at fit time from the vocab sizes.

## Model (late fusion)

```
clicks [B, 61, F] → LSTM(F, 128, 2, dropout 0.3) → masked temporal attention → ctx [B, 128]
static [B, S]    ───────────────────────────────────────────────────────────┐
concat(ctx, static) [B, 128+S] → MLP(128+S → 64, ReLU, Dropout 0.3) → h [B,64]
  → cls_head:  Linear(64 → 1)   # dropout logit   (BCEWithLogitsLoss, no pos_weight)
  → risk_head: Linear(64 → 1)   # Cox risk scalar (Cox partial-likelihood)
```

- `PredNet.__init__(n_feat, n_static)`; `forward(x, lengths, static)`. Everything else (LSTM, attention mask, Cox loss, C-index, training loop, early stop) unchanged from v1.
- Joint loss `L = bce + LAMBDA_COX * cox`, `LAMBDA_COX = 1.0`. Adam lr=1e-3, ≤100 epochs, early stop patience 20 on val loss. Seed 42. cuda if available.

## Metrics & evaluation

Train on train+val pool (stratified 85/15 internal split for early-stop monitoring), final metrics on held-out **test** (2014 cohort) — same protocol as v1:
- `dropout_auc` = ROC-AUC of classifier head. **Target ≥ 0.85.**
- `c_index` = hand-rolled Harrell concordance of Cox risk head.
- `brier` = Brier score of classifier probabilities.

## Ablation (preserve clicks-only result)

Two recorded results for a dissertation-honest ablation:
- `results/oulad/pred_metrics_clicks_only.json` — v1 clicks-only baseline (already saved as `pred_metrics_baseline.json`; rename/copy to this name). `dropout_auc≈0.63`.
- `results/oulad/pred_metrics.json` — v2 fusion result (new default output).

Ablation table (clicks-only vs +static) quantifies how much signal lives in static features vs clicks — a clean contribution-§3.3 finding.

## Fallback B (only if v2 < 0.85)

If the fusion smoke build still misses `dropout_auc ≥ 0.85`, widen the click observation window from `date < first_deadline` to `date < 0.25 * module_presentation_length` (≈ first quarter of the course) via a constructor arg `window_frac` (default `None` = first_deadline). Static-assessment cutoff stays at `first_deadline` (leak-free). Documented as an early-warning→accuracy trade-off. Decided at smoke time, not pre-built.

## Artifacts

```
models/pred_oulad.pt                        # fusion model state_dict
results/oulad/pred_metrics.json             # v2 fusion: dropout_auc, c_index, brier, n_train, n_test, n_pos
results/oulad/pred_metrics_clicks_only.json # v1 baseline (ablation)
```
CLI `python -m models.pred_oulad --build --data-dir "../../датасет/oulad"`. Artifacts NOT committed (derived).

## Interface (changes from v1)

```python
class OULADPred:
    def __init__(self, processed_dir="data/oulad", data_dir="../../датасет/oulad",
                 model_dir="models", results_dir="results/oulad", window_frac=None): ...
    def build_static(self, target, assess, keys, fit=False) -> np.ndarray   # [N, S]; fit=True pins vocab/stats
    def build_sequences(...) -> (X, lengths, y_drop, t_event, e_event)      # unchanged
    def train(self, Xtr, len_tr, str_tr, ytr, ttr, etr, Xval, ...) -> model # +static arg
    def evaluate(self, Xte, len_te, str_te, yte, tte, ete) -> dict          # +static arg
    def build(self) -> dict                                                  # full pipeline, writes both jsons
```

## Tests (`tests/test_pred_oulad.py`)

Keep v1's 7 tests (sequences, cox, c-index, train/eval, integration). Add:
8. `test_build_static_shapes_and_leakfree` — small target+assessment fixture → static `[N, S]`; assessment rows with `date_submitted >= first_deadline` excluded (`submitted_any=0` when only late submission exists); one-hot/standardize fit on train applied to a held-out row with an unseen category → all-zero one-hot block, no crash.
9. `test_fusion_forward_shapes` — `PredNet(n_feat, n_static)` forward with clicks+static returns `(cls [B], risk [B])`; static actually influences output (two different static vectors → different logits).

Update v1 tests that call `train`/`evaluate`/`PredNet` for the new static argument (pass a zero static matrix where static is irrelevant to the assertion).

After tests pass: real smoke build, record fusion `dropout_auc/c_index/brier` + the clicks-only baseline for the ablation table. If <0.85, apply fallback B and re-smoke.

## Known limitations (dissertation honesty)

- Click window stays `date < first_deadline` (early warning); static cutoff identical. v1's signal-poverty finding (clicks-only ≈0.63) is reported as the ablation baseline.
- Early-assessment "score available at first_deadline" assumes a submitted-by-cutoff score is observed at prediction time — true for an at-deadline early-warning decision, stated explicitly.
- Survival event still `Withdrawn` only; Cox in-batch risk set; tie handling via logcumsumexp approximation (c_index unaffected). Single seed 42.
- region/imd one-hot is high-cardinality for an MLP (a GBM would handle it natively); standardization mitigates. Acceptable given the click branch needs the neural encoder.

## Out of scope (YAGNI)

§3.4 outcome-weighted ranking, orchestrator wiring, calibration recalibration, global Cox risk set, multi-seed, figures, XES PredictionAgent changes, GBM comparison model.

## Constraints

No AI authorship in commits. Stage only `models/pred_oulad.py`, `tests/test_pred_oulad.py`, docs. Never `git add -A` (unrelated WIP in tree). bash on Windows (forward slashes, /dev/null).
