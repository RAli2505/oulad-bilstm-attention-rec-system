# OULAD Prediction (LSTM + Survival) — Design Spec

**Piece D of MARS v2** (cross-domain), contribution §3.3 of the dissertation plan. Standalone trainable dropout-prediction model with a multi-task LSTM encoder. Outcome-weighted ranking (§3.4) is a SEPARATE later piece (integration, depends on C and D) and is OUT OF SCOPE here.

## Goal

Predict student dropout from early VLE engagement on OULAD. A shared LSTM encoder over the daily click sequence feeds two heads: a binary dropout classifier (ROC-AUC) and a Cox proportional-hazards risk head (C-index). Mirrors the role of the XES `PredictionAgent` (sequence encoder → risk vector) but for OULAD outcomes, and adds survival analysis.

## Scope

- OULAD only. The existing XES `agents/prediction_agent.py` is NOT touched.
- New module `models/pred_oulad.py` + `tests/test_pred_oulad.py`.
- Branch `feat/oulad-pred` (off `feat/oulad-bcm` = pieces A∪B).
- Multi-task model (Approach A): one shared encoder, two heads. No new third-party dependency — Cox partial-likelihood loss and C-index are implemented by hand (torch + numpy).

## Data source

Reads piece-A artifacts directly (consistent leak-free cutoff, reuses A's filtered students):
- `data/oulad/processed/daily_vle.parquet` — `id_student, code_module, code_presentation, date, activity_type, sum_click`
- `data/oulad/processed/student_target.parquet` — incl. `final_result, first_deadline, date_unregistration, module_presentation_length`
- `data/oulad/splits/{train,val,test}.parquet` — enrollment keys (A's ≥10 pre-deadline active-day filter already applied)

Students = the A enrollment keys `(id_student, code_module, code_presentation)`. Train+val used for training (early stop on val); **test** is the held-out final-evaluation split (as intended in A — no leakage).

## Observation window (leak-free cutoff)

Features use only clicks with `date < first_deadline` (same cutoff as pieces A/B). This frames the task as **early-warning prediction from first-weeks engagement**, not the plan's literal "4 weeks before module end" (which would leak in-window withdrawals into features). Documented honestly as a limitation. Withdrawals occurring before first_deadline are already excluded by A's active-day filter.

## Sequence representation

Per enrollment, build a matrix `[T, F]`:
- `T` = day index over `date < first_deadline`, padded/truncated to a fixed `MAX_DAYS = 61` (max observed first_deadline). The real per-student length is kept for masking.
- `F` = clicks per activity_type (fixed ~20 categories enumerated from `vle`/`daily_vle`; a stable sorted category list pinned at build time). Days with no activity = zero vector.
- Negative `date` values (pre-module access) are clipped into the window start; only `date < first_deadline` rows contribute.

## Labels

Two distinct targets (kept separate — honest to the data):
- **Binary dropout** (classifier head): `dropout = 1` if `final_result ∈ {Fail, Withdrawn}`, else 0 (`{Pass, Distinction}`).
- **Survival** (Cox head): `event = 1` only for `Withdrawn` (has `date_unregistration`); `time = date_unregistration`. All others are censored at `time = module_presentation_length` (`event = 0`). Fail is censored in survival (no withdrawal date) but positive in the binary head — these are deliberately different targets.

## Model (`PredOULAD` / `PredNet`, torch)

```
input  [B, T=61, F]
  → LSTM(input=F, hidden=128, num_layers=2, dropout=0.3, batch_first=True)
  → temporal attention (masked softmax over T on LSTM outputs) → context [B, 128]
  → MLP(128 → 64, ReLU, dropout)
  → cls_head:  Linear(64 → 1)   # dropout logit       (BCEWithLogitsLoss, pos_weight)
  → risk_head: Linear(64 → 1)   # Cox risk scalar      (Cox partial-likelihood)
```

- Padding mask from real sequence lengths so padded days do not contribute to attention.
- Joint loss: `L = bce + LAMBDA_COX * cox`, `LAMBDA_COX = 1.0` (configurable).
- Cox loss by hand: negative log partial likelihood over the in-batch risk set (sort by event time within the batch). `# ponytail: in-batch risk set, global risk set if C-index poor`.
- Class imbalance (~35% dropout): `pos_weight` in BCE.
- Adam lr=1e-3, ≤100 epochs, early stop patience 10 on val loss. Device: cuda if available else cpu. Seed 42.

## Metrics & evaluation

Train on train+val (early stop on val), final metrics on held-out **test**:
- `dropout_auc` = ROC-AUC of the classifier head (sklearn `roc_auc_score`). Target ≥0.85.
- `c_index` = concordance of the Cox risk head, computed by hand: over comparable pairs `(i, j)` where `event_i = 1` and `time_i < time_j`, fraction with `risk_i > risk_j` (ties = 0.5). Target ≥0.80.
- `brier` = Brier score (sklearn `brier_score_loss`) of the classifier head's calibrated probabilities. Target ≤0.15.

## Artifacts

```
models/pred_oulad.pt              # model state_dict
results/oulad/pred_metrics.json   # dropout_auc, c_index, brier, n_train, n_test, n_pos
```
Output dirs configurable via constructor (default `models`, `results/oulad`), like `bcm_oulad`/`hgk_oulad`. CLI `python -m models.pred_oulad --build`. Artifacts NOT committed (derived).

## Interface

```python
class OULADPred:
    def __init__(self, processed_dir="data/oulad",
                 model_dir="models", results_dir="results/oulad"): ...
    def build_sequences(self, daily, target, keys) -> (X, lengths, y_drop, t_event, e_event)
    def train(self, Xtr, lengths_tr, y_tr, t_tr, e_tr,
              Xval, lengths_val, y_val, t_val, e_val) -> model
    def evaluate(self, Xte, lengths_te, y_te, t_te, e_te) -> dict   # {dropout_auc, c_index, brier}
    def build(self) -> dict                                          # full pipeline, writes artifacts
```

## Tests (`tests/test_pred_oulad.py`)

Synthetic-fixture unit tests for logic correctness (no real data; fast, hermetic). Real-data validation is a separate smoke run of `build()` on the full dataset (mirrors piece C).

1. `test_build_sequences` — small daily/target → `X` shape `[N, MAX_DAYS, F]`, correct `lengths`, correct `y_drop`/`t_event`/`e_event`; padding is zero; only `date < first_deadline` rows counted.
2. `test_cox_loss_and_cindex` — Cox loss is finite on a toy batch; hand C-index = 1.0 on a perfectly risk-ordered toy set and 0.0 on the reversed order (verifies the formula on controlled inputs — cannot be checked on real data).
3. `test_train_evaluate` — mini synthetic dataset → `train` → `evaluate` returns `dropout_auc`, `c_index`, `brier`, each in `[0, 1]`; model output shapes correct.
4. `test_build_integration` (tmp_path) — minimal synthetic `daily_vle`/`student_target`/splits parquet → `build()` writes `pred_metrics.json` (with the metric keys) + `.pt` to tmp dirs.

After tests pass: run the real smoke build on the full OULAD data and record `dropout_auc`, `c_index`, `brier`, `n_train`, `n_test`, `n_pos`.

## Known limitations (for dissertation honesty)

- Observation window is `date < first_deadline` (early warning), not the plan's "4 weeks before module end" — chosen for leak-free reuse of A/B/C splits.
- Survival event is `Withdrawn` only (Fail has no withdrawal date and is censored in the Cox head).
- Cox risk set is in-batch, not global — a standard mini-batch approximation.
- Cox partial likelihood uses the sort+`logcumsumexp` approximation, which drops tied event-time peers from each other's risk set. OULAD `date_unregistration` is integer-day, so ties are common and the *training loss* is mildly biased. The reported survival metric (`c_index`, a ranking measure) is unaffected; proper tie handling (Efron) deferred.
- Single seed (42); multi-seed stability deferred to the experiments chapter.

## Out of scope (YAGNI)

§3.4 outcome-weighted ranking, orchestrator/Alert pipeline wiring, calibration recalibration (Platt/isotonic), global risk set, multi-seed runs, figures, XES PredictionAgent changes.

## Constraints

- No AI authorship in commits. Stage only `models/pred_oulad.py`, `tests/test_pred_oulad.py`, docs. Never `git add -A` (unrelated WIP in tree). bash on Windows (forward slashes, /dev/null).
