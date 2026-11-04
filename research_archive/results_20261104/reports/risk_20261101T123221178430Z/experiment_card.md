# OULAD BCE + Cox experiment

Primary endpoint: Fail + Withdrawn vs Pass + Distinction. Survival event: withdrawal.

Learner-disjoint 2013 training/validation only. External 2014 test disabled. System-only evaluation at the predeclared primary day; no comparator or ablation trained. Numerical assessment scores are omitted unless an availability proxy was explicitly requested.

Resolved settings:

```json
{
  "windows": [
    7,
    14,
    21,
    28,
    56,
    84,
    112,
    140,
    168
  ],
  "primary_day": 56,
  "seeds": [
    42
  ],
  "epochs": 100,
  "patience": 20,
  "batch_size": 256,
  "learning_rate": 0.001,
  "cox_weight": 1.0,
  "score_lag": 7,
  "score_policy": "omit",
  "bootstrap": 1000,
  "permutations": 1000,
  "threads": 6,
  "device": "cuda",
  "split_seed": 42,
  "folds": 1,
  "split_scheme": "holdout",
  "train_pool_fraction": 0.7,
  "validation_fraction": 0.3,
  "min_delta": 0.0001,
  "precision": "FP32",
  "deterministic": true,
  "inference_batch_size": 512,
  "train_probe_max_rows": 512,
  "optimizer": "Adam",
  "optimizer_fused": true,
  "evaluate_test": false,
  "mlp_iterations": 200,
  "synthetic": false,
  "improved": true,
  "feature_profile": "rich",
  "tuning_trials": 4,
  "screen_epochs": 15,
  "models": [
    "proposed",
    "lstm",
    "lightgbm",
    "ensemble_lstm_attention_lgbm_weighted"
  ],
  "system_only": true,
  "protocol": "oulad_risk_nested_tuning_v2"
}
```

See run_manifest.json, environment.json, source/, day_*/cohort_audit.json, fold metrics, learning curves, fit diagnostics and calibration reports for complete details.
