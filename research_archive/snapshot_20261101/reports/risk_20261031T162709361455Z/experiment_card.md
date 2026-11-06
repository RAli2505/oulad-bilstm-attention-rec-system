# OULAD BCE + Cox experiment

Primary endpoint: Fail + Withdrawn vs Pass + Distinction. Survival event: withdrawal.

Learner-disjoint 2013 training/validation only. External 2014 test disabled. Primary contrast: proposed vs no_cox at the predeclared primary day. Numerical assessment scores are omitted unless an availability proxy was explicitly requested.

Resolved settings:

```json
{
  "windows": [
    7,
    14,
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
    "lr",
    "rf",
    "svm",
    "xgboost",
    "lightgbm",
    "mlp",
    "proposed",
    "lstm",
    "bilstm",
    "transformer",
    "no_attention",
    "no_static",
    "no_cox",
    "paper_hybrid",
    "ensemble_equal",
    "ensemble_inner_weighted",
    "ensemble_lstm_attention_lgbm_equal",
    "ensemble_lstm_attention_lgbm_weighted"
  ],
  "protocol": "oulad_risk_nested_tuning_v2"
}
```

See run_manifest.json, environment.json, source/, day_*/cohort_audit.json, fold metrics, learning curves, fit diagnostics and calibration reports for complete details.
