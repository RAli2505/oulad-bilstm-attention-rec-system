# Full hybrid system experiment implementation plan

> Execute inline with superpowers:executing-plans; independently review the whole change.

**Goal:** User-authorized full retraining command plus scientific recommendation evaluation, without evaluating 2014 or altering archived forecasts.

**Architecture:** Existing improved forecast runner remains numerically unchanged. A wrapper captures its output directory, samples child-process memory and passes read-only saved components, weights, calibration and exact split indices to an independent recommendation experiment. Features are reused across methods/seeds. Only recommendation rankers/calibrators are newly trained by this module.

**Tech stack:** Python, numpy/pandas/scipy, existing sklearn/joblib/pyarrow/matplotlib. ctypes RSS sampling on Windows; no new dependencies.

## Constraints

- Exactly windows 7,14,28,56,84,112,140,168; only 2013 train/validation.
- Fit / selection / calibration / held groups remain learner-disjoint.
- Training history date<t; future clicks [t,t+7) are proxy labels only.
- Outcome endpoints Fail+Withdrawn and complementary Pass+Distinction, each with ROC/PR-AUC, P/R/F1, calibration and confusion at fixed threshold .5.
- Top-K P/R/NDCG/F1/MAP/MRR/HitRate at 5/10/20; complete candidate sets and explicit cold/no-positive-query behavior.
- Scores are not probabilities. Fixed Platt maps use calibration only. Selection alone chooses logistic C.
- Fresh output directories, SHA verification of raw data, no source-model writes, no test evaluation.
- Primary day56 NDCG@10 contrast hybrid risk versus neutral risk. Other windows exploratory; learner-cluster CIs and seed variability separate.
- Fast execution preserves baseline FP32, deterministic settings, budget and student splits. No fabricated epoch curves for heuristics.

## Tasks

- [x] RED tests for metrics, complementary endpoint PR-AUC, candidate exports and CLI defaults.
- [x] Implement metrics, calibration diagnostics, cluster bootstrap and sampled process memory.
- [x] Implement shared candidate feature generation, fit-only popularity, two tuned logistic rankers, ablations, independent calibration, artifacts and plots.
- [x] Implement one-command wrapper around existing forecast training with new outputs and source snapshots.
- [x] Prove held/calibration labels cannot select ranker C; end-to-end synthetic run; source SHA preservation.
- [x] Independent review; fix raw provenance and exact-fit enrollment popularity issues with regression evidence.
- [x] Final full suite, real bounded pilot, preflight and commit.

## Verification ledger

- Initial 8 metric/API tests RED; implemented metrics and candidate export. AUC complement check uses tolerance for floating-point roundoff.
- Nonbinary-label regression RED, then validation before conversion prevents silent rounding.
- Small synthetic full run passed: nine recommendation methods, selection-only C search, calibration, both outcome classes, artifacts and source SHA unchanged.
- Review found raw input provenance gap: all three raw CSV hashes now required to match the source forecast manifest before loading.
- Exact-fit enrollment regression RED: popularity previously included other course enrollments of fit students. Now joins full enrollment keys; no cross-course contribution.
- Preflight detected NVIDIA GeForce RTX 5050 Laptop GPU; exact eight windows, seed42, 18 methods, holdout, FP32. No forecast training started.
- Real pilot day56, seed42, 12 learners per split completed. This is a functionality check, not a scientific quality claim; rerun after final population fix.
- Ruling: keep external test closed and preserve original forecast code; prospective generalization and causal educational benefit remain outside these validation results.
- Ruling: five seeds are default for stability; one seed remains an explicit faster first pass. New AUC after retraining is measured, not guaranteed to exceed the historical value.
- Final full suite: 101 passed, 15 joblib/NumPy deprecation warnings, PYTEST_EXIT_CODE=0. The outer command again reported exit 1 after test completion; this process-exit discrepancy is recorded rather than presented as a clean shell exit. The session-only normal-permission tmp_path fixture was used as in the earlier recommendation implementation; existing test code was not changed for that workaround.
- Final real pilot after the exact-fit popularity fix completed at test-artifacts/recommendation-metrics-final-pilot-585e103dc38a4d55bf2809744512acfd: day56, seed42, maximum 12 learners per split, bootstrap25. Artifacts and outcome/recommendation reports generated successfully; this is not an estimate of full-cohort recommendation quality.
- Protected forecast/data/archive paths have an empty diff against 56d0a9f. All full-retraining work remains unlaunched; the user requested the command.
