# OULAD adaptive recommendations implementation plan

> Implement inline with superpowers:executing-plans. The user approved the separate OULAD module and requested an adaptive preference model.

**Goal:** Recommend specific OULAD resources using changing preferences, observed activity, course schedule and an unchanged ensemble forecast.

**Architecture:** Adapt the evidence scoring and diversity ideas of xes-mars recommendation_agent.py, retaining its separation of forecasts and recommendation utilities. Implement independent modules in `recommendations/`; do not import the MARS orchestrator, IRT, FAISS or knowledge graph. OULAD course risk is not a topic probability. Do not apply relative_topic_need to the scalar risk.

**Tech stack:** Python 3.12, pandas, numpy, pytest; no new dependencies.

## Constraints and design

- Base: b461d88, branch feature/oulad-adaptive-recommendations.
- Existing model code, input preprocessing, predictions, archives and xes-mars source stay unchanged.
- Enrollment identity is (id_student, code_module, code_presentation).
- History uses date < as_of_day; all supplied history is filtered before aggregation. Never consume final_result, target, score or future interactions.
- Preserve id_site. Catalog metadata, not interaction history, defines the candidate set.
- `available_from_day`/`available_until_day`, if supplied, are explicit release/expiry bounds. OULAD week_from/week_to are planned teaching weeks, not verified release dates: use week_from as a conservative scheduling gate and week_to only for calendar relevance. Unknown week_from is allowed and reported as unknown.
- PreferenceModel.profile(events, resources, enrollment, as_of_day) recomputes a snapshot: 14-day half-life, log1p daily clicks capped at 20, normalize by type's eligible or previously observed catalog size, neutral Dirichlet smoothing strength 3. Confidence uses decayed distinct resource-days. This estimates interaction preference, not mastery or educational benefit; opportunity correction is approximate without impression logs. Expiry must not erase historical evidence.
- Short-term versus older type shares expose change in interests; low confidence shrinks preference contributions toward neutral. Snapshots are JSON-serializable and deterministic, do not mutate caller inputs, and can be rebuilt after new events.
- Rank candidates with preference, calendar and engagement evidence. Course risk changes the calendar/engagement tradeoff only; no estimated intervention effect is claimed. Apply an MMR-style type diversity penalty. Filter recently seen materials by default; expose an opt-in review mode.
- CLI reads raw OULAD and one explicitly selected validation prediction by enrollment, never target columns. It requires an explicit day and prediction source; records forecast provenance; creates output exclusively, without overwriting results.
- No model retraining or claims of measured recommendation accuracy. Resource names/content are absent; explanations use id_site and activity_type.

## Review focus

1. Future events, other enrollments and future resources must not affect today's profile/ranking.
2. Empty history/catalog, zero clicks and missing planned weeks must return stable finite results.
3. Repeated clicks, duplicates and reordered history must not dominate or change outputs unexpectedly.
4. Invalid probabilities, days, resource IDs, negative clicks and ambiguous prediction rows must fail clearly.
5. Recommendations must leave original forecasts and input tables unchanged.

## Tasks

- [x] Write behavioral tests for snapshots, preference drift, isolation, candidate eligibility, diversity, risk-dependent ranking and forecast preservation; observe RED.
- [x] Implement recommendation_signals.py and preferences.py with validated snapshot inputs and immutable results; verify preference tests.
- [x] Implement recommendation_agent.py and explicit prediction-file CLI; verify ranking and CLI tests.
- [x] Document formulas, assumptions, source adaptation and executable use; run full suite and a read-only real-data example to a new temporary output.
- [x] Review the complete diff independently, address material findings, verify protected files unchanged and commit on the new branch.

## Verification record

- Initial RED: 26 failures for missing module; then 26 passed. Adapter/update tests RED: 4 failures; then 30 passed.
- Independent read-only review found historical observations were erased by candidate expiry. Two regression tests failed first; separate history metadata now preserves observations. Recommendation suite: 32 passed.
- First full suite: 86 passed, eight existing joblib/NumPy deprecation warnings. Final suite after review fix: 88 passed, eight joblib/NumPy deprecation warnings; explicitly printed PYTEST_EXIT_CODE=0. The outer command nevertheless reported exit 1 after completion; the reason for this environment/process-exit discrepancy is unresolved. No test assertion failures remain.
- Windows pytest temporary-directory mode caused access errors. Tests ran with a session-only fixture creating normal temporary directories under ignored test-artifacts, disabling tmpdir/cacheprovider; no production code or existing tests modified for this environment workaround.
- Real CLI example: enrollment 101781/AAA/2013J, day 112, five resources. Forecast 0.08119283146293353 returned unchanged; saved forecast file SHA-256 still matches provenance after inference.
- Recomputed saved validation ROC-AUC: 0.872423079586162, consistent with 0.8724. No retraining and no test 2014.
- Protected paths diff against b461d88 is empty: models, risk_experiments, data, evaluation_metrics.py, training runners, research_archive, results.
- Ruling: implement in this session following the user's explicit approval to implement, with one independent read-only review. No extra plan approval round.
- Ruling: opportunity counts include currently eligible and previously observed resources, so expired material does not erase interests; exposure correction remains approximate.
- No recommendation quality or intervention benefit has been measured. Default ranking coefficients and decay parameters require a separate chronological evaluation.
