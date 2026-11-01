import math

import numpy as np
import pytest


def test_ranking_metrics_use_full_eligible_truth_and_requested_slots():
    from recommendations.experiment_metrics import ranking_metrics
    m = ranking_metrics([2, 1, 4], {1, 3}, 3)
    assert m['precision'] == pytest.approx(1 / 3)
    assert m['recall'] == .5
    assert m['map'] == .25
    assert m['mrr'] == .5
    assert m['ndcg'] == pytest.approx((1 / math.log2(3)) / (1 + 1 / math.log2(3)))
    assert m['hit_rate'] == 1
    assert ranking_metrics([], set(), 5)['ndcg'] == 0


def test_success_is_complementary_but_pr_auc_is_recomputed():
    from recommendations.experiment_metrics import outcome_metrics
    m = outcome_metrics([1, 0, 0, 0], [.8, .9, .3, .1])
    assert m['risk']['pr_auc'] == .5
    assert m['success']['pr_auc'] == pytest.approx(11 / 12)
    assert m['risk']['roc_auc'] == pytest.approx(m['success']['roc_auc'])
    assert m['risk']['brier'] == pytest.approx(m['success']['brier'])


def test_calibration_reports_error_and_reliability_counts():
    from recommendations.experiment_metrics import probability_metrics
    m, bins = probability_metrics([0, 1], [.1, .9], bins=10)
    assert m['brier'] == pytest.approx(.01)
    assert m['ece'] == pytest.approx(.1)
    assert m['log_loss'] == pytest.approx(-math.log(.9))
    assert sum(b['n'] for b in bins) == 2


def test_one_class_metrics_are_explicitly_undefined():
    from recommendations.experiment_metrics import probability_metrics
    m, _ = probability_metrics([0, 0], [.1, .2])
    assert m['roc_auc'] is None
    assert m['pr_auc'] is None
    assert m['prevalence'] == 0


def test_cluster_bootstrap_is_deterministic_and_respects_constant_delta():
    from recommendations.experiment_metrics import cluster_interval
    m = cluster_interval([.1, .1, .1], [1, 1, 2], repeats=100, seed=42)
    assert m['low'] == pytest.approx(.1)
    assert m['high'] == pytest.approx(.1)


def test_candidate_snapshot_preserves_original_top_k():
    from tests.test_recommendations import agent, KEY
    a = agent()
    standard = a.recommend(KEY, 28, .5)
    detailed = a.recommend(KEY, 28, .5, include_candidates=True)
    assert detailed.pop('candidates')
    assert detailed == standard


def test_experiment_config_is_validation_only_and_exact_windows():
    from run_system_experiment import build_parser, risk_command
    args = build_parser().parse_args([])
    command = risk_command(args)
    assert args.windows == [7, 14, 21, 28, 56, 84, 112, 140, 168]
    assert '--improved' in command
    assert '--system-only' in command
    assert '--score-policy' in command
    assert '--test' not in command
    assert args.seeds == [42]


def test_system_forecast_scope_is_only_three_components_and_one_ensemble():
    from run_risk_article import system_model_names
    assert system_model_names() == ['proposed', 'lstm', 'lightgbm', 'ensemble_lstm_attention_lgbm_weighted']


def test_native_exit_does_not_accept_incomplete_forecast(tmp_path):
    import json
    from run_system_experiment import build_parser, check_forecast_exit
    args = build_parser().parse_args([])
    (tmp_path / 'run_manifest.json').write_text(json.dumps({'status': 'running'}), encoding='utf-8')
    with pytest.raises(RuntimeError, match='completed'):
        check_forecast_exit(3221226505, tmp_path, args)
    with pytest.raises(RuntimeError, match='3221225786'):
        check_forecast_exit(3221225786, tmp_path, args)


def completed_forecast_fixture(directory):
    import json
    import pandas as pd
    from recommendations.experiment import sha256
    from run_risk_article import system_model_names
    from run_system_experiment import build_parser
    args = build_parser().parse_args([])
    config = {k: getattr(args, k) for k in ['windows', 'seeds', 'epochs', 'patience', 'batch_size',
              'threads', 'tuning_trials', 'screen_epochs', 'bootstrap', 'permutations']}
    config.update(models=system_model_names(), system_only=True, evaluate_test=False,
                  split_scheme='holdout', synthetic=False, device='cuda', precision='FP32',
                  deterministic=True, score_policy='omit', improved=True)
    (directory / 'run_manifest.json').write_text(json.dumps({'status': 'completed', 'signature': 'sig', 'config': config}), encoding='utf-8')
    for day in args.windows:
        base = directory / f'day_{day}'
        base.mkdir()
        (base / 'status.json').write_text(json.dumps({'status': 'completed', 'signature': 'sig'}), encoding='utf-8')
        pd.DataFrame([dict(id_student=i, code_module='AAA', code_presentation='2013J', target=i % 2) for i in range(4)]).to_csv(base / 'cohort_and_folds.csv', index=False)
        fold = base / 'fold_1'
        fold.mkdir()
        np.savez_compressed(fold / 'indices.npz', fit=[0], validation=[1], calibration=[2], held=[3], external_test=np.array([], dtype=int))
        pd.DataFrame({'model': config['models'], 'roc_auc': [.8] * 4}).to_csv(base / 'summary.csv', index=False)
        pd.DataFrame({'id_student': [3], 'code_module': ['AAA'], 'code_presentation': ['2013J'], 'target': [1]}).to_csv(base / 'validation_2013_predictions.csv', index=False)
        for model in config['models']:
            job = fold / 'seed_42' / model
            job.mkdir(parents=True)
            np.savez_compressed(job / 'predictions.npz', held_probability=[.8], held_score=[1.], val_score=[1.], calibration_score=[1.], fit_score=[1.])
            (job / 'training.json').write_text('{}', encoding='utf-8')
            (job / 'calibration.joblib').write_bytes(b'fixture-calibration')
            hashes = {name: sha256(job / name) for name in ['predictions.npz', 'training.json', 'calibration.joblib']}
            (job / 'status.json').write_text(json.dumps({'status': 'completed', 'signature': 'sig', 'artifacts_sha256': hashes}), encoding='utf-8')
    return args


def test_verified_forecasts_survive_post_completion_native_exit(tmp_path):
    from run_system_experiment import check_forecast_exit
    args = completed_forecast_fixture(tmp_path)
    result = check_forecast_exit(3221226505, tmp_path, args)
    assert result['recovered_after_verified_completion'] is True
    assert result['verified_model_jobs'] == 36
    assert result['exit_code'] == 3221226505
    assert result['warning']


def test_native_exit_recovery_rejects_corrupted_checkpoint(tmp_path):
    from run_system_experiment import check_forecast_exit
    args = completed_forecast_fixture(tmp_path)
    target = tmp_path / 'day_7/fold_1/seed_42/lstm/training.json'
    target.write_text('{"corrupted": true}', encoding='utf-8')
    with pytest.raises(RuntimeError, match='checksum'):
        check_forecast_exit(3221226505, tmp_path, args)


def test_probability_validation_rejects_nonfinite_and_out_of_range():
    from recommendations.experiment_metrics import probability_metrics
    for p in [[np.nan], [1.1], [-.1]]:
        with pytest.raises(ValueError):
            probability_metrics([1], p)


def test_nonbinary_targets_are_not_silently_rounded():
    from recommendations.experiment_metrics import probability_metrics
    with pytest.raises(ValueError):
        probability_metrics([.5, 1], [.1, .9])


def test_feature_building_keeps_future_clicks_out_of_preferences_and_popularity():
    import pandas as pd
    from recommendations.experiment import make_candidates
    from tests.test_recommendations import catalog, history
    frame = pd.DataFrame([dict(id_student=i, code_module='AAA', code_presentation='2013J') for i in range(1, 5)])
    indices = {r: np.array([i]) for i, r in enumerate(['fit', 'selection', 'calibration', 'held'])}
    events = pd.concat([history([(1, 1, 4), (2, 15, 1)]).assign(id_student=i) for i in range(1, 5)])
    courses = pd.DataFrame([dict(code_module='AAA', code_presentation='2013J', module_presentation_length=200)])
    before, queries, _, _ = make_candidates(catalog(), events, frame, indices, 14, 7, courses)
    events.loc[(events.id_student == 4) & (events.date == 15), 'id_site'] = 3
    after, _, _, _ = make_candidates(catalog(), events, frame, indices, 14, 7, courses)
    pd.testing.assert_frame_equal(before.drop(columns=['label']), after.drop(columns=['label']))
    assert before.label.tolist() != after.label.tolist()
    assert set(queries.role) == {'fit', 'selection', 'calibration', 'held'}


def test_ranker_selection_does_not_use_held_or_calibration_labels(tmp_path):
    import pandas as pd
    from recommendations.experiment import fit_ranker
    rows, queries = [], []
    for query, role in enumerate(['fit', 'selection', 'calibration', 'held']):
        queries.append(dict(query=query, role=role, start=len(rows), stop=len(rows) + 4))
        for item in range(4):
            rows.append(dict(query=query, item=item, kind='content', label=item % 2,
                             preference=.1 + .2 * (item % 2), calendar=.5, engagement=1.,
                             popularity=.2, confidence=.5))
    data, q = pd.DataFrame(rows), pd.DataFrame(queries)
    columns = ['preference', 'calendar', 'engagement', 'popularity', 'confidence']
    first = fit_ranker(data, q, columns, 42, tmp_path / 'a')
    data.loc[data['query'] >= 2, 'label'] = 1 - data.loc[data['query'] >= 2, 'label']
    second = fit_ranker(data, q, columns, 42, tmp_path / 'b')
    np.testing.assert_array_equal(first, second)


def test_popularity_uses_exact_fit_enrollments_not_other_courses_of_fit_students():
    import pandas as pd
    from recommendations.experiment import make_candidates
    from tests.test_recommendations import catalog, history
    frame = pd.DataFrame([dict(id_student=i, code_module='AAA', code_presentation='2013J') for i in range(1, 5)] +
                         [dict(id_student=99, code_module='BBB', code_presentation='2013J')])
    indices = {'fit': np.array([0, 4]), 'selection': np.array([1]), 'calibration': np.array([2]), 'held': np.array([3])}
    resources = pd.concat([catalog(), catalog().assign(code_module='BBB')])
    courses = pd.DataFrame([dict(code_module=c, code_presentation='2013J', module_presentation_length=200) for c in ['AAA', 'BBB']])
    events = pd.concat([history([(1, 1, 2)]).assign(id_student=i) for i in range(1, 5)] +
                       [history([(1, 1, 2)]).assign(id_student=99, code_module='BBB')])
    before, _, _, _ = make_candidates(resources, events, frame, indices, 14, 7, courses)
    extra = history([(2, 1, 4)]).assign(id_student=99)
    after, _, _, _ = make_candidates(resources, pd.concat([events, extra]), frame, indices, 14, 7, courses)
    np.testing.assert_array_equal(before.popularity, after.popularity)


def test_complete_experiment_on_small_fixture(tmp_path):
    import json
    import joblib
    import pandas as pd
    from scipy.special import logit
    from recommendations.experiment import run_recommendations, sha256
    from risk_experiments.training import Calibration
    from tests.test_recommendations import catalog, history
    raw, source, output = tmp_path / 'raw', tmp_path / 'risk', tmp_path / 'recommendations'
    raw.mkdir()
    day = source / 'day_14'
    fold = day / 'fold_1'
    fold.mkdir(parents=True)
    frame = pd.DataFrame([dict(id_student=i, code_module='AAA', code_presentation='2013J',
                              target=i % 2, role='train_pool_2013' if i < 19 else 'validation_2013') for i in range(1, 25)])
    frame.to_csv(day / 'cohort_and_folds.csv', index=False)
    indices = {r: np.arange(i * 6, (i + 1) * 6) for i, r in enumerate(['fit', 'validation', 'calibration', 'held'])}
    np.savez_compressed(fold / 'indices.npz', **indices, external_test=np.array([], dtype=int))
    catalog().to_csv(raw / 'vle.csv', index=False)
    pd.DataFrame([dict(code_module='AAA', code_presentation='2013J', module_presentation_length=200)]).to_csv(raw / 'courses.csv', index=False)
    events = pd.concat([history([(1, 1, 4), (2 if i % 2 else 3, 15, 2)]).assign(id_student=i) for i in range(1, 25)])
    events.to_csv(raw / 'studentVle.csv', index=False)
    model = fold / 'seed_42/ensemble_lstm_attention_lgbm_weighted'
    model.mkdir(parents=True)
    scores = logit(np.where(frame.target.to_numpy() == 1, .8, .2))
    cal = Calibration().fit(scores[indices['calibration']], frame.target.to_numpy()[indices['calibration']])
    joblib.dump(cal, model / 'calibration.joblib')
    for member in ['lstm', 'proposed', 'lightgbm']:
        directory = fold / 'seed_42' / member
        directory.mkdir()
        np.savez_compressed(directory / 'predictions.npz', fit_score=scores[indices['fit']])
    np.savez_compressed(model / 'predictions.npz', val_score=scores[indices['validation']],
                       calibration_score=scores[indices['calibration']], held_score=scores[indices['held']],
                       held_probability=cal.predict(scores[indices['held']]))
    (model / 'training.json').write_text(json.dumps({'members': ['lstm', 'proposed', 'lightgbm'], 'weights': [1 / 3] * 3}), encoding='utf-8')
    (source / 'run_manifest.json').write_text(json.dumps({'config': {'evaluate_test': False,
        'split_scheme': 'holdout', 'seeds': [42], 'windows': [14]}, 'status': 'completed',
        'data_sha256': {name: sha256(raw / name) for name in ['vle.csv', 'courses.csv', 'studentVle.csv']}}), encoding='utf-8')
    baseline = sha256(model / 'predictions.npz')
    run_recommendations(source, output, [14], [42], raw=raw, bootstrap=10, threads=1, methods=['hybrid_risk'])
    assert json.loads((output / 'manifest.json').read_text(encoding='utf-8'))['status'] == 'completed'
    summary = pd.read_csv(output / 'ranking_summary.csv')
    assert {'precision', 'recall', 'ndcg', 'map', 'mrr'} <= set(summary)
    assert set(summary.method) == {'hybrid_risk'}
    assert not list(output.glob('day_*/seed_*/learned*'))
    config = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    assert config['primary_contrast'] is None
    weights = pd.read_csv(output / 'day_14/seed_42/hybrid_weights.csv')
    np.testing.assert_allclose(weights[['preference_weight', 'calendar_weight', 'engagement_weight']].sum(axis=1), 1.)
    endpoints = pd.read_csv(output / 'course_outcome_metrics.csv')
    assert set(endpoints.endpoint) == {'risk', 'success'}
    assert {'pr_auc', 'precision', 'recall', 'f1', 'brier'} <= set(endpoints)
    assert sha256(model / 'predictions.npz') == baseline
