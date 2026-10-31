import copy
import json

import numpy as np
import pandas as pd
import pytest

from risk_experiments.enhanced_features import enhanced_window, EnhancedTransform, EXTRA_NUM
from risk_experiments.features import FoldTransform
from risk_experiments.pipeline import synthetic_data
from risk_experiments.tuning import tune_and_predict, fit_ensemble, ENSEMBLE_MEMBERS, ENSEMBLES, ENSEMBLE_GROUPS


@pytest.mark.parametrize('day', [7, 14, 28, 56, 84, 112, 140, 168])
def test_enriched_features_cover_all_requested_landmarks(day):
    w = enhanced_window(*synthetic_data(), day)
    assert np.isfinite(w.frame[EXTRA_NUM].to_numpy()).all()
    assert w.counts.shape[1] == day
    assert w.frame.mean_score.isna().all()


def test_future_events_cannot_change_enriched_features():
    target, daily, assessment = synthetic_data()
    before = enhanced_window(target, daily, assessment, 28)
    daily.loc[daily.date >= 28, 'sum_click'] = 999999
    assessment.loc[assessment.date_submitted >= 28, 'date_submitted'] = 80
    assessment['score'] = 100
    after = enhanced_window(target, daily, assessment, 28)
    pd.testing.assert_frame_equal(before.frame[EXTRA_NUM], after.frame[EXTRA_NUM])
    assert np.array_equal(before.counts, after.counts)


def test_future_submission_stays_missing_before_its_timestamp():
    target, daily, _ = synthetic_data()
    f = target.iloc[[0]]
    record = dict(id_student=0, code_module='BBB', code_presentation='2013J', id_assessment=999,
                  date=10, date_submitted=20, is_banked=0, score=100)
    assessment = pd.DataFrame([record])
    assessment.attrs['assessment_catalog'] = [{k: record[k] for k in ['code_module', 'code_presentation', 'id_assessment', 'date']}]
    early = enhanced_window(f, daily, assessment, 14)
    later = enhanced_window(f, daily, assessment, 28)
    assert early.frame.overdue_count.iloc[0] == 1
    assert early.frame.max_overdue_days.iloc[0] == 4
    assert later.frame.overdue_count.iloc[0] == 0
    assert later.frame.late_submissions.iloc[0] == 1
    assert later.frame.mean_submission_delay.iloc[0] == 10


def test_enriched_transform_fits_only_training_rows():
    w = enhanced_window(*synthetic_data(), 28)
    first = EnhancedTransform().fit(w, np.arange(140))
    modified = copy.deepcopy(w)
    modified.frame.loc[140:, 'days_since_last_activity'] = 99999
    modified.frame.loc[140:, 'code_module'] = 'UNKNOWN'
    second = EnhancedTransform().fit(modified, np.arange(140))
    for key in ('x', 's', 'flat', 'weekly'):
        assert np.allclose(first.transform(w, np.arange(140))[key], second.transform(modified, np.arange(140))[key])


def test_outer_labels_and_calibration_labels_cannot_select_hyperparameters(tmp_path):
    w = enhanced_window(*synthetic_data(), 28)
    transform = EnhancedTransform().fit(w, np.arange(120))
    fit, selection, calibration, held = [transform.transform(w, ids) for ids in
        (np.arange(120), np.arange(120, 160), np.arange(160, 200), np.arange(200, 240))]
    config = {'device': 'cpu', 'threads': 1, 'synthetic': True, 'epochs': 1,
              'screen_epochs': 1, 'tuning_trials': 2, 'mlp_iterations': 2}
    _, first = tune_and_predict('lr', fit, selection, calibration, held, tmp_path / 'first', 42, config)
    calibration['y'], held['y'] = 1 - calibration['y'], 1 - held['y']
    _, second = tune_and_predict('lr', fit, selection, calibration, held, tmp_path / 'second', 42, config)
    assert first['selected_configuration'] == second['selected_configuration']
    assert first['calibration']['fit_partition'] == 'independent_inner_calibration'


def test_ensemble_weights_do_not_use_outer_labels(tmp_path):
    rng = np.random.default_rng(42)
    predictions = {name: {'val_score': rng.normal(size=40), 'calibration_score': rng.normal(size=40),
                           'held_score': rng.normal(size=40)} for name in ENSEMBLE_MEMBERS}
    labels = np.tile([0, 1], 20)
    p1, r1 = fit_ensemble('ensemble_inner_weighted', predictions, labels, labels, labels, tmp_path / 'a')
    p2, r2 = fit_ensemble('ensemble_inner_weighted', predictions, labels, labels, 1 - labels, tmp_path / 'b')
    assert np.allclose(r1['weights'], r2['weights'])
    assert np.allclose(p1['held_probability'], p2['held_probability'])
    assert all(w >= 0 for w in r1['weights'])
    assert abs(sum(r1['weights']) - 1) < 1e-8
    assert len(ENSEMBLE_MEMBERS) == 3


@pytest.mark.parametrize('name', ENSEMBLES)
def test_both_requested_ensemble_families_are_supported(name, tmp_path):
    rng = np.random.default_rng(7)
    members = ENSEMBLE_GROUPS[name]
    predictions = {m: {key: rng.normal(size=40) for key in ['val_score', 'calibration_score', 'held_score']}
                   for m in members}
    y = np.tile([0, 1], 20)
    pred, report = fit_ensemble(name, predictions, y, y, y, tmp_path)
    assert report['members'] == members
    assert report['seconds'] > 0
    assert len(pred['held_probability']) == 40
    assert np.isfinite(pred['held_probability']).all()


def test_single_holdout_has_no_learners_shared_between_fit_selection_calibration_and_validation(tmp_path):
    from risk_experiments.pipeline import run_window
    config = {'days': 28, 'score_lag': 7, 'score_policy': 'omit', 'split_seed': 42,
              'improved': True, 'feature_profile': 'rich', 'split_scheme': 'holdout'}
    run_window(*synthetic_data(), config, tmp_path, 'proof', prepare_only=True)
    manifest = pd.read_csv(tmp_path / 'cohort_and_folds.csv')
    assert set(manifest.role) == {'train_pool_2013', 'validation_2013'}
    groups = manifest.groupby('role').id_student.apply(set)
    assert not groups['train_pool_2013'] & groups['validation_2013']
    assert (manifest.oof_fold == 1).sum() < len(manifest) / 3
    assert (manifest.oof_fold == 1).sum() > len(manifest) / 4
    assert manifest.loc[manifest.role == 'train_pool_2013', 'oof_fold'].eq(-1).all()


def test_learner_holdout_assignment_stays_fixed_across_landmarks():
    from risk_experiments.pipeline import holdout_student_groups
    target, daily, assess = synthetic_data()
    target = target[target.code_presentation == '2013J']
    selected = holdout_student_groups(target, 42)
    assert abs(np.isin(target.id_student, selected).mean() - .3) < .03
    for day in [7, 28, 56, 112, 168]:
        w = enhanced_window(target, daily, assess, day)
        train = w.groups[~np.isin(w.groups, selected)]
        val = w.groups[np.isin(w.groups, selected)]
        assert not set(train) & set(val)
