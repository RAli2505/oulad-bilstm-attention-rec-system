import copy
import math

import numpy as np
import pandas as pd
import pytest
import torch

from models.pred_oulad import BiLSTMAttentionNet
from risk_experiments.features import build_window, FoldTransform, temporal_split
from risk_experiments.metrics import bootstrap_auc, concordance, calibration_stats
from risk_experiments.networks import MODELS, NEURAL, make_model, breslow_loss
from risk_experiments.pipeline import synthetic_data, grouped_folds, permutation_p
from risk_experiments.training import batch_indices


def test_breslow_matches_direct_risk_sets_with_ties():
    risk = torch.tensor([.2, -.4, .7, 1.2], requires_grad=True)
    time = torch.tensor([2., 2., 1., 3.])
    event = torch.tensor([1., 1., 1., 0.])
    direct = torch.stack([torch.logsumexp(risk[time >= time[i]], 0) - risk[i]
                          for i in range(4) if event[i]]).mean()
    loss = breslow_loss(risk, time, event)
    assert torch.allclose(loss, direct)
    a = torch.autograd.grad(loss, risk, retain_graph=True)[0]
    b = torch.autograd.grad(direct, risk)[0]
    assert torch.allclose(a, b)
    perm = torch.tensor([1, 0, 2, 3])
    assert torch.allclose(loss, breslow_loss(risk[perm], time[perm], event[perm]))
    assert breslow_loss(risk, time, torch.zeros(4)).item() == 0


def test_future_events_and_scores_cannot_change_observed_features():
    target, daily, assessment = synthetic_data()
    original = build_window(target, daily, assessment, 28)
    altered = daily.copy()
    altered.loc[altered.date >= 28, 'sum_click'] = 999999
    altered_a = assessment.copy()
    altered_a.loc[altered_a.date_submitted >= 28, 'score'] = 0
    result = build_window(target, altered, altered_a, 28)
    assert np.array_equal(original.counts, result.counts)
    pd.testing.assert_frame_equal(original.frame, result.frame)
    assert original.audit['zero_history_enrollments'] > 0


def test_inactive_or_unregistered_students_are_excluded_at_landmark():
    target, daily, assessment = synthetic_data()
    target.loc[0, 'date_unregistration'] = 28
    target.loc[1, 'date_registration'] = 35
    w = build_window(target, daily, assessment, 28)
    assert len(w.frame) == len(target) - 2
    assert (w.time > 0).all()


def test_preprocessing_ignores_held_features_and_unknown_channels():
    target, daily, assessment = synthetic_data()
    w = build_window(target, daily, assessment, 28)
    fit_idx = np.arange(100)
    first = FoldTransform().fit(w, fit_idx)
    modified = copy.deepcopy(w)
    modified.frame.loc[100:, 'studied_credits'] = 999999
    modified.frame.loc[100:, 'region'] = 'new-region'
    modified.counts[100:] = 99999
    second = FoldTransform().fit(modified, fit_idx)
    assert np.array_equal(first.mean, second.mean)
    assert np.array_equal(first.std, second.std)
    assert first.channel_names == second.channel_names
    for k in ('s', 'x', 'flat', 'weekly'):
        assert np.allclose(first.transform(w, fit_idx)[k], second.transform(modified, fit_idx)[k])


def test_temporal_and_inner_outer_splits_are_student_disjoint():
    w = build_window(*synthetic_data(), 28)
    dev, test, removed = temporal_split(w)
    assert not set(w.groups[dev]) & set(w.groups[test])
    coverage = np.zeros(len(dev))
    for fit, held in grouped_folds(w.y[dev], w.groups[dev], 42):
        assert not set(w.groups[dev][fit]) & set(w.groups[dev][held])
        inner, val = next(grouped_folds(w.y[dev][fit], w.groups[dev][fit], 43))
        assert not set(w.groups[dev][fit][inner]) & set(w.groups[dev][fit][val])
        coverage[held] += 1
    assert (coverage == 1).all()


@pytest.mark.parametrize('name', NEURAL)
def test_all_neural_architectures_train(name):
    model = make_model(name, 3, 10, 14)
    output, risk = model(torch.randn(4, 14, 3), torch.randn(4, 10), torch.randn(4, 2))
    assert output.shape == (4,) and torch.isfinite(output).all()
    output.sum().backward()


def test_main_architecture_matches_historical_model_and_no_cox_initialization():
    torch.manual_seed(42)
    main = make_model('proposed', 3, 10, 14).eval()
    torch.manual_seed(42)
    ablation = make_model('no_cox', 3, 10, 14).eval()
    assert all(torch.equal(v, ablation.state_dict()[k]) for k, v in main.state_dict().items())
    old = BiLSTMAttentionNet(3, 10).eval()
    old.load_state_dict(main.state_dict())
    x, s = torch.randn(4, 14, 3), torch.randn(4, 10)
    assert torch.allclose(main(x, s)[0], old(x, torch.full((4,), 14), s)[0], atol=1e-6)


def test_batches_keep_every_student_without_batchnorm_singleton():
    batches = batch_indices(torch.arange(257), 256)
    assert all(len(b) > 1 for b in batches)
    assert torch.equal(torch.cat(batches), torch.arange(257))


def test_paired_inference_is_null_for_identical_predictions():
    y = np.tile([0, 1], 20)
    groups = np.repeat(np.arange(20), 2)
    p = np.linspace(.1, .9, len(y))
    assert bootstrap_auc(y, p, groups, 20, comparator=p) == [0., 0.]
    assert permutation_p(y, p, p, groups, 20) == 1
    assert concordance(np.array([3, 2, 1]), np.array([1, 2, 3]), np.ones(3)) == 1
    assert len(MODELS) == 14


def test_primary_excludes_scores_and_proxy_is_explicit_sensitivity_analysis():
    target, daily, assessment = synthetic_data()
    strict = build_window(target, daily, assessment, 28)
    proxy = build_window(target, daily, assessment, 28, score_policy='proxy')
    assert strict.frame.mean_score.isna().all()
    assert (strict.frame.score_observed == 0).all()
    assert proxy.frame.mean_score.notna().any()


def test_calibration_diagnostics_return_counts_and_finite_measures():
    y = np.tile([0, 1], 30)
    p = np.linspace(.01, .99, len(y))
    measures, bins = calibration_stats(y, p)
    assert sum(b['n'] for b in bins) == len(y)
    assert np.isfinite(list(measures.values())).all()


def test_development_only_preparation_never_includes_2014(tmp_path):
    from risk_experiments.pipeline import run_window
    run_window(*synthetic_data(), {'days': 28, 'score_lag': 7, 'score_policy': 'omit',
               'split_seed': 42, 'evaluate_test': False}, tmp_path, 'proof', prepare_only=True)
    manifest = pd.read_csv(tmp_path / 'cohort_and_folds.csv')
    assert set(manifest.code_presentation) == {'2013J'}
    assert not any('test_2014' in str(p) for p in tmp_path.rglob('*'))


def test_no_external_predictions_when_test_is_disabled(tmp_path):
    from risk_experiments.training import fit_predict
    w = build_window(*synthetic_data(), 28)
    transform = FoldTransform().fit(w, np.arange(140))
    fit, val, held = [transform.transform(w, ids) for ids in (np.arange(140), np.arange(140, 200), np.arange(200, 280))]
    prediction, _ = fit_predict('lr', fit, val, held, None, tmp_path, 42,
                                {'device': 'cpu', 'threads': 1, 'mlp_iterations': 200})
    assert not any(k.startswith('test_') for k in prediction)


def test_appending_seeds_preserves_job_signature_but_not_other_changes():
    from run_risk_article import experiment_signature
    one = {'config': {'seeds': [42], 'epochs': 100}, 'source': 'same'}
    five = {'config': {'seeds': [42, 13, 7, 23, 101], 'epochs': 100}, 'source': 'same'}
    changed = {'config': {'seeds': [42], 'epochs': 50}, 'source': 'same'}
    assert experiment_signature(one) == experiment_signature(five)
    assert experiment_signature(one) != experiment_signature(changed)
