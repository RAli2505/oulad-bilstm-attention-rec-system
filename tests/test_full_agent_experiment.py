"""Small CPU-only tests; no original risk training and no real OULAD run."""
import json
from pathlib import Path
import shutil
import uuid

import numpy as np
import pandas as pd
import pytest

from recommendations import full_agent_experiment as experiment


@pytest.fixture
def scratch():
    # Local unique directory avoids restricted Windows pytest temp-root ACLs.
    directory = Path(__file__).resolve().parents[1] / 'test-artifacts' / ('full-agent-' + uuid.uuid4().hex)
    directory.mkdir(parents=True)
    yield directory
    shutil.rmtree(directory)


def test_baseline_uses_full_truth_but_recent_mask():
    data = pd.DataFrame({'item': [1, 2, 3], 'kind': ['a', 'a', 'b'],
                         'preference': [.9, .5, .4], 'calendar': [1., 1., 1.], 'engagement': [0., 1., 1.]})
    history = pd.DataFrame({'id_site': [1, 2, 3], 'date': [9, 2, 11], 'sum_click': [1, 1, 1]})
    ranking = experiment.replay_baseline(data, None, history, 10, .8)
    assert 1 not in ranking and set(ranking) == {2, 3}
    assert experiment.ranking_metrics(ranking, {1, 2}, 10)['recall'] == .5


def test_contextual_probability_head_does_not_use_held_labels():
    rng = np.random.default_rng(8)
    x = rng.normal(size=(80, 3))
    score = 1 / (1 + np.exp(-x[:, 0]))
    labels = (x[:, 1] > 0).astype(int)
    roles = np.repeat(['fit', 'selection', 'calibration', 'held'], 20)
    a, pa = experiment.fit_probability_head(x, score, labels, roles, 42)
    changed = labels.copy()
    changed[roles == 'held'] = 1 - changed[roles == 'held']
    b, pb = experiment.fit_probability_head(x, score, changed, roles, 42)
    np.testing.assert_array_equal(pa, pb)
    assert a[-1].coef_.shape == (1, 4)
    assert a[-1].coef_[0, 1] > 0


def test_fusion_rejects_selection_recall_regression(scratch):
    from types import SimpleNamespace
    # A deliberately worse learned ranker must remain rejected, independently
    # of its excellent fit fit/proxy classification accuracy.
    class Store:
        x = np.array([[0.], [1.], [0.], [1.], [0.], [1.], [0.], [1.]])
        fallback = np.full(8, .5)
        templates = {0: {'feature_names': ['test_feature']}}
        queries = pd.DataFrame([{'query': 0, 'role': 'selection', 'start': 2, 'stop': 4}])
        data = pd.DataFrame({'item': [1, 2] * 4, 'label': [0, 1] * 4})
        baseline = {0: [2, 1]}

        def selection(self, agent, scores=None):
            return {'ndcg': 0., 'recall': 0.}

    agent = SimpleNamespace(candidate_model=None)
    labels = np.array([0, 1] * 4)
    roles = np.repeat(['fit', 'selection', 'calibration', 'held'], 2)
    scores = experiment.fit_fusion(agent, Store(), roles, labels, 42, scratch)
    assert agent.candidate_model is None
    np.testing.assert_array_equal(scores, Store.fallback)
    report = json.loads((scratch / 'fusion_search_and_weights.json').read_text())
    assert report['fallback'] and report['selected_C'] is None
    assert not any(trial['accepted'] for trial in report['trials'])


def test_source_provenance_is_checked(scratch):
    source, raw, output = scratch / 'risk', scratch / 'raw', scratch / 'output'
    source.mkdir(); raw.mkdir()
    for name in ('vle.csv', 'courses.csv', 'studentVle.csv'):
        (raw / name).write_text('a\n1\n', encoding='utf-8')
    manifest = {'status': 'completed', 'config': {'split_scheme': 'holdout', 'evaluate_test': False,
                                               'seeds': [42], 'windows': [7]},
                'data_sha256': {name: experiment.sha256(raw / name) for name in ('vle.csv', 'courses.csv', 'studentVle.csv')}}
    experiment.write_json(source / 'run_manifest.json', manifest)
    assert len(experiment.validate_source(source, output, [7], [42], raw)) == 3
    with pytest.raises(ValueError, match='separate'):
        experiment.validate_source(source, source / 'nested', [7], [42], raw)
    (raw / 'vle.csv').write_text('changed', encoding='utf-8')
    with pytest.raises(ValueError, match='differs'):
        experiment.validate_source(source, output, [7], [42], raw)


def test_compact_selector_exact_facade_parity_for_ties_and_exposure():
    from recommendations.oulad_agent import OULADRecommendationAgent
    agent = OULADRecommendationAgent(pd.DataFrame(), pd.DataFrame(
        columns=['id_student', 'code_module', 'code_presentation', 'id_site', 'date', 'sum_click']))
    rng = np.random.default_rng(82)
    for trial in range(24):
        n = 31
        items = rng.permutation(np.arange(100, 100 + n))
        kinds = rng.choice(['forum', 'resource', 'quiz'], n)
        # Include exact ties, neutral SAS, and repeated exposure at varied days.
        scores = rng.choice([0., .2, .4, 1.], n) if trial % 3 else np.zeros(n)
        affinity = rng.choice([0., .5, 1.], n) if trial % 4 else np.zeros(n)
        days = rng.integers(0, 20, n)
        candidates = [{'item_id': int(i), 'activity_type': kind, 'score': float(score),
                       'sasrec_affinity': float(a), 'history_resource_days': int(d)}
                      for i, kind, score, a, d in zip(items, kinds, scores, affinity, days)]
        pool = sorted(candidates, key=lambda c: (-c['score'], c['item_id']))
        agent.mmr_lambda = [0., .8, 1.][trial % 3]
        agent.sasrec_weight = [0., .05][trial % 2]
        selected, _ = agent._select_sequence(agent._mmr_rerank(pool, 20), pool)
        expected = [c['item_id'] for c in selected]
        actual = experiment.compact_rank(items, kinds, scores, affinity, days, 20,
                                          agent.mmr_lambda, agent.sasrec_weight)
        assert actual == expected
        assert len(actual) == len(set(actual))


def test_tiny_full_runner_save_load(scratch, monkeypatch):
    from recommendations.oulad_agent import OULADRecommendationAgent
    raw, source, output = scratch / 'raw', scratch / 'risk', scratch / 'result'
    raw.mkdir(); source.mkdir()
    resources = pd.DataFrame([{'id_site': i, 'code_module': 'AAA', 'code_presentation': '2013J',
                               'activity_type': 'forum' if i % 2 else 'resource', 'week_from': 0, 'week_to': 10}
                              for i in range(1, 7)])
    resources.to_csv(raw / 'vle.csv', index=False)
    pd.DataFrame([{'code_module': 'AAA', 'code_presentation': '2013J', 'module_presentation_length': 100}]).to_csv(raw / 'courses.csv', index=False)
    events = pd.DataFrame([{'id_student': student, 'code_module': 'AAA', 'code_presentation': '2013J',
                            'id_site': item, 'date': day, 'sum_click': 2}
                           for student in range(1, 9) for day, item in [(0, 1), (1, 2), (3, 3), (6, 1), (8, 4 if student % 2 else 5)]])
    events.to_csv(raw / 'studentVle.csv', index=False)
    frame = pd.DataFrame([{'id_student': i, 'code_module': 'AAA', 'code_presentation': '2013J', 'target': i % 2} for i in range(1, 9)])
    indices = {role: np.arange(i * 2, i * 2 + 2) for i, role in enumerate(experiment.ROLES)}
    manifest = {'status': 'completed', 'config': {'split_scheme': 'holdout', 'evaluate_test': False,
                                               'seeds': [42], 'windows': [7]},
                'data_sha256': {name: experiment.sha256(raw / name) for name in ('vle.csv', 'courses.csv', 'studentVle.csv')}}
    experiment.write_json(source / 'run_manifest.json', manifest)
    original_digest = experiment.sha256(source / 'run_manifest.json')
    monkeypatch.setattr(experiment, 'source_splits', lambda path: (frame, indices))
    monkeypatch.setattr(experiment, 'reconstructed_risk', lambda *args: (
        {role: np.array([.2, .8]) for role in experiment.ROLES}, {}, {'frozen': True}))
    compact_class = experiment.CompactSnapshots

    class CheckedCompact(compact_class):
        def __init__(self, agent, data, queries, all_events, day, risk, **kwargs):
            super().__init__(agent, data, queries, all_events, day, risk, **kwargs)
            for q in queries.itertuples(index=False):
                key = (q.id_student, q.code_module, q.code_presentation)
                history = all_events.loc[(all_events.id_student == q.id_student) & (all_events.date < day)]
                original = agent.build_snapshot(key, day, float(risk[q.role][q.forecast_position]),
                    prepared={'candidates': data.iloc[q.start:q.stop], 'query': q._asdict()}, history=history)
                restored = self.snapshot(q)
                assert [c['history_resource_days'] for c in original['candidates']] == [
                    c['history_resource_days'] for c in restored['candidates']]
                np.testing.assert_array_equal(original['features'], restored['features'])
                assert experiment._rank(agent, original, 5) == experiment._rank(agent, restored, 5)
                assert self.rank(q, agent, 5) == experiment._rank(agent, original, 5)

    monkeypatch.setattr(experiment, 'CompactSnapshots', CheckedCompact)
    experiment.run_full_agent(source, output, [7], [42], raw=raw, epochs=1, sasrec_dimension=16,
                              threads=1, bootstrap=5, device='cpu')
    assert json.loads((output / 'manifest.json').read_text())['status'] == 'completed'
    assert experiment.sha256(source / 'run_manifest.json') == original_digest
    bundle = output / 'day_7/seed_42/agent_bundle'
    loaded = OULADRecommendationAgent.load(bundle, resources, events, device='cpu')
    snapshot = loaded.build_snapshot((8, 'AAA', '2013J'), 7, .8)
    result = loaded.rank_snapshot(snapshot, n=3)
    assert result['forecast_probability'] == .8
    assert len(result['recommendations']) == 3
    search = json.loads((output / 'day_7/seed_42/fusion_search_and_weights.json').read_text())
    assert len(search['trials']) == 3
    metrics = pd.read_csv(output / 'ranking_summary.csv')
    assert set(metrics.role) == set(experiment.ROLES)
    assert set(metrics.k) == {5, 10, 20}
    checkpoint = output / 'day_7/seed_42/checkpoint.json'
    assert checkpoint.is_file()
    model_hash = experiment.sha256(bundle / 'sequential.pt')
    def forbidden_fit(*args, **kwargs):
        raise AssertionError('A completed checkpoint must not retrain')
    monkeypatch.setattr('recommendations.sequential.DailySASRecService.fit', forbidden_fit)
    experiment.run_full_agent(source, output, [7], [42], raw=raw, epochs=1, sasrec_dimension=16,
                              threads=1, bootstrap=5, device='cpu', resume=True)
    assert experiment.sha256(bundle / 'sequential.pt') == model_hash
    from recommendations.checkpoints import complete_job
    assert complete_job(bundle.parent, 7, 42)
    (bundle.parent / 'ranking_summary.csv').write_text('corrupted', encoding='utf-8')
    with pytest.raises(ValueError, match='checksum'):
        complete_job(bundle.parent, 7, 42)
