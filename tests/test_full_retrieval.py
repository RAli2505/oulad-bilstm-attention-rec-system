import io

import joblib
import numpy as np
import pandas as pd
import pytest

from recommendations.retrieval import OULADRetrieval
from recommendations.strategy import StrategyController


def fixture():
    resources = pd.DataFrame([('AAA', '2013J', i, kind, 0, 8) for i, kind in [(1, 'forum'), (2, 'forum'), (3, 'quiz')]], columns=['code_module', 'code_presentation', 'id_site', 'activity_type', 'week_from', 'week_to'])
    clicks = pd.DataFrame([(u, 'AAA', '2013J', i, d, c) for u, i, d, c in [(1, 1, 1, 5), (1, 2, 2, 4), (2, 2, 1, 8), (2, 3, 3, 4)]], columns=['id_student', 'code_module', 'code_presentation', 'id_site', 'date', 'sum_click'])
    return resources, clicks


def test_fit_excludes_other_roles_and_future_and_scopes_history():
    r, h = fixture()
    extra = h.copy(); extra['id_student'] = 999
    future = h.copy(); future['date'] = 56
    a = OULADRetrieval(n_factors=2).fit(r, h, [(1, 'AAA', '2013J'), (2, 'AAA', '2013J')], 56)
    b = OULADRetrieval(n_factors=2).fit(r, pd.concat([h, extra, future]), [(1, 'AAA', '2013J'), (2, 'AAA', '2013J')], 56)
    np.testing.assert_allclose(a.item_factors_, b.item_factors_)
    for key in ('knowledge_based', 'content_based', 'collaborative'):
        np.testing.assert_allclose(a.score((1, 'AAA', '2013J'), h, r, 56)[key], b.score((1, 'AAA', '2013J'), pd.concat([h, extra, future]), r, 56)[key])


def test_new_user_foldin_coldstart_and_serialization():
    r, h = fixture()
    model = OULADRetrieval(n_factors=2).fit(r, h, [(1, 'AAA', '2013J'), (2, 'AAA', '2013J')], 56)
    cold = model.score((3, 'AAA', '2013J'), h, r, 56)
    assert not cold['cf_available']
    assert not cold['collaborative'].any()
    own = h.loc[h.id_student == 1].copy(); own['id_student'] = 3
    warm = model.score((3, 'AAA', '2013J'), own, r, 56)
    assert warm['cf_available'] and np.isfinite(warm['collaborative']).all()
    assert warm['content_based'][0] > warm['content_based'][2]
    buf = io.BytesIO(); joblib.dump(model, buf); buf.seek(0)
    np.testing.assert_allclose(joblib.load(buf).score((3, 'AAA', '2013J'), own, r, 56)['collaborative'], warm['collaborative'])


@pytest.mark.parametrize('policy', ['thompson', 'ucb1', 'epsilon_greedy', 'fixed_kb', 'fixed_round_robin'])
def test_controller_is_deterministic_and_respects_cf(policy):
    a, b = StrategyController(policy=policy), StrategyController(policy=policy)
    assert [a.choose('u', 30) for _ in range(12)] == [b.choose('u', 30) for _ in range(12)]
    assert all(a.choose('u', 30, False) != 'collaborative' for _ in range(20))
    before = a.weights('u')
    a.update_reward('u', 'content_based', 1)
    assert a.weights('u')['content_based'] > before['content_based']
    for reward in [float('nan'), float('inf'), -1, 1.1, True]:
        with pytest.raises(ValueError): a.update_reward('u', 'content_based', reward)


def test_ucb_observations_and_round_robin():
    c = StrategyController(policy='ucb1')
    for strategy in ['knowledge_based', 'content_based', 'collaborative']:
        assert c.choose('u', 20) == strategy
        c.update_reward('u', strategy, 0.5)
    c = StrategyController(policy='fixed_round_robin')
    assert [c.choose('u', 0) for _ in range(4)] == ['knowledge_based', 'content_based'] * 2


def test_composite_identity_unknown_items_and_empty_fit():
    r, h = fixture()
    other = r.copy(); other['code_module'] = 'BBB'
    catalog = pd.concat([r, other], ignore_index=True)
    model = OULADRetrieval(n_factors=2).fit(catalog, h, [(1, 'AAA', '2013J')], 56)
    assert model.item_index_[('AAA', '2013J', 1)] != model.item_index_[('BBB', '2013J', 1)]
    scores = model.score((1, 'AAA', '2013J'), h, catalog, 56)
    assert not scores['collaborative'][3:].any()
    unknown = r.iloc[:1].copy(); unknown['id_site'] = 999
    assert model.score((1, 'AAA', '2013J'), h, unknown, 56)['collaborative'][0] == 0
    model.fit(catalog, h, [], 56)
    assert not model.score((1, 'AAA', '2013J'), h, r, 56)['cf_available']
    assert model.score((1, 'AAA', '2013J'), h, r.iloc[:0], 56)['collaborative'].shape == (0,)


def test_click_order_and_global_rng_are_unchanged():
    r, h = fixture()
    keys = [(1, 'AAA', '2013J'), (2, 'AAA', '2013J')]
    np.random.seed(17)
    expected = np.random.random(3)
    np.random.seed(17)
    a = OULADRetrieval(n_factors=2).fit(r, h, keys, 56)
    b = OULADRetrieval(n_factors=2).fit(r, h.iloc[::-1], keys[::-1], 56)
    np.testing.assert_allclose(a.item_factors_, b.item_factors_)
    StrategyController().choose('u', 20)
    np.testing.assert_allclose(np.random.random(3), expected)


def test_controller_serialization_preserves_next_random_choice():
    controller = StrategyController()
    controller.update_reward((1, 'AAA', '2013J'), 'collaborative', 0.4)
    controller.choose((1, 'AAA', '2013J'), 10)
    buffer = io.BytesIO(); joblib.dump(controller, buffer); buffer.seek(0)
    restored = joblib.load(buffer)
    assert [controller.choose((1, 'AAA', '2013J'), 10) for _ in range(20)] == [restored.choose((1, 'AAA', '2013J'), 10) for _ in range(20)]


@pytest.mark.parametrize('policy', ['thompson', 'ucb1', 'epsilon_greedy', 'fixed_kb', 'fixed_round_robin'])
def test_peek_matches_full_copy_without_mutating_state(policy):
    from copy import deepcopy
    controller = StrategyController(policy=policy)
    for i in range(100):
        controller.update_reward((i, 'AAA', '2013J'), 'collaborative', .4)
    key = (40, 'AAA', '2013J')
    before = deepcopy(controller)
    expected = deepcopy(controller).choose(key, 30, True)
    assert controller.peek(key, 30, True) == expected
    assert controller._state == before._state
    assert controller._steps == before._steps
    assert controller.choose(key, 30, True) == before.choose(key, 30, True)


def test_vectorized_fit_matches_scoped_reference_and_never_scans_per_user(monkeypatch):
    import recommendations.retrieval as retrieval
    from recommendations.recommendation_signals import observable_context
    r, h = fixture()
    duplicate = h.iloc[:1].copy(); duplicate['sum_click'] = 300
    unknown = h.iloc[:1].copy(); unknown['id_site'] = 999
    future = h.iloc[:1].copy(); future['date'] = 56; future['sum_click'] = 'not observable'
    other = h.iloc[:1].copy(); other['id_student'] = 999; other['date'] = 'not observable'
    events = pd.concat([h, duplicate, unknown, future, other], ignore_index=True)
    keys = [(1, 'AAA', '2013J'), (2, 'AAA', '2013J')]
    reference = np.zeros((2, 3))
    for row, key in enumerate(keys):
        _, observed = observable_context(r, events, key, 56)
        for site, clicks in observed.groupby('id_site')['sum_click'].sum().items():
            reference[row, int(site) - 1] = np.log1p(min(clicks, 100))
    original = retrieval.TruncatedSVD.fit
    captured = []
    def capture(self, matrix, *args, **kwargs):
        captured.append(matrix.toarray())
        return original(self, matrix, *args, **kwargs)
    monkeypatch.setattr(retrieval.TruncatedSVD, 'fit', capture)
    def forbidden(*args, **kwargs):
        raise AssertionError('fit must not perform per-enrollment context scans')
    monkeypatch.setattr(retrieval, 'observable_context', forbidden)
    fitted = OULADRetrieval(n_factors=2).fit(r, events, keys, 56)
    np.testing.assert_allclose(captured[0], reference)
    assert fitted.fit_metadata_['elapsed_seconds'] >= 0
    assert fitted.fit_metadata_['fit_enrollments'] == 2
    assert fitted.fit_metadata_['nonzero_user_items'] == 4


def test_metadata_cosine_uses_planned_calendar_and_blocks_earlier_inference():
    r, h = fixture()
    r['activity_type'] = 'forum'
    r.loc[r.id_site == 3, ['week_from', 'week_to']] = [2, 3]
    model = OULADRetrieval(n_factors=2).fit(r, h, [(1, 'AAA', '2013J')], 56)
    scores = model.score((1, 'AAA', '2013J'), h, r, 56)['content_based']
    assert scores[0] > scores[2] > 0
    assert np.all((scores >= 0) & (scores <= 1 + 1e-12))
    with pytest.raises(ValueError, match='fit cutoff'):
        model.score((1, 'AAA', '2013J'), h, r, 55)
    invalid = h.copy(); invalid.loc[0, 'sum_click'] = -1
    with pytest.raises(ValueError, match='negative'):
        OULADRetrieval().fit(r, invalid, [(1, 'AAA', '2013J')], 56)
