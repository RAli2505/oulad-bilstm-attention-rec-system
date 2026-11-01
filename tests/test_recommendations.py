"""Behavioral contracts for the independent OULAD recommendation layer."""
import json

import numpy as np
import pandas as pd
import pytest


def catalog():
    return pd.DataFrame([
        dict(id_site=i, code_module='AAA', code_presentation='2013J',
             activity_type=kind, week_from=0, week_to=12)
        for i, kind in [(1, 'oucontent'), (2, 'forumng'), (3, 'oucontent'), (4, 'forumng')]
    ])


def history(rows=()):
    return pd.DataFrame([
        dict(id_student=9, code_module='AAA', code_presentation='2013J',
             id_site=site, date=day, sum_click=clicks)
        for site, day, clicks in rows
    ], columns=['id_student', 'code_module', 'code_presentation', 'id_site', 'date', 'sum_click'])


KEY = (9, 'AAA', '2013J')


def preference(events, day=28, resources=None):
    from recommendations.preferences import PreferenceModel
    return PreferenceModel().profile(events, catalog() if resources is None else resources, KEY, day)


def agent(resources=None, events=None):
    from recommendations.recommendation_agent import RecommendationAgent
    return RecommendationAgent(catalog() if resources is None else resources,
                               history() if events is None else events)


def test_cold_start_is_neutral_and_serializable():
    p = preference(history())
    assert p['preferences'] == {'forumng': .5, 'oucontent': .5}
    assert p['confidence'] == 0
    assert p['n_resource_days'] == 0
    json.dumps(p, allow_nan=False)


def test_recent_interest_outweighs_old_interest_and_ages_toward_neutral():
    events = history([(1, 0, 10), (2, 26, 10)])
    p = preference(events)
    assert p['preferences']['forumng'] > p['preferences']['oucontent']
    assert p['preference_shift']['forumng'] > 0
    later = preference(events, 140)
    assert abs(later['preferences']['forumng'] - .5) < abs(p['preferences']['forumng'] - .5)
    assert later['confidence'] < p['confidence']


def test_future_and_other_enrollments_do_not_affect_profile():
    base = history([(1, 5, 4)])
    extra = history([(2, 28, 999), (2, 40, 999), (2, 4, 999), (2, 4, 999)])
    extra.loc[2, 'id_student'] = 10
    extra.loc[3, 'code_presentation'] = '2014J'
    assert preference(pd.concat([base, extra])) == preference(base)


def test_click_saturation_and_daily_aggregation_are_consistent():
    a = preference(history([(1, 5, 100000)]))
    b = preference(history([(1, 5, 20)]))
    c = preference(history([(1, 5, 10), (1, 5, 10)]))
    assert a == b == c


def test_opportunity_correction_does_not_reward_larger_type_catalog():
    resources = catalog().iloc[:3]
    p = preference(history([(1, 5, 5), (3, 5, 5), (2, 5, 5)]), resources=resources)
    assert p['preferences']['forumng'] == pytest.approx(.5)


def test_history_order_does_not_change_profile():
    events = history([(1, 5, 3), (2, 25, 4), (1, 8, 2)])
    assert preference(events) == preference(events.iloc[::-1])


@pytest.mark.parametrize('clicks', [-1, np.nan, np.inf])
def test_invalid_observed_clicks_are_rejected(clicks):
    with pytest.raises(ValueError):
        preference(history([(1, 5, clicks)]))


def test_future_catalog_does_not_change_preference_snapshot():
    resources = catalog()
    future = resources.iloc[[0]].copy()
    future['id_site'] = 99
    future['week_from'] = 20
    assert preference(history([(1, 5, 2)]), resources=pd.concat([resources, future])) == preference(history([(1, 5, 2)]))


def test_recommendations_filter_future_other_course_and_recently_seen():
    resources = catalog()
    resources.loc[2, 'week_from'] = 10
    resources.loc[3, 'code_module'] = 'BBB'
    result = agent(resources, history([(1, 27, 2)])).recommend(KEY, 28, .8, n=10)
    assert [r['item_id'] for r in result['recommendations']] == [2]
    assert result['forecast_probability'] == .8


def test_explicit_availability_overrides_planned_week_and_expiry_filters():
    resources = catalog()
    resources['available_from_day'] = [0, 40, 0, 0]
    resources['available_until_day'] = [np.nan, np.nan, 20, np.nan]
    resources.loc[0, 'week_from'] = 10
    result = agent(resources).recommend(KEY, 28, .2, n=10)
    assert {r['item_id'] for r in result['recommendations']} == {1, 4}


def test_missing_schedule_and_empty_catalog_are_supported():
    resources = catalog().drop(columns=['week_from', 'week_to'])
    result = agent(resources).recommend(KEY, 28, .2)
    assert result['recommendations']
    json.dumps(result, allow_nan=False)
    assert agent(resources.iloc[:0]).recommend(KEY, 28, .2)['recommendations'] == []


def test_future_events_do_not_change_recommendations():
    before = agent(events=history([(1, 5, 4)])).recommend(KEY, 28, .5)
    after = agent(events=history([(1, 5, 4), (2, 28, 999)])).recommend(KEY, 28, .5)
    assert before == after


def test_preference_updates_affect_ranking_without_changing_forecast():
    events = history([(2, day, 10) for day in range(20, 28)])
    result = agent(events=events).recommend(KEY, 28, .4, allow_review=True)
    assert result['recommendations'][0]['activity_type'] == 'forumng'
    assert result['profile']['preferences']['forumng'] > .5
    assert result['forecast_probability'] == .4


def test_risk_changes_calendar_tradeoff_and_diversity_is_exposed():
    resources = catalog()
    resources.loc[0, ['week_from', 'week_to']] = [4, 4]
    resources.loc[1:, ['week_from', 'week_to']] = [0, 0]
    a = agent(resources)
    low = a.recommend(KEY, 28, 0.)
    high = a.recommend(KEY, 28, 1.)
    assert high['recommendations'][0]['item_id'] == 1
    assert high['weights']['calendar'] > low['weights']['calendar']
    assert len({r['activity_type'] for r in high['recommendations'][:2]}) == 2


def test_inputs_are_unchanged_and_outputs_are_deterministic():
    resources, events = catalog(), history([(1, 5, 3)])
    original_r, original_e = resources.copy(deep=True), events.copy(deep=True)
    a = agent(resources, events)
    first = a.recommend(KEY, 28, .8724)
    assert first == a.recommend(KEY, 28, .8724)
    pd.testing.assert_frame_equal(resources, original_r)
    pd.testing.assert_frame_equal(events, original_e)
    first['profile']['preferences']['forumng'] = -10
    assert a.recommend(KEY, 28, .8724)['profile']['preferences']['forumng'] >= 0


@pytest.mark.parametrize('probability', [-.1, 1.1, np.nan, np.inf])
def test_invalid_risk_rejected(probability):
    with pytest.raises(ValueError):
        agent().recommend(KEY, 28, probability)


@pytest.mark.parametrize('day', [-1, np.nan, np.inf])
def test_invalid_cutoff_rejected(day):
    with pytest.raises(ValueError):
        agent().recommend(KEY, day, .5)


def test_duplicate_catalog_identity_is_rejected():
    with pytest.raises(ValueError):
        agent(pd.concat([catalog(), catalog().iloc[[0]]])).recommend(KEY, 28, .5)


def test_zero_clicks_do_not_mark_resource_seen():
    result = agent(events=history([(1, 27, 0)])).recommend(KEY, 28, .5)
    assert 1 in {r['item_id'] for r in result['recommendations']}
    assert result['profile']['n_resource_days'] == 0


def test_expired_material_keeps_preference_evidence_but_is_not_recommended():
    resources = catalog()
    resources['available_until_day'] = np.nan
    resources.loc[1, 'available_until_day'] = 20
    events = history([(2, 10, 10)])
    p = preference(events, resources=resources)
    assert p['n_resource_days'] == 1
    assert p['preferences']['forumng'] > .5
    result = agent(resources, events).recommend(KEY, 28, .5)
    assert 2 not in {r['item_id'] for r in result['recommendations']}


def test_actual_past_access_is_not_erased_by_future_planned_week():
    resources = catalog()
    resources.loc[1, 'week_from'] = 10
    p = preference(history([(2, 10, 10)]), resources=resources)
    assert p['n_resource_days'] == 1
    assert p['preferences']['forumng'] > .5


def test_new_history_adapts_same_agent_without_double_counting():
    a = agent()
    before = a.recommend(KEY, 28, .4)
    new = history([(2, day, 10) for day in range(20, 28)])
    a.update_history(new)
    after = a.recommend(KEY, 28, .4, allow_review=True)
    a.update_history(new)
    assert a.recommend(KEY, 28, .4, allow_review=True) == after
    assert after['profile']['preferences']['forumng'] > before['profile']['preferences']['forumng']
    assert after['forecast_probability'] == before['forecast_probability']
    assert after['recommendations'][0]['activity_type'] == 'forumng'


def prediction_table():
    return pd.DataFrame([dict(id_student=9, code_module='AAA', code_presentation='2013J',
                             ensemble_lstm_attention_lgbm_weighted_probability=.8724, target=1)])


def test_forecast_adapter_selects_one_identity_and_preserves_probability(tmp_path):
    from recommendations.cli import read_forecast
    path = tmp_path / 'predictions.csv'
    frame = prediction_table()
    other = frame.copy()
    other['id_student'] = 100
    other['ensemble_lstm_attention_lgbm_weighted_probability'] = .1
    pd.concat([frame, other]).to_csv(path, index=False)
    assert read_forecast(path, KEY, 112, prediction_day=112) == .8724
    with pytest.raises(ValueError):
        read_forecast(path, KEY, 56, prediction_day=112)
    with pytest.raises(ValueError):
        read_forecast(path, (90, 'AAA', '2013J'), 112, prediction_day=112)


def test_ambiguous_forecast_is_rejected(tmp_path):
    from recommendations.cli import read_forecast
    path = tmp_path / 'predictions.csv'
    frame = prediction_table()
    pd.concat([frame, frame]).to_csv(path, index=False)
    with pytest.raises(ValueError):
        read_forecast(path, KEY, 112, prediction_day=112)


def test_cli_creates_serializable_result_without_overwriting_inputs(tmp_path):
    from recommendations.cli import main
    raw = tmp_path / 'raw'
    raw.mkdir()
    catalog().to_csv(raw / 'vle.csv', index=False)
    history([(1, 5, 4), (2, 200, 99)]).to_csv(raw / 'studentVle.csv', index=False)
    predictions = tmp_path / 'predictions.csv'
    prediction_table().to_csv(predictions, index=False)
    output = tmp_path / 'recommendations.json'
    args = ['--data-dir', str(raw), '--predictions', str(predictions),
            '--student', '9', '--module', 'AAA', '--presentation', '2013J',
            '--day', '112', '--prediction-day', '112', '--output', str(output)]
    original = predictions.read_bytes()
    assert main(args) == 0
    result = json.loads(output.read_text(encoding='utf-8'))
    assert result['forecast_probability'] == .8724
    assert result['profile']['n_resource_days'] == 1
    assert result['forecast_source']['prediction_day'] == 112
    with pytest.raises(FileExistsError):
        main(args)
    assert predictions.read_bytes() == original
