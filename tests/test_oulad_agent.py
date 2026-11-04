import numpy as np
import pandas as pd
import pytest

from tests.test_recommendations import catalog, history, KEY


def fitted_agent():
    from recommendations.oulad_agent import OULADRecommendationAgent
    resources = catalog()
    events = pd.concat([history([(1, 1, 4), (2, 2, 6), (1, 8, 3), (2, 12, 4)]),
                        history([(3, 1, 5), (4, 3, 3), (3, 9, 8)]).assign(id_student=10)])
    keys = events[['id_student', 'code_module', 'code_presentation']].drop_duplicates()
    return OULADRecommendationAgent(resources, events, random_seed=42).initialize(keys, 28, train_sequence=False)


def test_full_agent_exposes_three_strategies_and_preserves_forecast():
    agent = fitted_agent()
    result = agent.recommend(KEY, 28, .8724, n=4)
    assert result['forecast_probability'] == .8724
    assert len(result['recommendations']) == 4
    assert len({r['item_id'] for r in result['recommendations']}) == 4
    assert set(result['strategy_weights']) == {'knowledge_based', 'content_based', 'collaborative'}
    assert result['diagnostics']['irt'] == 'unavailable_without_observed_response_metadata'
    assert result['diagnostics']['content_source'] == 'activity_and_calendar_metadata'


def test_feedback_credits_only_actual_shown_item_once():
    agent = fitted_agent()
    result = agent.recommend(KEY, 28, .4, n=2)
    interaction = dict(decision_id=result['decision_id'], id_site=result['recommendations'][0]['item_id'],
                       date=29, action='click')
    before = agent.get_ts_weights(KEY)
    assert agent.record_interaction(KEY, interaction)['accepted']
    assert not agent.record_interaction(KEY, interaction)['accepted']
    assert agent.get_ts_weights(KEY) != before
    assert not agent.record_interaction((999, 'AAA', '2013J'), interaction)['accepted']
    assert not agent.record_answer(KEY, {'question_id': 'q1', 'correct': True})['accepted']


def test_future_events_do_not_change_features_or_preferences():
    agent = fitted_agent()
    before = agent.build_snapshot(KEY, 28, .5)
    agent.update_history(pd.concat([agent.interactions, history([(4, 29, 999)])]))
    after = agent.build_snapshot(KEY, 28, .5)
    np.testing.assert_allclose(before['features'], after['features'])
    assert before['profile'] == after['profile']


def test_optional_diagnostic_is_neutral_without_reliability():
    from recommendations.oulad_agent import FEATURE_NAMES
    agent = fitted_agent()
    agent.set_irt_params({('AAA', '2013J', 1): .5})
    neutral = agent.build_snapshot(KEY, 28, .2)
    column = FEATURE_NAMES.index('difficulty_match')
    assert (neutral['features'][:, column] == .5).all()
    agent.update_user_profile(KEY, diagnostic={'ability': 0., 'se': 1.})
    informed = agent.build_snapshot(KEY, 28, .2)
    assert informed['features'][0, column] == pytest.approx(.75)


def test_new_profile_updates_and_recent_materials_remain_eligible():
    agent = fitted_agent()
    agent.update_history(history([(2, day, 4) for day in range(21, 28)]))
    result = agent.recommend(KEY, 28, .5, n=4)
    assert 2 in {r['item_id'] for r in result['recommendations']}
    assert result['profile']['preferences']['forumng'] > .5


def test_wrong_cutoff_is_rejected():
    agent = fitted_agent()
    with pytest.raises(ValueError):
        agent.recommend(KEY, 7, .5)
