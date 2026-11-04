import ast
import inspect

import numpy as np
import pandas as pd
import pytest

from run_risk_practical_reports import top_fraction, scores
from run_course_transfer import course_cohort, logged_fit_function
from run_recommendation_controls import make_variant


def test_capacity_ranking_has_fixed_budget_with_stable_ties():
    p = np.array([.5, .8, .8, .1, .4])
    selected = top_fraction(p, .4)
    assert selected.tolist() == [False, True, True, False, False]
    assert top_fraction(p, .2).tolist() == [False, True, False, False, False]
    result = scores([1, 1, 0, 0, 0], p, selected)
    assert result['precision'] == result['recall'] == .5


def test_course_holdout_removes_other_enrollments_of_held_students():
    data = pd.DataFrame({'id_student': [1, 1, 2, 3, 4], 'code_module': ['AAA', 'BBB', 'AAA', 'BBB', 'CCC']})
    cohort, held = course_cohort(data, 'AAA')
    assert set(held) == {1, 2}
    assert not ((cohort.id_student == 1) & (cohort.code_module == 'BBB')).any()
    assert set(cohort.loc[cohort.code_module != 'AAA', 'id_student']) == {3, 4}
    with pytest.raises(ValueError):
        course_cohort(data, 'ZZZ')


def test_epoch_logging_changes_only_print_statements():
    import risk_experiments.training as training
    original = ast.parse(inspect.getsource(training.fit_predict))
    logged, source = logged_fit_function(training)
    class StripPrint(ast.NodeTransformer):
        def visit_Expr(self, node):
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == 'print':
                return None
            return self.generic_visit(node)
    assert ast.dump(StripPrint().visit(ast.parse(source)), include_attributes=False) == ast.dump(StripPrint().visit(original), include_attributes=False)
    assert logged is not training.fit_predict


def test_ablations_remove_dependent_strategy_signal_and_keep_identity_order():
    names = ['preference', 'preference_confidence', 'risk_calendar', 'collaborative', 'content_based', 'selected_strategy_score']
    x = np.ones((3, len(names)))
    data = pd.DataFrame({'query': [2, 1, 2], 'kind': ['a', 'b', 'b']})
    queries = pd.DataFrame({'query': [1, 2], 'id_student': [10, 20], 'code_module': ['A', 'A'], 'code_presentation': ['2013J', '2013J']})
    diagnostic = pd.DataFrame({'query': [1, 2], 'strategy_selected': ['knowledge_based', 'collaborative']})
    frozen = pd.DataFrame({'id_student': [20, 10, 20], 'code_module': ['A']*3, 'code_presentation': ['2013J']*3,
                           'kind': ['b', 'b', 'a'], 'frozen_preference': [.3, .4, .7], 'frozen_confidence': [.5, .6, .8]})
    removed = make_variant('without_cf', x, names, data, queries, diagnostic, frozen)
    np.testing.assert_array_equal(removed[:, 3], 0)
    np.testing.assert_array_equal(removed[:, 5], [0, 1, 0])
    static = make_variant('frozen_profile_day7', x, names, data, queries, diagnostic, frozen)
    np.testing.assert_array_equal(static[:, 0], [.7, .4, .3])
    np.testing.assert_array_equal(static[:, 1], [.8, .6, .5])
    np.testing.assert_array_equal(x, np.ones_like(x))
