"""Hybrid OULAD ranking adapted from MARS evidence scoring and MMR.

No topic mastery, semantic embeddings or collaborative model is inferred from
the course failure probability. All scores below are ranking utilities.
"""
from __future__ import annotations

import numpy as np

from .preferences import PreferenceModel
from .recommendation_signals import finite_number, observable_context


class RecommendationAgent:
    def __init__(self, resources, interactions, preference_model=None, diversity_penalty=.15):
        self.resources = resources.copy(deep=True)
        self.interactions = interactions.copy(deep=True)
        self.preference_model = preference_model or PreferenceModel()
        self.diversity_penalty = finite_number(diversity_penalty, 'diversity_penalty', maximum=1)

    def update_history(self, interactions):
        """Replace the complete event snapshot; repeated refreshes are idempotent.

        This does not append logs or train models. Call recommend again with
        the new cutoff and its corresponding forecast to refresh the ranking.
        """
        self.interactions = interactions.copy(deep=True)

    def recommend(self, enrollment, as_of_day, forecast_probability, n=5, *, allow_review=False, include_candidates=False):
        """Read one risk forecast and rank available resources; no model writes.

        Refresh history using update_history (or create a new agent). Dates
        are course-relative integers; history excludes the cutoff day itself.
        """
        risk = finite_number(forecast_probability, 'forecast_probability', maximum=1)
        if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
            raise ValueError('n must be a positive integer')
        catalog, _ = observable_context(self.resources, self.interactions, enrollment, as_of_day)
        profile = self.preference_model.profile(self.interactions, self.resources, enrollment, as_of_day)
        weights = {'preference': .5, 'calendar': .2 + .2 * risk, 'engagement': .3 - .2 * risk}
        candidates = []
        for row in catalog.itertuples(index=False):
            item_id = int(row.id_site)
            seen = profile['last_seen'].get(str(item_id))
            if not allow_review and seen is not None and as_of_day - seen <= 7:
                continue
            preference = profile['preferences'][row.activity_type]
            # Snapshot smoothing already moves uncertain preferences to neutral.
            scheduled = np.isfinite(row.week_from)
            if scheduled:
                start = row.week_from * 7
                end = row.week_to * 7 + 6 if np.isfinite(row.week_to) else start + 6
                distance = max(start - as_of_day, as_of_day - end, 0.)
                calendar = float(np.exp(-distance / 28))
            else:
                calendar = .5
            engagement = 1. if seen is None else min(1., max(0., as_of_day - seen) / 28)
            score = weights['preference'] * preference + weights['calendar'] * calendar + weights['engagement'] * engagement
            reasons = ['matches_course_calendar' if calendar > .8 else 'available_learning_resource']
            if preference > 1 / max(len(profile['preferences']), 1):
                reasons.append('matches_observed_preferences')
            if seen is None:
                reasons.append('not_previously_observed')
            candidates.append({
                'item_id': item_id, 'item_type': 'vle_resource',
                'activity_type': row.activity_type, 'score': float(score),
                'strategy': 'preference_calendar_behavior', 'reasons': reasons,
                'signals': {'preference': preference, 'calendar': calendar, 'engagement': engagement},
                'availability_basis': 'explicit' if np.isfinite(row.available_from_day) else ('planned_week' if scheduled else 'unknown'),
            })
        # MMR-style novelty: repeat types receive a penalty; no random changes.
        unranked = [dict(c) for c in candidates] if include_candidates else None
        selected = []
        while candidates and len(selected) < n:
            seen_types = {r['activity_type'] for r in selected}
            for candidate in candidates:
                candidate['selection_score'] = candidate['score'] - self.diversity_penalty * (candidate['activity_type'] in seen_types)
            best = min(candidates, key=lambda c: (-c['selection_score'], c['item_id']))
            selected.append(best)
            candidates.remove(best)
        result = {
            'enrollment': {'id_student': int(enrollment[0]), 'code_module': str(enrollment[1]), 'code_presentation': str(enrollment[2])},
            'as_of_day': int(as_of_day), 'forecast_probability': risk,
            'forecast_semantics': 'Fail_or_Withdrawn',
            'profile': profile, 'weights': weights, 'recommendations': selected,
            'policy': {'history_boundary': 'date < as_of_day', 'recent_seen_days': 7,
                       'allow_review': bool(allow_review), 'diversity_penalty': self.diversity_penalty},
            'limitations': ['ranking_utility_not_learning_effect', 'clicks_are_implicit_preferences',
                            'planned_week_is_not_verified_release', 'no_topic_mastery_inferred'],
        }
        if include_candidates:
            result['candidates'] = unranked
        return result
