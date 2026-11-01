"""Deterministic, time-adaptive interaction preferences, not knowledge estimates."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .recommendation_signals import finite_number, observable_context


class PreferenceModel:
    """Rebuild a smoothed preference snapshot from history available at a cutoff.

    Rebuilding makes repeated requests idempotent and supports corrected logs.
    Catalog-size correction approximates opportunity; real impressions are
    unavailable in OULAD and click frequency is not educational utility.
    """

    def __init__(self, half_life_days=14., prior_strength=3., daily_click_cap=20.):
        self.half_life_days = finite_number(half_life_days, 'half_life_days')
        self.prior_strength = finite_number(prior_strength, 'prior_strength')
        self.daily_click_cap = finite_number(daily_click_cap, 'daily_click_cap')
        if min(self.half_life_days, self.prior_strength, self.daily_click_cap) <= 0:
            raise ValueError('Preference parameters must be positive')

    def profile(self, interactions, resources, enrollment, as_of_day):
        catalog, observed = observable_context(resources, interactions, enrollment, as_of_day)
        opportunities = pd.concat([catalog[['id_site', 'activity_type']],
                                   observed[['id_site', 'activity_type']]]).drop_duplicates('id_site')
        types = sorted(opportunities['activity_type'].unique())
        counts = opportunities.groupby('activity_type').size().to_dict()
        n_types = len(types)
        neutral = 1. / n_types if n_types else 0.
        age = float(as_of_day) - observed['date'].to_numpy(dtype=float)
        decay = np.exp2(-age / self.half_life_days)
        evidence = np.log1p(np.minimum(observed['sum_click'].to_numpy(dtype=float), self.daily_click_cap)) * decay
        opportunity = observed['activity_type'].map(counts).to_numpy(dtype=float)
        observed = observed.assign(evidence=evidence / opportunity if len(evidence) else evidence)
        total = observed.groupby('activity_type')['evidence'].sum().to_dict()
        strength = float(sum(total.values()))
        preferences = {t: float((total.get(t, 0.) + self.prior_strength * neutral)
                                / (strength + self.prior_strength)) for t in types}
        effective_days = float(decay.sum())
        confidence = effective_days / (effective_days + self.prior_strength)

        def shares(mask):
            sums = observed.loc[mask].groupby('activity_type')['evidence'].sum().to_dict()
            mass = sum(sums.values())
            return {t: float(sums.get(t, 0.) / mass) if mass else neutral for t in types}

        recent = shares(observed['date'] >= float(as_of_day) - 7)
        previous = shares(observed['date'] < float(as_of_day) - 7)
        last_seen = observed.groupby('id_site')['date'].max().to_dict()
        entropy = -sum(p * np.log(p) for p in preferences.values() if p > 0)
        return {
            'preferences': preferences,
            'confidence': float(confidence),
            'n_resource_days': int(len(observed)),
            'effective_resource_days': effective_days,
            'last_seen': {str(int(k)): int(v) for k, v in sorted(last_seen.items())},
            'last_activity_day': int(observed['date'].max()) if len(observed) else None,
            'recent_preferences': recent,
            'preference_shift': {t: float(recent[t] - previous[t]) for t in types},
            'normalized_entropy': float(entropy / np.log(n_types)) if n_types > 1 else 0.,
            'as_of_day': int(as_of_day),
            'half_life_days': self.half_life_days,
            'prior_strength': self.prior_strength,
            'daily_click_cap': self.daily_click_cap,
            'evidence': 'decayed_capped_daily_clicks_per_eligible_or_observed_resource',
        }
