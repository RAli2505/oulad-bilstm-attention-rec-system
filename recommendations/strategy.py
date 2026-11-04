"""Serializable per-enrollment policies, updated only by explicit rewards."""
from __future__ import annotations

import numpy as np
from copy import copy, deepcopy

from .recommendation_signals import finite_number

STRATEGIES = ('knowledge_based', 'content_based', 'collaborative')
POLICIES = ('thompson', 'ucb1', 'epsilon_greedy', 'fixed_kb', 'fixed_round_robin')
CF_MIN_INTERACTIONS = 20


class StrategyController:
    def __init__(self, seed=42, policy='thompson'):
        if policy not in POLICIES:
            raise ValueError(f'Unknown policy: {policy}')
        self.seed, self.policy = seed, policy
        self._rng = np.random.default_rng(seed)
        self._state = {}
        self._steps = {}

    def _params(self, user_key):
        if isinstance(user_key, list):
            user_key = tuple(user_key)
        if user_key not in self._state:
            self._state[user_key] = {s: {'alpha': float(a), 'beta': 1.0, 'count': 0, 'reward': 0.0} for s, a in zip(STRATEGIES, (10, 5, 1))}
        return user_key, self._state[user_key]

    def weights(self, user_key):
        _, params = self._params(user_key)
        means = {s: p['alpha'] / (p['alpha'] + p['beta']) for s, p in params.items()}
        total = sum(means.values())
        return {s: value / total for s, value in means.items()}

    def choose(self, user_key, n_interactions, cf_available=True):
        n = finite_number(n_interactions, 'n_interactions')
        key, params = self._params(user_key)
        eligible = list(STRATEGIES if cf_available and n >= CF_MIN_INTERACTIONS else STRATEGIES[:2])
        if self.policy == 'fixed_kb':
            return STRATEGIES[0]
        if self.policy == 'fixed_round_robin':
            step = self._steps.get(key, 0)
            self._steps[key] = step + 1
            return eligible[step % len(eligible)]
        if self.policy == 'epsilon_greedy':
            if self._rng.random() < 0.1:
                return eligible[int(self._rng.integers(len(eligible)))]
            return max(eligible, key=lambda s: params[s]['alpha'] / (params[s]['alpha'] + params[s]['beta']))
        if self.policy == 'ucb1':
            for s in eligible:
                if not params[s]['count']:
                    return s
            total = sum(p['count'] for p in params.values())
            return max(eligible, key=lambda s: params[s]['reward'] / params[s]['count'] + np.sqrt(2 * np.log(total + 1) / params[s]['count']))
        return max(eligible, key=lambda s: self._rng.beta(params[s]['alpha'], params[s]['beta']))

    def peek(self, user_key, n_interactions, cf_available=True):
        """Replay one choice without copying every other student's state."""
        key = tuple(user_key) if isinstance(user_key, list) else user_key
        local = copy(self)
        local._rng = deepcopy(self._rng)
        local._state = {key: deepcopy(self._state[key])} if key in self._state else {}
        local._steps = {key: self._steps[key]} if key in self._steps else {}
        return local.choose(key, n_interactions, cf_available)

    def update_reward(self, user_key, strategy, reward):
        if strategy not in STRATEGIES:
            raise ValueError(f'Unknown strategy: {strategy}')
        reward = finite_number(reward, 'reward', maximum=1)
        _, params = self._params(user_key)
        p = params[strategy]
        p['alpha'] += reward
        p['beta'] += 1 - reward
        p['count'] += 1
        p['reward'] += reward
