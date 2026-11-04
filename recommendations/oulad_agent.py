"""Full standalone OULAD adaptation of the final XES-MARS recommendation path.

Calendar/content/frozen implicit SVD -> learned fusion -> MMR -> bounded
daily-basket SASRec personalization. Forecast probabilities remain unchanged.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .preferences import PreferenceModel
from .recommendation_agent import RecommendationAgent as ReferenceAgent
from .recommendation_signals import ENROLLMENT, finite_number, observable_context, require_columns
from .retrieval import OULADRetrieval
from .sequential import DailySASRecService
from .strategy import StrategyController

FEATURE_NAMES = ['knowledge_based', 'content_based', 'collaborative', 'preference',
                 'calendar', 'engagement', 'preference_confidence', 'log_resource_days',
                 'log_capped_clicks', 'seen_before', 'query_activity', 'risk_calendar',
                 'difficulty_match', 'behavior_need', 'selected_strategy_score']


def _normalize(values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite recommendation utility')
    span = float(np.ptp(values)) if len(values) else 0.
    return (values - values.min()) / span if span > 1e-12 else np.zeros(len(values))


def _digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class OULADRecommendationAgent:
    def __init__(self, resources, interactions, retrieval=None, sequential=None, random_seed=42,
                 bandit_policy='thompson', allow_review=True, mmr_lambda=.8, sasrec_weight=.05):
        self.resources = resources.copy(deep=True)
        self.random_seed = int(random_seed)
        self.allow_review = bool(allow_review)
        self.mmr_lambda = finite_number(mmr_lambda, 'mmr_lambda', maximum=1)
        self.sasrec_weight = finite_number(sasrec_weight, 'sasrec_weight', maximum=1)
        self.retrieval = retrieval
        self.sequential = sequential
        self.controller = StrategyController(random_seed, bandit_policy)
        self.candidate_model = self.probability_model = None
        self.profiles, self.difficulties, self.prerequisites = {}, {}, {}
        self._shown, self._decision_number, self._catalog_cache = {}, 0, {}
        self.metadata = {'source': 'XES-MARS full agent plus external SASRec selector',
                         'adaptation': 'OULAD VLE resources and daily baskets; no pretrained XES weights',
                         'irt': 'unavailable_without_observed_response_metadata',
                         'content_source': 'activity_and_calendar_metadata',
                         'sasrec_group_semantics': 'activity_type unless explicit topic_tags supplied'}
        self.update_history(interactions)

    def update_history(self, interactions):
        require_columns(interactions, [*ENROLLMENT, 'id_site', 'date', 'sum_click'])
        self.interactions = interactions.copy(deep=True)
        self._event_positions = self.interactions.groupby(ENROLLMENT, sort=False).indices

    def initialize(self, fit_keys, as_of_day, selection_keys=None, train_sequence=True,
                   epochs=20, patience=4, device='auto', **kwargs):
        self.train_collaborative(fit_keys, as_of_day)
        self.sequential = self.sequential or DailySASRecService(seed=self.random_seed)
        self.sequential.fit(self.resources, self.interactions, fit_keys, as_of_day,
                            selection_keys=selection_keys, epochs=epochs if train_sequence else 0,
                            patience=patience, device=device, **kwargs)
        return self

    def train_collaborative(self, fit_keys, as_of_day, n_factors=32):
        self.retrieval = OULADRetrieval(self.random_seed, n_factors).fit(
            self.resources, self.interactions, fit_keys, as_of_day)
        return self

    def build_content_index(self, resources=None):
        if resources is not None:
            if self.retrieval is not None:
                raise ValueError('Catalog changes require refitting the frozen retrieval bundle')
            self.resources = resources.copy(deep=True)
            self._catalog_cache.clear()
        return {'content_source': self.metadata['content_source'], 'requires_fit': self.retrieval is None}

    def _history(self, key, day, history=None):
        if history is None:
            positions = self._event_positions.get(tuple(key), [])
            history = self.interactions.iloc[positions]
        return history.loc[(history.id_student == key[0]) & (history.code_module == key[1])
                           & (history.code_presentation == key[2]) & (history.date < day)].copy()

    def _available(self, key, day):
        cache_key = (key[1], key[2], int(day))
        if cache_key not in self._catalog_cache:
            catalog, _ = observable_context(self.resources, self.interactions.iloc[:0], key, day)
            self._catalog_cache[cache_key] = catalog
        return self._catalog_cache[cache_key]

    def update_user_profile(self, enrollment, interactions=None, confidence_result=None, diagnostic=None, **kwargs):
        key = tuple(enrollment)
        if interactions is not None:
            # Explicit complete history replacement, never append duplicate snapshots.
            self.update_history(interactions)
        state = self.profiles.setdefault(key, {})
        if diagnostic is not None:
            state['diagnostic'] = deepcopy(diagnostic)
        if confidence_result is not None:
            state['behavior_review_need'] = deepcopy(confidence_result.get('behavior', {}).get('review_need', {}))
        if 'topic_focus' in kwargs:
            state['topic_focus'] = deepcopy(kwargs['topic_focus'])

    def set_irt_params(self, item_params):
        self.difficulties = {tuple(k): float(v) for k, v in item_params.items()}

    def set_irt_response_params(self, params):
        if params:
            raise ValueError('OULAD has no per-resource correctness; provide a separately validated diagnostic instead')

    def set_confidence_schema(self, schema=None):
        if schema not in (None, 'behavioral_evidence'):
            raise ValueError('Only observed behavioral evidence is available for OULAD')

    def set_curriculum(self, prerequisites):
        self.prerequisites = {tuple(k): list(v) for k, v in prerequisites.items()}

    def select_strategy(self, enrollment, n_interactions=0, cf_available=True):
        return self.controller.choose(tuple(enrollment), n_interactions, cf_available)

    def get_ts_weights(self, enrollment):
        return self.controller.weights(tuple(enrollment))

    def update_reward(self, enrollment, strategy, reward):
        self.controller.update_reward(tuple(enrollment), strategy, reward)

    def build_snapshot(self, enrollment, as_of_day, forecast_probability, prepared=None, history=None):
        key = tuple(enrollment)
        risk = finite_number(forecast_probability, 'forecast_probability', maximum=1)
        day = finite_number(as_of_day, 'as_of_day')
        if day != int(day):
            raise ValueError('Use an integer course-relative day')
        if self.retrieval is None:
            raise RuntimeError('Initialize or load the agent before inference')
        history = self._history(key, day, history)
        available = self._available(key, day)
        if prepared is None:
            base = ReferenceAgent(available, history).recommend(key, day, risk, n=1,
                allow_review=self.allow_review, include_candidates=True)
            profile, candidates = base['profile'], base['candidates']
        else:
            part, query = prepared['candidates'], prepared['query']
            profile = {'preferences': part.groupby('kind', sort=True).preference.first().to_dict(),
                       'confidence': float(query['preference_confidence']),
                       'n_resource_days': int(query['history_resource_days']), 'as_of_day': int(day)}
            candidates = [{'item_id': int(r.item), 'activity_type': str(r.kind), 'item_type': 'vle_resource',
                           'score': .5 * r.preference + (.2 + .2 * risk) * r.calendar + (.3 - .2 * risk) * r.engagement,
                           'signals': {'preference': float(r.preference), 'calendar': float(r.calendar), 'engagement': float(r.engagement)}}
                          for r in part.itertuples()]
        ids = [c['item_id'] for c in candidates]
        if len(ids) != len(set(ids)) or not set(ids).issubset(set(available.id_site)):
            raise ValueError('Candidate identities differ from the available catalog')
        if self.allow_review and set(ids) != set(available.id_site):
            raise ValueError('Full-agent snapshots require the complete available catalog')
        available = available.set_index('id_site', drop=False).loc[ids].reset_index(drop=True)
        signals = self.retrieval.score(key, history, available, day)
        cf_ready = bool(signals['cf_available'] and profile['n_resource_days'] >= 20)
        normalized = {name: _normalize(signals[name]) for name in ['knowledge_based', 'content_based', 'collaborative']}
        if not cf_ready:
            normalized['collaborative'] = np.zeros(len(ids))
        # Feature construction is replayable: only actual exposure advances policy RNG.
        selected = self.controller.peek(key, profile['n_resource_days'], cf_ready)
        if prepared is not None and 'sasrec_affinity' in prepared:
            affinities = np.asarray(prepared['sasrec_affinity'], dtype=float)
            if affinities.shape != (len(ids),):
                raise ValueError('Prepared SASRec affinity shape mismatch')
        else:
            affinities = (self.sequential.affinity(key, history, available, day)
                          if self.sequential is not None else np.zeros(len(ids)))
        known = history.loc[history.id_site.isin(self.resources.loc[
            (self.resources.code_module == key[1]) & (self.resources.code_presentation == key[2]), 'id_site'])]
        daily = known.groupby(['id_site', 'date'], as_index=False).sum(numeric_only=True)
        daily = daily.loc[daily.sum_click > 0]
        resource_days = daily.groupby('id_site').size()
        clicks = daily.assign(capped=daily.sum_click.clip(upper=20)).groupby('id_site').capped.sum()
        latest = float(daily.date.max()) if len(daily) else None
        query_activity = float(np.exp(-(day - latest) / 7)) if latest is not None else 0.
        state = self.profiles.get(key, {})
        diagnostic = state.get('diagnostic', {})
        ability, error = diagnostic.get('ability'), diagnostic.get('se', float('inf'))
        reliability = 1 / (1 + float(error) ** 2) if np.isfinite(error) else 0.
        matrix = []
        for i, candidate in enumerate(candidates):
            item, kind = candidate['item_id'], candidate['activity_type']
            difficulty = self.difficulties.get((key[1], key[2], item))
            match = .5 if ability is None or difficulty is None else .5 + reliability * (max(0., 1 - abs(difficulty - (float(ability) + .5)) / 4) - .5)
            behavior = finite_number(state.get('behavior_review_need', {}).get(kind, 0.), 'behavior_need', maximum=1)
            f = [normalized[name][i] for name in ['knowledge_based', 'content_based', 'collaborative']]
            f += [candidate['signals']['preference'], candidate['signals']['calendar'], candidate['signals']['engagement'],
                  profile['confidence'], float(np.log1p(resource_days.get(item, 0))), float(np.log1p(clicks.get(item, 0))),
                  float(item in resource_days), query_activity, risk * candidate['signals']['calendar'],
                  float(match), behavior, float(normalized[selected][i])]
            matrix.append(f)
            candidate.update(sasrec_affinity=float(affinities[i]), strategy=selected,
                             agent_features=[float(v) for v in f], history_resource_days=int(resource_days.get(item, 0)))
            # Utility, not a probability. The learned fusion may replace this score.
            candidate['score'] = float(.35 * f[4] + .25 * f[3] + .15 * f[1] + .15 * f[2] + .1 * f[5] + .05 * f[-1])
        features = np.asarray(matrix, dtype=float).reshape(len(ids), len(FEATURE_NAMES))
        if not np.isfinite(features).all():
            raise ValueError('Nonfinite agent feature')
        return {'enrollment': dict(zip(ENROLLMENT, [int(key[0]), str(key[1]), str(key[2])])),
                'as_of_day': int(day), 'forecast_probability': risk, 'profile': profile,
                'candidates': candidates, 'features': features, 'feature_names': FEATURE_NAMES.copy(),
                'strategy_selected': selected, 'cf_available': cf_ready,
                'strategy_weights': self.get_ts_weights(key)}

    @staticmethod
    def _tag_similarity(a, b):
        left, right = set(a.get('topic_tags', [])), set(b.get('topic_tags', []))
        return len(left & right) / len(left | right) if left and right else float(a['activity_type'] == b['activity_type'])

    def _mmr_rerank(self, candidates, n=10, lam=None):
        lam = self.mmr_lambda if lam is None else finite_number(lam, 'lam', maximum=1)
        candidates = deepcopy(candidates)
        normalized = _normalize([c['score'] for c in candidates])
        for c, score in zip(candidates, normalized):
            c['_mmr_relevance'] = float(score)
        chosen = []
        while candidates and len(chosen) < n:
            at = max(range(len(candidates)), key=lambda i: (
                lam * candidates[i]['_mmr_relevance'] - (1 - lam) * max((self._tag_similarity(candidates[i], c) for c in chosen), default=0.),
                -candidates[i]['item_id']))
            chosen.append(candidates.pop(at))
        for c in chosen:
            c.pop('_mmr_relevance', None)
        return chosen

    def rank_candidates(self, enrollment, candidates, user_profile=None):
        output = deepcopy(candidates)
        if self.candidate_model is not None and output:
            x = np.asarray([c['agent_features'] for c in output], dtype=float)
            p = self.candidate_model.predict_proba(x)[:, 1]
            for c, value in zip(output, p):
                c['score'] = float(value)
        return sorted(output, key=lambda c: (-c['score'], c['item_id']))

    def _select_sequence(self, selected, pool):
        if self.sasrec_weight == 0:
            return selected, []
        groups = {}
        for c in pool:
            groups.setdefault(c['activity_type'], []).append(c)
        normalized = {kind: dict(zip([c['item_id'] for c in group], _normalize([c.get('sasrec_affinity', 0.) for c in group]))) for kind, group in groups.items()}
        output, used, trace = [], set(), []
        for slot in selected:
            choices = [slot] + [c for c in groups[slot['activity_type']] if c['item_id'] != slot['item_id']]
            choices = [c for c in choices if c['item_id'] not in used]
            def utility(c):
                affinity = normalized[c['activity_type']][c['item_id']]
                exposure = 1 / (1 + c.get('history_resource_days', 0))
                return min(c['score'], slot['score']) + self.sasrec_weight * (affinity + exposure) / 2
            chosen = deepcopy(max(choices, key=utility))
            chosen['slot_priority_score'] = float(slot['score'])
            used.add(chosen['item_id'])
            output.append(chosen)
            trace.append({'slot_item_id': slot['item_id'], 'selected_item_id': chosen['item_id'],
                          'activity_type_preserved': slot['activity_type'], 'sasrec_affinity': chosen.get('sasrec_affinity', 0.)})
        return output, trace

    def rank_snapshot(self, snapshot, n=10, record_exposure=False):
        if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
            raise ValueError('Positive integer recommendation count required')
        if list(snapshot['feature_names']) != FEATURE_NAMES:
            raise ValueError('Feature schema mismatch')
        key = tuple(snapshot['enrollment'][name] for name in ENROLLMENT)
        candidates = deepcopy(snapshot['candidates'])
        x = np.asarray(snapshot['features'], dtype=float)
        for candidate, features in zip(candidates, x):
            candidate['agent_features'] = features.tolist()
        ranked = self.rank_candidates(key, candidates)
        selected, trace = self._select_sequence(self._mmr_rerank(ranked, n), ranked)
        if self.probability_model is not None and candidates:
            raw = self.candidate_model.predict_proba(x)[:, 1] if self.candidate_model is not None else np.asarray([c['score'] for c in candidates])
            probabilities = self.probability_model.predict_proba(np.column_stack([x, raw]))[:, 1]
            probability_map = dict(zip([c['item_id'] for c in candidates], probabilities))
            for c in selected:
                c['interaction_probability'] = float(probability_map[c['item_id']])
        decision_id = None
        if record_exposure:
            self.select_strategy(key, snapshot['profile']['n_resource_days'], snapshot['cf_available'])
            self._decision_number += 1
            decision_id = hashlib.sha256(repr((key, snapshot['as_of_day'], self._decision_number,
                                               [c['item_id'] for c in selected])).encode()).hexdigest()[:24]
            self._shown[decision_id] = {'key': key, 'day': snapshot['as_of_day'],
                                        'items': {c['item_id']: snapshot['strategy_selected'] for c in selected}, 'consumed': set()}
        for c in selected:
            c.pop('agent_features', None)
        return {key_: deepcopy(snapshot[key_]) for key_ in ['enrollment', 'as_of_day', 'forecast_probability', 'profile', 'strategy_selected', 'strategy_weights']} | {
            'decision_id': decision_id, 'recommendations': selected, 'n_candidates': len(candidates),
            'sequence_personalization': trace, 'diagnostics': deepcopy(self.metadata),
            'ranking_model': 'learned_fusion' if self.candidate_model is not None else 'fallback_utility',
            'mmr_lambda': self.mmr_lambda, 'sasrec_weight': self.sasrec_weight,
            'probability_semantics': 'contextual future-interaction head, not course success or learning gain'}

    def recommend(self, enrollment, as_of_day, forecast_probability, n=10, **kwargs):
        return self.rank_snapshot(self.build_snapshot(enrollment, as_of_day, forecast_probability, **kwargs), n, record_exposure=True)

    def rerank(self, enrollment, as_of_day, forecast_probability, n=10, **kwargs):
        return self.recommend(enrollment, as_of_day, forecast_probability, n, **kwargs)

    def _get_strategy_items(self, strategy, enrollment, as_of_day, n):
        history = self._history(tuple(enrollment), as_of_day)
        available = self._available(tuple(enrollment), as_of_day)
        scores = self.retrieval.score(tuple(enrollment), history, available, as_of_day)[strategy]
        order = np.lexsort((available.id_site.to_numpy(), -np.asarray(scores)))[:n]
        return [{'item_id': int(available.iloc[i].id_site), 'item_type': 'vle_resource',
                 'activity_type': str(available.iloc[i].activity_type), 'score': float(scores[i]), 'strategy': strategy} for i in order]

    def get_knowledge_based(self, enrollment, as_of_day, n=10):
        return self._get_strategy_items('knowledge_based', enrollment, as_of_day, n)

    def get_content_based(self, enrollment, as_of_day, n=10):
        return self._get_strategy_items('content_based', enrollment, as_of_day, n)

    def get_collaborative(self, enrollment, as_of_day, n=10):
        return self._get_strategy_items('collaborative', enrollment, as_of_day, n)

    def record_interaction(self, enrollment, interaction):
        decision = self._shown.get(interaction.get('decision_id'))
        if decision is None or tuple(enrollment) != decision['key']:
            return {'accepted': False, 'reason': 'no_matching_exposure'}
        item = interaction.get('id_site')
        if item not in decision['items'] or item in decision['consumed']:
            return {'accepted': False, 'reason': 'not_shown_or_already_credited'}
        day = finite_number(interaction.get('date'), 'date')
        if day < decision['day']:
            return {'accepted': False, 'reason': 'feedback_precedes_exposure'}
        action = interaction.get('action')
        if action not in ['click', 'completion', 'dismiss']:
            return {'accepted': False, 'reason': 'explicit_observed_action_required'}
        reward = float(action != 'dismiss')
        strategy = decision['items'][item]
        self.update_reward(enrollment, strategy, reward)
        decision['consumed'].add(item)
        return {'accepted': True, 'strategy': strategy, 'reward': reward, 'reward_source': 'observed_response_to_shown_resource'}

    def record_answer(self, enrollment, interaction):
        if 'id_site' not in interaction:
            return {'accepted': False, 'reason': 'OULAD_resource_id_required; correctness_is_not_preference'}
        return self.record_interaction(enrollment, interaction)

    def load_ranking_models(self, path, checkpoint_sha256=None):
        bundle = joblib.load(path)
        if bundle.get('feature_names') != FEATURE_NAMES:
            raise ValueError('Ranking feature schema mismatch')
        if checkpoint_sha256 is not None and bundle.get('forecast_checkpoint_sha256') != checkpoint_sha256:
            raise ValueError('Forecast checkpoint mismatch')
        self.candidate_model, self.probability_model = bundle.get('candidate_model'), bundle.get('probability_model')

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        joblib.dump(self.retrieval, directory / 'retrieval.joblib')
        if self.sequential is not None:
            self.sequential.save(directory / 'sequential.pt')
        joblib.dump({'feature_names': FEATURE_NAMES, 'candidate_model': self.candidate_model,
                     'probability_model': self.probability_model, 'controller': self.controller,
                     'profiles': self.profiles, 'difficulties': self.difficulties, 'prerequisites': self.prerequisites,
                     'decision_number': self._decision_number, 'shown': self._shown}, directory / 'ranking.joblib')
        metadata = {'format_version': 1, 'random_seed': self.random_seed, 'bandit_policy': self.controller.policy,
                    'allow_review': self.allow_review, 'mmr_lambda': self.mmr_lambda, 'sasrec_weight': self.sasrec_weight,
                    'feature_names': FEATURE_NAMES, 'metadata': self.metadata,
                    'artifacts_sha256': {p.name: _digest(p) for p in directory.iterdir() if p.is_file()}}
        (directory / 'bundle.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')

    @classmethod
    def load(cls, directory, resources, interactions, device='cpu'):
        directory = Path(directory)
        metadata = json.loads((directory / 'bundle.json').read_text(encoding='utf-8'))
        if metadata['format_version'] != 1 or metadata['feature_names'] != FEATURE_NAMES:
            raise ValueError('Agent bundle schema mismatch')
        for name, digest in metadata['artifacts_sha256'].items():
            path = (directory / name).resolve()
            if not path.is_relative_to(directory.resolve()) or _digest(path) != digest:
                raise ValueError('Agent bundle checksum mismatch')
        retrieval = joblib.load(directory / 'retrieval.joblib')
        sequential = DailySASRecService.load(directory / 'sequential.pt', device) if (directory / 'sequential.pt').exists() else None
        agent = cls(resources, interactions, retrieval, sequential,
                    **{k: metadata[k] for k in ['random_seed', 'bandit_policy', 'allow_review', 'mmr_lambda', 'sasrec_weight']})
        saved = joblib.load(directory / 'ranking.joblib')
        for name in ['candidate_model', 'probability_model', 'controller', 'profiles', 'difficulties', 'prerequisites']:
            setattr(agent, name, saved[name])
        agent._decision_number, agent._shown, agent.metadata = saved['decision_number'], saved['shown'], metadata['metadata']
        return agent
