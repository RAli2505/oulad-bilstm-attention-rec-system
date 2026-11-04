"""Observable calendar/content utilities and frozen implicit-click SVD.

Utilities are not probabilities of mastery or learning gain. All enrollment
identities use (student, module, presentation), resources use the full course
and site identity. New users are folded into frozen factors using ridge.
"""
from __future__ import annotations

from time import perf_counter

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import normalize

from .recommendation_signals import ENROLLMENT, finite_number, numeric_column, observable_context, require_columns

RESOURCE = ['code_module', 'code_presentation', 'id_site']


def _metadata(row):
    features = {'activity_type=' + str(row['activity_type']): 1.0}
    for column in ('week_from', 'week_to'):
        value = row.get(column, np.nan)
        if pd.notna(value):
            features[f'{column}={float(value):g}'] = 0.5
    return features


class OULADRetrieval:
    def __init__(self, seed=42, n_factors=32):
        if isinstance(n_factors, bool) or int(n_factors) != n_factors or n_factors < 1:
            raise ValueError('n_factors must be a positive integer')
        self.seed = seed
        self.n_factors = int(n_factors)

    def fit(self, resources, interactions, fit_keys, as_of_day):
        started = perf_counter()
        day = finite_number(as_of_day, 'as_of_day')
        if day != int(day):
            raise ValueError('as_of_day must be an integer course day')
        require_columns(resources, RESOURCE + ['activity_type'])
        require_columns(interactions, ENROLLMENT + ['id_site', 'date', 'sum_click'])
        if resources[RESOURCE].isna().any().any() or resources.duplicated(RESOURCE).any():
            raise ValueError('Resource identities must be complete and unique')
        self.resources_ = resources.copy(deep=True)
        site = numeric_column(self.resources_, 'id_site')
        if ((site < 0) | (site % 1 != 0)).any():
            raise ValueError('id_site must be a nonnegative integer')
        self.resources_['id_site'] = site.astype('int64')
        if self.resources_['activity_type'].isna().any() or self.resources_['activity_type'].astype(str).str.strip().eq('').any():
            raise ValueError('Missing activity_type')
        for column in ('week_from', 'week_to', 'available_from_day', 'available_until_day'):
            if column in self.resources_:
                self.resources_[column] = numeric_column(self.resources_, column, optional=True)
        start = self.resources_.get('available_from_day', pd.Series(np.nan, index=resources.index))
        weeks = self.resources_.get('week_from', pd.Series(np.nan, index=resources.index))
        start = start.where(start.notna(), weeks * 7)
        end = self.resources_.get('available_until_day', pd.Series(np.nan, index=resources.index))
        if ((start > end) & start.notna() & end.notna()).any():
            raise ValueError('Availability ends before it starts')
        self.course_resources_ = {key: group.copy() for key, group in self.resources_.groupby(RESOURCE[:2], sort=False)}
        keys = list(fit_keys[ENROLLMENT].itertuples(index=False, name=None)) if isinstance(fit_keys, pd.DataFrame) else list(fit_keys)
        keys = [tuple(k) for k in keys]
        if any(len(k) != 3 or any(pd.isna(x) for x in k) for k in keys):
            raise ValueError('Expected complete fit enrollment keys')
        keys = sorted(set(keys), key=repr)
        self.fit_keys_ = tuple(keys)
        self.as_of_day_ = day
        self.resource_keys_ = sorted(map(tuple, self.resources_[RESOURCE].itertuples(index=False, name=None)), key=repr)
        self.item_index_ = {k: i for i, k in enumerate(self.resource_keys_)}
        # A single exact-key join avoids an O(enrollments * event_count) scan.
        key_frame = pd.DataFrame(keys, columns=ENROLLMENT)
        key_frame['_row'] = np.arange(len(keys))
        selected = interactions.merge(key_frame, on=ENROLLMENT, how='inner', validate='many_to_one')
        selected['date'] = numeric_column(selected, 'date')
        if (selected['date'] % 1 != 0).any():
            raise ValueError('date must use integer course days')
        selected = selected.loc[selected['date'] < day].copy()
        selected['sum_click'] = numeric_column(selected, 'sum_click')
        if (selected['sum_click'] < 0).any():
            raise ValueError('sum_click cannot be negative')
        selected['id_site'] = numeric_column(selected, 'id_site')
        if ((selected['id_site'] < 0) | (selected['id_site'] % 1 != 0)).any():
            raise ValueError('id_site must be a nonnegative integer')
        catalog_index = pd.DataFrame(self.resource_keys_, columns=RESOURCE)
        catalog_index['_col'] = np.arange(len(catalog_index))
        selected = selected.merge(catalog_index, on=RESOURCE, how='inner', validate='many_to_one')
        daily = selected.groupby(['_row', '_col', 'date'], sort=True)['sum_click'].sum()
        daily = daily[daily > 0]
        totals = daily.groupby(level=['_row', '_col'], sort=True).sum()
        matrix = csr_matrix((np.log1p(totals.clip(upper=100).to_numpy()),
                             (totals.index.get_level_values('_row'), totals.index.get_level_values('_col'))),
                            shape=(len(keys), len(self.item_index_)), dtype=float)
        self.metadata_vectorizer_ = DictVectorizer()
        ordered = self.resources_.merge(catalog_index, on=RESOURCE, validate='one_to_one').sort_values('_col')
        if len(ordered):
            self.metadata_vectors_ = normalize(self.metadata_vectorizer_.fit_transform([_metadata(row) for row in ordered.to_dict('records')]))
        else:
            self.metadata_vectors_ = csr_matrix((0, 0))
        self.item_factors_ = np.zeros((len(self.item_index_), 0))
        self.svd_ = None
        if matrix.nnz and matrix.shape[1] >= 2:
            self.svd_ = TruncatedSVD(n_components=min(self.n_factors, *matrix.shape), random_state=self.seed)
            self.svd_.fit(matrix)
            self.item_factors_ = self.svd_.components_.T.copy()
        elif matrix.nnz:
            self.item_factors_ = np.ones((1, 1))
        self.fit_metadata_ = {'elapsed_seconds': perf_counter() - started, 'fit_enrollments': len(keys),
                              'past_known_events': len(selected), 'nonzero_user_items': matrix.nnz,
                              'content_features': self.metadata_vectors_.shape[1]}
        return self

    def score(self, enrollment, history, available, as_of_day):
        if not hasattr(self, 'item_index_'):
            raise RuntimeError('Call fit before score')
        as_of_day = finite_number(as_of_day, 'as_of_day')
        if as_of_day < self.as_of_day_:
            raise ValueError('Cannot score before the frozen fit cutoff')
        enrollment = tuple(enrollment)
        catalog = self.course_resources_.get(enrollment[1:], self.resources_.iloc[:0])
        _, observed = observable_context(catalog, history, enrollment, as_of_day)
        require_columns(available, RESOURCE + ['activity_type'])
        n = len(available)
        kb, cb, cf = np.zeros(n), np.zeros(n), np.zeros(n)
        totals = observed.groupby('id_site')['sum_click'].sum()
        recent = observed.groupby('id_site')['date'].max()
        indices, strength = [], []
        for site, clicks in totals.items():
            idx = self.item_index_.get((enrollment[1], enrollment[2], site))
            if idx is not None:
                indices.append(idx); strength.append(np.log1p(min(float(clicks), 100.0)))
        cf_available = bool(indices and self.item_factors_.shape[1])
        if indices and n and self.metadata_vectors_.shape[1]:
            profile = csr_matrix(np.asarray(self.metadata_vectors_[indices].multiply(np.asarray(strength)[:, None]).sum(axis=0)))
            profile = normalize(profile)
            candidates = normalize(self.metadata_vectorizer_.transform([_metadata(row) for row in available.to_dict('records')]))
            cb = (candidates @ profile.T).toarray().ravel()
        user = None
        if cf_available:
            factors = self.item_factors_[indices]
            user = np.linalg.solve(factors.T @ factors + np.eye(factors.shape[1]), factors.T @ np.asarray(strength))
        for i, resource in enumerate(available.to_dict('records')):
            key = tuple(resource[c] for c in RESOURCE)
            if key[:2] != enrollment[1:]:
                cb[i] = 0.0
                continue
            site = key[2]
            # Absence of observed access is a curriculum coverage signal only.
            start = resource.get('week_from', np.nan)
            end = resource.get('week_to', np.nan)
            current = (pd.isna(start) or start * 7 <= as_of_day) and (pd.isna(end) or as_of_day <= (end + 1) * 7)
            elapsed = max(0.0, as_of_day - recent.get(site, as_of_day))
            kb[i] = (1.0 if site not in totals else 0.25 * (1 - np.exp(-elapsed / 14))) + 0.5 * current
            idx = self.item_index_.get(key)
            if user is not None and idx is not None:
                cf[i] = max(0.0, float(self.item_factors_[idx] @ user))
        return {'knowledge_based': kb, 'content_based': cb, 'collaborative': cf, 'cf_available': cf_available}
