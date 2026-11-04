"""Bounded batch implementation of the reference recommendation features.

No sampling, approximation or changed learning policy. Batch boundaries are
implementation details; every query retains the complete eligible catalog.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .recommendation_signals import ENROLLMENT, numeric_column, observable_context

ROLES = ('fit', 'selection', 'calibration', 'held')
COLUMNS = ['query', 'item', 'kind', 'label', 'preference', 'calendar', 'engagement',
           'popularity', 'confidence', 'no_decay']
QUERY_COLUMNS = ['query', *ENROLLMENT, 'role', 'forecast_position', 'start', 'stop',
                 'n_future_all', 'n_relevant', 'reachable_fraction', 'history_resource_days',
                 'preference_confidence', 'preference_shift_l1']


def _python_sum(series):
    # Match reference sum(dict.values()) in sorted activity-type order.
    return float(sum(series.tolist()))


def _profiles(observed, opportunities, day, half_life, compute_shift=True):
    keys = ['query', 'activity_type']
    counts = opportunities.drop_duplicates(['query', 'id_site']).groupby(keys, sort=True).size().rename('opportunities').reset_index()
    if counts.empty:
        return pd.DataFrame(columns=[*keys, 'preference']), pd.DataFrame(columns=['query', 'confidence', 'shift'])
    obs = observed.merge(counts, on=keys, how='left', validate='many_to_one', sort=False)
    obs['decay'] = np.exp2(-(float(day) - obs['date'].to_numpy(dtype=float)) / half_life)
    obs['evidence'] = (np.log1p(np.minimum(obs['sum_click'].to_numpy(dtype=float), 20.))
                       * obs['decay'].to_numpy() / obs['opportunities'].to_numpy(dtype=float))
    totals = obs.groupby(keys, sort=True).evidence.sum().reset_index()
    profiles = counts.merge(totals, on=keys, how='left', validate='one_to_one', sort=False)
    profiles['evidence'] = profiles.evidence.fillna(0.)
    n_types = profiles.groupby('query', sort=True).size()
    neutral = 1. / profiles['query'].map(n_types).to_numpy(dtype=float)
    strength = profiles.groupby('query', sort=True).evidence.agg(_python_sum)
    profiles['preference'] = ((profiles.evidence.to_numpy() + 3. * neutral)
                              / (profiles['query'].map(strength).to_numpy(dtype=float) + 3.))
    effective = obs.groupby('query', sort=True).decay.agg(lambda s: float(s.to_numpy().sum()))
    query_stats = pd.DataFrame({'query': n_types.index})
    mass = query_stats['query'].map(effective).fillna(0.).to_numpy(dtype=float)
    query_stats['confidence'] = mass / (mass + 3.)
    if compute_shift:
        shares = []
        for recent in [True, False]:
            subset = obs.loc[(obs.date >= float(day) - 7) if recent else (obs.date < float(day) - 7)]
            values = subset.groupby(keys, sort=True).evidence.sum().rename('value').reset_index()
            aligned = profiles[keys].merge(values, on=keys, how='left', validate='one_to_one', sort=False)
            aligned['value'] = aligned.value.fillna(0.)
            total = aligned.groupby('query', sort=True).value.agg(_python_sum)
            denominator = aligned['query'].map(total).to_numpy(dtype=float)
            result = neutral.copy()
            np.divide(aligned.value.to_numpy(), denominator, out=result, where=denominator > 0)
            shares.append(result)
        shift = profiles[['query']].assign(shift=np.abs(shares[0] - shares[1])).groupby('query', sort=True)['shift'].agg(_python_sum)
        query_stats['shift'] = query_stats['query'].map(shift)
    else:
        query_stats['shift'] = 0.
    return profiles[[*keys, 'preference']], query_stats


def _batch(queries, mapped, catalog, metadata, day, horizon, include_no_decay, allow_review=False):
    day = float(day)
    past = mapped.loc[mapped.date < day, ['query', 'id_site', 'date', 'sum_click']].copy()
    past['sum_click'] = numeric_column(past, 'sum_click')
    if (past.sum_click < 0).any():
        raise ValueError('sum_click cannot be negative')
    past['id_site'] = numeric_column(past, 'id_site')
    if (past.id_site < 0).any() or (past.id_site % 1 != 0).any():
        raise ValueError('id_site must be a nonnegative integer')
    past['id_site'] = past.id_site.astype('int64')
    observed = past.groupby(['query', 'id_site', 'date'], sort=True, as_index=False).sum()
    observed = observed.loc[observed.sum_click > 0].merge(metadata, on='id_site', how='inner', validate='many_to_one', sort=False)
    available = queries[['query']].merge(catalog, how='cross')
    opportunities = pd.concat([available[['query', 'id_site', 'activity_type']], observed[['query', 'id_site', 'activity_type']]], ignore_index=True)
    profiles, profile_stats = _profiles(observed, opportunities, day, 14.)
    candidates = available.merge(profiles, on=['query', 'activity_type'], how='left', validate='many_to_one', sort=False)
    if include_no_decay:
        static, _ = _profiles(observed, opportunities, day, 1e12, compute_shift=False)
        candidates = candidates.merge(static.rename(columns={'preference': 'no_decay'}), on=['query', 'activity_type'], how='left', validate='many_to_one', sort=False)
    else:
        candidates['no_decay'] = candidates.preference
    last = observed.groupby(['query', 'id_site'], sort=True).date.max().rename('last_seen').reset_index()
    candidates = candidates.merge(last, on=['query', 'id_site'], how='left', validate='one_to_one', sort=False)
    if not allow_review:
        candidates = candidates.loc[candidates.last_seen.isna() | (day - candidates.last_seen > 7)].copy()
    age = day - candidates.last_seen.to_numpy(dtype=float)
    candidates['engagement'] = np.where(np.isnan(age), 1., np.clip(age / 28., 0., 1.))
    start = candidates.week_from.to_numpy(dtype=float) * 7
    week_to = candidates.week_to.to_numpy(dtype=float)
    end = np.where(np.isfinite(week_to), week_to * 7 + 6, start + 6)
    distance = np.maximum(np.maximum(start - day, day - end), 0.)
    candidates['calendar'] = np.where(np.isfinite(start), np.exp(-distance / 28.), .5)
    future = mapped.loc[(mapped.date >= day) & (mapped.date < day + horizon) & (mapped.sum_click > 0), ['query', 'id_site']].copy()
    future['id_site'] = future.id_site.astype(int)
    future = future.drop_duplicates(['query', 'id_site'])
    future_counts = future.groupby('query', sort=True).size()
    candidates = candidates.merge(future.assign(label=1), on=['query', 'id_site'], how='left', validate='one_to_one', sort=False)
    candidates['label'] = candidates.label.fillna(0).astype(int)
    profile_stats = profile_stats.set_index('query')
    candidates['confidence'] = candidates['query'].map(profile_stats.confidence).fillna(0.)
    relevant = candidates.groupby('query', sort=True).label.sum()
    query_stats = queries[['query']].copy()
    query_stats['n_future_all'] = query_stats['query'].map(future_counts).fillna(0).astype(int)
    query_stats['n_relevant'] = query_stats['query'].map(relevant).fillna(0).astype(int)
    query_stats['reachable_fraction'] = [r / f if f else None for r, f in zip(query_stats.n_relevant, query_stats.n_future_all)]
    days = observed.groupby('query', sort=True).size()
    query_stats['history_resource_days'] = query_stats['query'].map(days).fillna(0).astype(int)
    query_stats['preference_confidence'] = query_stats['query'].map(profile_stats.confidence).fillna(0.)
    query_stats['preference_shift_l1'] = query_stats['query'].map(profile_stats['shift']).fillna(0.)
    candidates = candidates.rename(columns={'id_site': 'item', 'activity_type': 'kind'})
    return candidates[COLUMNS], query_stats


def make_candidates_batched(resources, events, frame, indices, day, horizon, courses,
                            max_learners=None, include_no_decay=True, batch_size=256, progress=False, allow_review=False):
    """Same rows and feature formulas as make_candidates_reference, in batches.

    Does not report per-query percentile timings: batch work is measured by
    the caller's stage monitor and amortized throughput, not fictitious latencies.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    cohorts = []
    for role in ROLES:
        cohort = frame.iloc[indices[role]][ENROLLMENT].copy()
        cohort['role'] = role
        cohort['forecast_position'] = np.arange(len(cohort))
        if max_learners:
            cohort = cohort.loc[cohort.id_student.isin(sorted(cohort.id_student.unique())[:max_learners])]
        cohorts.append(cohort)
    cohort = pd.concat(cohorts, ignore_index=True)
    splits = cohort[[*ENROLLMENT, 'role']].copy()
    lengths = {(r.code_module, r.code_presentation): r.module_presentation_length for r in courses.itertuples()}
    complete = [lengths[(r.code_module, r.code_presentation)] >= day + horizon for r in cohort.itertuples()]
    queries = cohort.loc[complete].reset_index(drop=True)
    queries.insert(0, 'query', np.arange(len(queries)))
    if queries.empty:
        return pd.DataFrame(columns=COLUMNS), pd.DataFrame(), splits, []
    mapped = events.merge(queries[[*ENROLLMENT, 'query']], on=ENROLLMENT, how='inner', validate='many_to_one', sort=False)
    mapped['date'] = numeric_column(mapped, 'date')
    if (mapped.date % 1 != 0).any():
        raise ValueError('date must use integer course days')
    # All relevant date validation precedes this filter, matching the reference.
    mapped = mapped.loc[mapped.date < day + horizon]
    event_positions = mapped.groupby('query', sort=False).indices
    fit_keys = frame.iloc[indices['fit']][ENROLLMENT]
    fit = events.loc[(events.date < day) & (events.sum_click > 0)].merge(fit_keys, on=ENROLLMENT, how='inner', validate='many_to_one')
    distinct = fit.drop_duplicates(['id_student', 'code_module', 'code_presentation', 'id_site'])
    popularity = distinct.groupby(['code_module', 'code_presentation', 'id_site']).size().to_dict()
    users = fit_keys.groupby(['code_module', 'code_presentation']).id_student.nunique().to_dict()
    catalogs = {key: group for key, group in resources.groupby(['code_module', 'code_presentation'], sort=False)}
    data_parts, query_parts, completed = [], [], 0
    for (module, presentation), group in queries.groupby(['code_module', 'code_presentation'], sort=False):
        raw_catalog = catalogs[(module, presentation)]
        key = (int(group.iloc[0].id_student), module, presentation)
        eligible, _ = observable_context(raw_catalog, events.iloc[:0], key, day)
        catalog = eligible[['id_site', 'activity_type', 'week_from', 'week_to']].copy()
        catalog['popularity'] = [(popularity.get((module, presentation, int(site)), 0) + 1)
                                  / (users.get((module, presentation), 0) + 2) for site in catalog.id_site]
        metadata = raw_catalog[['id_site', 'activity_type']].copy()
        metadata['id_site'] = pd.to_numeric(metadata.id_site).astype('int64')
        metadata['activity_type'] = metadata.activity_type.astype(str)
        for start in range(0, len(group), batch_size):
            batch_queries = group.iloc[start:start + batch_size]
            positions = [event_positions[q] for q in batch_queries['query'] if q in event_positions]
            batch_events = mapped.iloc[np.concatenate(positions)] if positions else mapped.iloc[:0]
            data, stats = _batch(batch_queries, batch_events, catalog, metadata, day, horizon, include_no_decay, allow_review)
            data_parts.append(data)
            query_parts.append(stats)
            completed += len(batch_queries)
            if progress:
                print(f'  shared features [batched]: {completed}/{len(queries)} queries', flush=True)
    data = pd.concat(data_parts, ignore_index=True).sort_values(['query', 'item'], kind='stable').reset_index(drop=True)
    stats = pd.concat(query_parts, ignore_index=True)
    queries = queries.merge(stats, on='query', how='left', validate='one_to_one', sort=False)
    if queries.reachable_fraction.notna().any():
        queries['reachable_fraction'] = queries.reachable_fraction.astype(float)
    counts = data.groupby('query', sort=False).size()
    sizes = queries['query'].map(counts).fillna(0).astype(int)
    queries['stop'] = sizes.cumsum()
    queries['start'] = queries.stop - sizes
    return data[COLUMNS], queries[QUERY_COLUMNS], splits, []
