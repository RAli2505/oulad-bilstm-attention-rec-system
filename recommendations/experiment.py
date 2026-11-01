"""Learner-disjoint, chronological recommendation experiments on OULAD 2013.

Existing forecast checkpoints are only read. Future clicks are proxy relevance,
not experimental treatment outcomes. No outer-validation labels select models.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .experiment_metrics import (ResourceMonitor, cluster_interval, outcome_metrics,
                                 probability_metrics, ranking_metrics, score_metrics)
from .preferences import PreferenceModel
from .recommendation_agent import RecommendationAgent
from .recommendation_signals import ENROLLMENT

ENSEMBLE = 'ensemble_lstm_attention_lgbm_weighted'
ROLES = ('fit', 'selection', 'calibration', 'held')
FEATURES = ['preference', 'calendar', 'engagement', 'popularity', 'confidence']
METHODS = ['calendar', 'popularity', 'preference', 'no_decay', 'hybrid_no_risk',
           'hybrid_risk', 'hybrid_no_diversity', 'learned_evidence', 'learned_no_preference']


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def sha256(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def source_splits(day_dir):
    """Indices are absolute row positions in the saved cohort manifest."""
    frame = pd.read_csv(day_dir / 'cohort_and_folds.csv')
    if not frame.code_presentation.isin(['2013B', '2013J']).all():
        raise ValueError('Only 2013 training and validation are permitted')
    if frame.duplicated(ENROLLMENT).any():
        raise ValueError('Duplicate enrollment in cohort')
    with np.load(day_dir / 'fold_1/indices.npz', allow_pickle=False) as source:
        if len(source['external_test']):
            raise ValueError('External test must remain disabled')
        indices = {role: source['validation' if role == 'selection' else role].copy() for role in ROLES}
    if len(np.unique(np.concatenate(list(indices.values())))) != sum(map(len, indices.values())):
        raise ValueError('Overlapping split indices')
    groups = {role: set(frame.iloc[ids].id_student) for role, ids in indices.items()}
    for i, a in enumerate(ROLES):
        for b in ROLES[i + 1:]:
            if groups[a] & groups[b]:
                raise ValueError(f'Learners overlap between {a} and {b}')
    return frame, indices


def load_events(raw, keys, last_day):
    """Read logs once, retain selected 2013 enrollments and required follow-up."""
    parts = []
    for chunk in pd.read_csv(raw / 'studentVle.csv', usecols=ENROLLMENT + ['id_site', 'date', 'sum_click'], chunksize=500_000):
        chunk = chunk.loc[chunk.code_presentation.isin(['2013B', '2013J']) & (chunk.date < last_day)]
        part = chunk.merge(keys, on=ENROLLMENT, how='inner', validate='many_to_one')
        if len(part):
            parts.append(part)
    if not parts:
        return pd.DataFrame(columns=ENROLLMENT + ['id_site', 'date', 'sum_click'])
    events = pd.concat(parts, ignore_index=True)
    if not np.isfinite(events[['date', 'sum_click']].to_numpy(dtype=float)).all() or (events.sum_click < 0).any():
        raise ValueError('Invalid observed interactions')
    return events


def reconstructed_risk(day_dir, seed, indices):
    """Reconstruct fit ensemble scores from saved component fit scores.

    Apply the already fitted original calibration without refitting it.
    Held forecasts are checked against the published CSV for seed-42 runs.
    """
    seed_dir = day_dir / f'fold_1/seed_{seed}'
    model_dir = seed_dir / ENSEMBLE
    config = json.loads((model_dir / 'training.json').read_text(encoding='utf-8'))
    calibrator = joblib.load(model_dir / 'calibration.joblib')
    with np.load(model_dir / 'predictions.npz', allow_pickle=False) as saved:
        scores = {role: saved[key].copy() for role, key in
                  [('selection', 'val_score'), ('calibration', 'calibration_score'), ('held', 'held_score')]}
        held = saved['held_probability'].copy()
    member_fit = []
    for member in config['members']:
        with np.load(seed_dir / member / 'predictions.npz', allow_pickle=False) as saved:
            member_fit.append(expit(saved['fit_score']))
    scores['fit'] = logit(np.clip(np.column_stack(member_fit) @ np.asarray(config['weights']), 1e-6, 1 - 1e-6))
    result = {role: calibrator.predict(scores[role]) for role in ROLES}
    if not np.allclose(result['held'], held, rtol=1e-12, atol=1e-12):
        raise ValueError('Reconstructed probabilities disagree with saved ensemble')
    if any(len(result[r]) != len(indices[r]) for r in ROLES):
        raise ValueError('Forecast lengths disagree with saved split indices')
    return result, {r: expit(scores[r]) for r in ROLES}, config


def make_candidates(resources, events, frame, indices, day, horizon, courses, max_learners=None, include_no_decay=True,
                    feature_engine='batched', feature_batch_size=256, progress=False):
    if feature_engine == 'batched':
        from .batch_features import make_candidates_batched
        return make_candidates_batched(resources, events, frame, indices, day, horizon, courses,
                                       max_learners, include_no_decay, feature_batch_size, progress)
    if feature_engine != 'reference':
        raise ValueError('Unknown feature engine')
    return make_candidates_reference(resources, events, frame, indices, day, horizon, courses, max_learners, include_no_decay)


def make_candidates_reference(resources, events, frame, indices, day, horizon, courses, max_learners=None, include_no_decay=True):
    """One shared candidate universe and label vector for every method/seed."""
    event_groups = events.groupby(ENROLLMENT, sort=False).indices
    catalog_groups = {key: group for key, group in resources.groupby(['code_module', 'code_presentation'], sort=False)}
    lengths = {(r.code_module, r.code_presentation): r.module_presentation_length for r in courses.itertuples()}
    fit_keys = frame.iloc[indices['fit']][ENROLLMENT]
    fit_events = events.loc[(events.date < day) & (events.sum_click > 0)].merge(
        fit_keys, on=ENROLLMENT, how='inner', validate='many_to_one')
    distinct = fit_events.drop_duplicates(['id_student', 'code_module', 'code_presentation', 'id_site'])
    counts = distinct.groupby(['code_module', 'code_presentation', 'id_site']).size().to_dict()
    users = frame.iloc[indices['fit']].groupby(['code_module', 'code_presentation']).id_student.nunique().to_dict()
    records, queries, latencies, split_records = [], [], [], []
    query_id = 0
    for role in ROLES:
        cohort = frame.iloc[indices[role]].copy()
        cohort['forecast_position'] = np.arange(len(cohort))
        if max_learners:
            chosen = sorted(cohort.id_student.unique())[:max_learners]
            cohort = cohort.loc[cohort.id_student.isin(chosen)]
        for row in cohort.itertuples():
            key = (row.id_student, row.code_module, row.code_presentation)
            split_records.append(dict(zip(ENROLLMENT, key), role=role))
            if lengths[(row.code_module, row.code_presentation)] < day + horizon:
                continue  # predeclared complete-follow-up cohort, not outcome-based
            observed = events.iloc[event_groups[key]] if key in event_groups else events.iloc[:0]
            catalog = catalog_groups[(row.code_module, row.code_presentation)]
            started = time.perf_counter()
            agent = RecommendationAgent(catalog, observed)
            snapshot = agent.recommend(key, day, .5, n=1, include_candidates=True)
            static = (PreferenceModel(half_life_days=1e12).profile(observed, catalog, key, day)
                      if include_no_decay else None)
            candidates = snapshot['candidates']
            future = set(observed.loc[(observed.date >= day) & (observed.date < day + horizon) & (observed.sum_click > 0), 'id_site'].astype(int))
            eligible = {c['item_id'] for c in candidates}
            available_truth = future & eligible
            offset = len(records)
            for candidate in candidates:
                item = candidate['item_id']
                signals = candidate['signals']
                count = counts.get((row.code_module, row.code_presentation, item), 0)
                popularity = (count + 1) / (users.get((row.code_module, row.code_presentation), 0) + 2)
                records.append({'query': query_id, 'item': item, 'kind': candidate['activity_type'],
                                'label': int(item in available_truth), **signals,
                                'popularity': popularity, 'confidence': snapshot['profile']['confidence'],
                                'no_decay': static['preferences'].get(candidate['activity_type'], 0.) if static else signals['preference']})
            queries.append({'query': query_id, **dict(zip(ENROLLMENT, key)), 'role': role,
                            'forecast_position': int(row.forecast_position), 'start': offset, 'stop': len(records),
                            'n_future_all': len(future), 'n_relevant': len(available_truth),
                            'reachable_fraction': len(available_truth) / len(future) if future else None,
                            'history_resource_days': snapshot['profile']['n_resource_days'],
                            'preference_confidence': snapshot['profile']['confidence'],
                            'preference_shift_l1': float(sum(abs(v) for v in snapshot['profile']['preference_shift'].values()))})
            latencies.append(time.perf_counter() - started)
            query_id += 1
    columns = ['query', 'item', 'kind', 'label', *FEATURES, 'no_decay']
    return pd.DataFrame(records, columns=columns), pd.DataFrame(queries), pd.DataFrame(split_records), latencies


def rank_indices(scores, items, kinds, k, penalty=0.):
    remaining = np.arange(len(scores))
    chosen, seen = [], set()
    while len(remaining) and len(chosen) < k:
        values = scores[remaining] - penalty * np.array([kind in seen for kind in kinds[remaining]])
        at = np.lexsort((items[remaining], -values))[0]
        index = remaining[at]
        chosen.append(int(index))
        seen.add(kinds[index])
        remaining = np.delete(remaining, at)
    return chosen


def selection_ndcg(queries, labels, scores, items, kinds):
    values = []
    for q in queries.itertuples():
        if q.role != 'selection':
            continue
        sl = slice(q.start, q.stop)
        order = rank_indices(scores[sl], items[sl], kinds[sl], 10)
        values.append(ranking_metrics(items[sl][order].tolist(), set(items[sl][labels[sl] == 1]), 10)['ndcg'])
    if not values:
        raise ValueError('Recommendation selection split has no queries')
    return float(np.mean(values))


def fit_ranker(data, queries, features, seed, directory):
    x, y = data[features].to_numpy(dtype=float), data.label.to_numpy(dtype=int)
    roles = queries.set_index('query').role
    row_roles = data['query'].map(roles).to_numpy()
    fit = row_roles == 'fit'
    if not fit.any() or len(np.unique(y[fit])) != 2:
        raise ValueError('Training candidates must contain clicked and unobserved items; enlarge the pilot')
    trials, best = [], None
    for c in [.01, .1, 1., 10.]:
        started = time.perf_counter()
        model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=500, random_state=seed))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', ConvergenceWarning)
            model.fit(x[fit], y[fit])
        p = model.predict_proba(x)[:, 1]
        ndcg = selection_ndcg(queries, y, p, data.item.to_numpy(), data.kind.to_numpy())
        trial = {'C': c, 'selection_ndcg_at_10': ndcg, 'seconds': time.perf_counter() - started,
                 'iterations': model[-1].n_iter_.tolist(), 'warnings': [str(w.message) for w in caught]}
        trials.append(trial)
        if best is None or ndcg > best[0]:
            best = (ndcg, model, p, c)
    _, model, p, c = best
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, directory / 'ranker.joblib')
    write_json(directory / 'weights.json', {'features': features, 'standardized_coefficients': model[-1].coef_[0].tolist(),
                                         'intercept': float(model[-1].intercept_[0]), 'scaler_mean': model[0].mean_.tolist(),
                                         'scaler_scale': model[0].scale_.tolist(), 'selected_C': c,
                                         'coefficients_are_not_convex_mixture_weights': True})
    pd.DataFrame(trials).to_csv(directory / 'inner_search.csv', index=False)
    return p


def calibrate_relevance(score, labels, calibration_mask, directory):
    """Fixed one-dimensional Platt map; fitted solely on calibration learners."""
    x = np.asarray(score)[:, None]
    y = np.asarray(labels)
    if not calibration_mask.any() or len(np.unique(y[calibration_mask])) < 2:
        raise ValueError('Independent calibration needs both click-proxy classes')
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1., max_iter=500))
    model.fit(x[calibration_mask], y[calibration_mask])
    if model[-1].coef_[0, 0] < 0:
        model[-1].coef_[0, 0] = 0.
        model[-1].intercept_[0] = logit(np.clip(y[calibration_mask].mean(), 1e-6, 1 - 1e-6))
    joblib.dump(model, directory / 'relevance_calibration.joblib')
    write_json(directory / 'relevance_calibration.json', {'fit_split': 'calibration',
               'coefficient': float(model[-1].coef_[0, 0]), 'intercept': float(model[-1].intercept_[0]),
               'mean': float(model[0].mean_[0]), 'scale': float(model[0].scale_[0]),
               'monotone_nonnegative': True, 'unclicked_is_proxy_negative': True})
    return model.predict_proba(x)[:, 1]


def evaluate_method(name, score, probability, data, queries, ks, repeats, seed, directory, previous):
    labels, items, kinds = data.label.to_numpy(), data.item.to_numpy(), data.kind.to_numpy()
    popularity = data.popularity.to_numpy()
    rows, runtimes, exposure, held_predictions = [], [], Counter(), []
    penalty = .15 if name in ['hybrid_no_risk', 'hybrid_risk'] else 0.
    for q in queries.itertuples():
        sl = slice(q.start, q.stop)
        started = time.perf_counter()
        order = rank_indices(score[sl], items[sl], kinds[sl], max(ks), penalty)
        latency = time.perf_counter() - started
        ranked = items[sl][order].tolist()
        truth = set(items[sl][labels[sl] == 1])
        key = (seed, name, q.id_student, q.code_module, q.code_presentation)
        old = previous.get(key)
        if q.role == 'held':
            previous[key] = set(ranked[:10])
        for k in ks:
            selected = ranked[:k]
            metrics = ranking_metrics(ranked, truth, k)
            selected_kinds = kinds[sl][order][:k]
            diversity = float(np.mean([a != b for i, a in enumerate(selected_kinds) for b in selected_kinds[i + 1:]])) if len(selected_kinds) > 1 else 0.
            novelty = float(np.mean(-np.log2(popularity[sl][order][:k]))) if selected else None
            rows.append({'query': q.query, 'id_student': q.id_student, 'role': q.role, 'method': name, 'k': k,
                         **metrics, 'type_diversity': diversity, 'novelty_bits': novelty,
                         'recall_all_future': metrics['hits'] / q.n_future_all if q.n_future_all else 0.,
                         'history_resource_days': q.history_resource_days,
                         'previous_landmark_jaccard_at_10': len(old & set(ranked[:10])) / len(old | set(ranked[:10])) if old is not None and old | set(ranked[:10]) else None})
        if q.role == 'held':
            exposure.update(ranked[:10])
            held_predictions.append({'query': q.query, 'items': ranked, 'scores': score[sl][order].tolist(),
                                     'probability': probability[sl][order].tolist()})
        runtimes.append(latency)
    row_frame = pd.DataFrame(rows)
    row_frame.to_csv(directory / 'query_metrics.csv', index=False)
    write_json(directory / 'top_recommendations_held.json', held_predictions)
    summary = []
    numeric = ['precision', 'recall', 'f1', 'ndcg', 'map', 'mrr', 'hit_rate', 'fill_rate',
               'type_diversity', 'novelty_bits', 'recall_all_future']
    for (role, k), part in row_frame.groupby(['role', 'k'], sort=False):
        values = {metric: float(part[metric].mean()) if part[metric].notna().any() else None for metric in numeric}
        positive = part.loc[part.n_relevant > 0]
        summary.append({'role': role, 'method': name, 'k': int(k), 'n_queries': len(part),
                        'positive_query_fraction': float((part.n_relevant > 0).mean()), **values,
                        'ndcg_conditional_positive': float(positive.ndcg.mean()) if len(positive) else None,
                        'recall_conditional_positive': float(positive.recall.mean()) if len(positive) else None})
    held = row_frame.loc[(row_frame.role == 'held') & (row_frame.k == 10)]
    uncertainty = {metric: cluster_interval(held[metric], held.id_student, repeats, seed)
                   for metric in ['ndcg', 'recall', 'precision', 'map']} if len(held) else {}
    candidate_universe = set(data.loc[data['query'].isin(queries.loc[queries.role == 'held', 'query']), 'item'])
    counts = np.array([exposure[item] for item in sorted(candidate_universe)], dtype=float)
    gini = float(np.sum((2 * np.arange(1, len(counts) + 1) - len(counts) - 1) * np.sort(counts)) / (len(counts) * counts.sum())) if counts.sum() else None
    write_json(directory / 'coverage_uncertainty_runtime.json', {
        'held_catalog_coverage_at_10': len(exposure) / len(candidate_universe) if candidate_universe else None,
        'held_exposure_gini_at_10': gini, 'cluster_bootstrap_at_10': uncertainty,
        'ranking_latency_seconds': {'p50': float(np.quantile(runtimes, .5)), 'p95': float(np.quantile(runtimes, .95)),
                                    'p99': float(np.quantile(runtimes, .99)), 'total': sum(runtimes)},
        'diversity_penalty': penalty, 'latency_excludes_shared_feature_construction': True})
    return summary, row_frame


def run_recommendations(risk_run, output, windows, seeds, *, horizon=7, ks=(5, 10, 20), bootstrap=1000, threads=6, max_learners=None, raw=None, methods=None,
                        feature_engine='batched', feature_batch_size=256):
    risk_run, output = Path(risk_run).resolve(), Path(output).resolve()
    methods = list(METHODS) if methods is None else list(methods)
    if not methods or len(methods) != len(set(methods)) or set(methods) - set(METHODS):
        raise ValueError('Select unique known recommendation methods')
    has_contrast = {'hybrid_risk', 'hybrid_no_risk'}.issubset(methods)
    if feature_engine not in ['batched', 'reference'] or feature_batch_size < 1:
        raise ValueError('Use batched/reference feature engine and positive batch size')
    manifest = json.loads((risk_run / 'run_manifest.json').read_text(encoding='utf-8'))
    if manifest.get('status') != 'completed':
        raise ValueError('Source forecast run must be completed')
    if output == risk_run or output.is_relative_to(risk_run):
        raise ValueError('Write recommendations outside the source forecast run')
    if manifest['config'].get('evaluate_test') or manifest['config'].get('split_scheme') != 'holdout':
        raise ValueError('Need a 2013-only grouped holdout source run')
    if not set(seeds).issubset(manifest['config']['seeds']) or not set(windows).issubset(manifest['config']['windows']):
        raise ValueError('Requested windows/seeds are missing from source run')
    if horizon < 1 or bootstrap < 1 or min(ks) < 1 or 10 not in ks or threads < 1:
        raise ValueError('Positive horizon, K, threads and bootstrap required; include K=10')
    root = Path(__file__).resolve().parents[1]
    raw = Path(raw).resolve() if raw is not None else root / 'data/oulad/raw'
    verified_raw = {}
    for name in ['vle.csv', 'courses.csv', 'studentVle.csv']:
        digest = sha256(raw / name)
        if digest != manifest.get('data_sha256', {}).get(name):
            raise ValueError(f'{name} differs from the forecast source data')
        verified_raw[name] = digest
    output.mkdir(parents=True, exist_ok=False)
    config = {'risk_source': str(risk_run), 'windows': windows, 'seeds': seeds, 'horizon': horizon,
              'ks': list(ks), 'bootstrap': bootstrap, 'threads': threads, 'max_learners_per_split': max_learners,
              'pilot_not_for_publication': max_learners is not None, 'external_test_used': False,
              'primary_day': 56, 'primary_metric': 'NDCG@10',
              'methods': methods, 'primary_contrast': 'hybrid_risk minus hybrid_no_risk' if has_contrast else None,
              'feature_engine': feature_engine, 'feature_batch_size': feature_batch_size,
              'source_manifest_sha256': sha256(risk_run / 'run_manifest.json'),
              'raw_source': str(raw), 'verified_raw_sha256': verified_raw,
              'source_code_sha256': {str(p.relative_to(root)): sha256(p) for p in [*Path(__file__).parent.glob('*.py'), root / 'run_system_experiment.py']},
              'limitations': ['future_click_is_proxy_relevance', 'unclicked_is_not_known_irrelevant',
                             'no_causal_learning_effect', 'validation_already_explored',
                             'no_negative_sampling; candidate-level pooled probabilities are not query-balanced']}
    write_json(output / 'manifest.json', {**config, 'status': 'running'})
    all_keys = pd.concat([pd.read_csv(risk_run / f'day_{d}/cohort_and_folds.csv', usecols=ENROLLMENT) for d in windows]).drop_duplicates()
    resources, courses = pd.read_csv(raw / 'vle.csv'), pd.read_csv(raw / 'courses.csv')
    with ResourceMonitor() as total_monitor, threadpool_limits(limits=threads):
        events = load_events(raw, all_keys, max(windows) + horizon)
        previous, all_summary, risk_rows, probability_rows, bins_rows, gap_rows, runtime_rows = {}, [], [], [], [], [], []
        try:
            for day in windows:
                print(f'Recommendation day {day}: shared features', flush=True)
                day_dir = risk_run / f'day_{day}'
                destination = output / f'day_{day}'
                destination.mkdir()
                frame, indices = source_splits(day_dir)
                with ResourceMonitor() as features_monitor:
                    data, queries, splits, latency = make_candidates(resources, events, frame, indices, day, horizon, courses,
                                                                      max_learners, include_no_decay='no_decay' in methods,
                                                                      feature_engine=feature_engine, feature_batch_size=feature_batch_size, progress=True)
                if not len(data) or any(not (queries.role == role).any() for role in ROLES):
                    raise ValueError('Each split needs eligible recommendation queries')
                data.to_parquet(destination / 'candidate_features_and_labels.parquet', index=False)
                queries.to_csv(destination / 'queries.csv', index=False)
                splits.to_csv(destination / 'split_membership.csv', index=False)
                write_json(destination / 'feature_runtime.json', {**features_monitor.result,
                           'query_latency_p50': float(np.quantile(latency, .5)) if latency else None,
                           'query_latency_p95': float(np.quantile(latency, .95)) if latency else None,
                           'amortized_seconds_per_query': features_monitor.result['wall_seconds'] / len(queries),
                           'feature_engine': feature_engine, 'feature_batch_size': feature_batch_size,
                           'independent_per_query_feature_latencies_measured': bool(latency),
                           'n_queries': len(queries), 'n_candidate_pairs': len(data), 'future_horizon': horizon})
                roles = data['query'].map(queries.set_index('query').role).to_numpy()
                x = data[FEATURES].to_numpy(dtype=float)
                y = data.label.to_numpy(dtype=int)
                for seed in seeds:
                    seed_dir = destination / f'seed_{seed}'
                    seed_dir.mkdir()
                    risk, risk_raw, ensemble_config = reconstructed_risk(day_dir, seed, indices)
                    write_json(seed_dir / 'original_ensemble_weights.json', ensemble_config)
                    for role in ROLES:
                        target = frame.iloc[indices[role]].target.to_numpy()
                        for stage, p in [('raw', risk_raw[role]), ('calibrated', risk[role])]:
                            for endpoint, values in outcome_metrics(target, p).items():
                                risk_rows.append({'day': day, 'seed': seed, 'role': role, 'endpoint': endpoint, 'stage': stage, **values})
                                _, bins = probability_metrics(target if endpoint == 'risk' else 1 - target, p if endpoint == 'risk' else 1 - p)
                                bins_rows.extend({'task': 'course_outcome', 'day': day, 'seed': seed, 'role': role, 'endpoint': endpoint, 'stage': stage, **b} for b in bins)
                    query_risk = {q.query: risk[q.role][q.forecast_position] for q in queries.itertuples()}
                    query_weights = queries[['query', *ENROLLMENT, 'role']].copy()
                    query_weights['forecast_probability'] = query_weights['query'].map(query_risk)
                    query_weights['preference_weight'] = .5
                    query_weights['calendar_weight'] = .2 + .2 * query_weights.forecast_probability
                    query_weights['engagement_weight'] = .3 - .2 * query_weights.forecast_probability
                    query_weights.to_csv(seed_dir / 'hybrid_weights.csv', index=False)
                    r = data['query'].map(query_risk).to_numpy(dtype=float)
                    fixed = .5 * x[:, 0] + .3 * x[:, 1] + .2 * x[:, 2]
                    hybrid = .5 * x[:, 0] + (.2 + .2 * r) * x[:, 1] + (.3 - .2 * r) * x[:, 2]
                    scores = {'calendar': x[:, 1], 'popularity': x[:, 3], 'preference': x[:, 0],
                              'no_decay': data.no_decay.to_numpy(), 'hybrid_no_risk': fixed,
                              'hybrid_risk': hybrid, 'hybrid_no_diversity': hybrid}
                    method_frames, saved_predictions = {}, {}
                    for name in methods:
                        method_dir = seed_dir / name
                        method_dir.mkdir()
                        with ResourceMonitor() as method_monitor:
                            if name.startswith('learned'):
                                columns = FEATURES if name == 'learned_evidence' else ['calendar', 'engagement', 'popularity']
                                scores[name] = fit_ranker(data, queries, columns, seed, method_dir)
                            score = scores[name]
                            probability = calibrate_relevance(score, y, roles == 'calibration', method_dir)
                            for role in ROLES:
                                mask = roles == role
                                if not mask.any():
                                    continue
                                for stage, p in [('calibrated', probability)] + ([('raw', score)] if name.startswith('learned') else []):
                                    values, bins = probability_metrics(y[mask], p[mask])
                                    probability_rows.append({'day': day, 'seed': seed, 'method': name, 'role': role, 'stage': stage, **values})
                                    bins_rows.extend({'task': 'resource_interaction', 'day': day, 'seed': seed, 'method': name, 'role': role, 'stage': stage, **b} for b in bins)
                                if not name.startswith('learned'):
                                    probability_rows.append({'day': day, 'seed': seed, 'method': name, 'role': role, 'stage': 'uncalibrated_utility',
                                                             **score_metrics(y[mask], score[mask]), 'brier': None, 'log_loss': None, 'ece': None})
                            summary, query_metrics = evaluate_method(name, score, probability, data, queries, ks, bootstrap, seed, method_dir, previous)
                            all_summary.extend({'day': day, 'seed': seed, **s} for s in summary)
                            method_frames[name] = query_metrics
                            saved_predictions[name + '_score'] = score[roles == 'held']
                            saved_predictions[name + '_probability'] = probability[roles == 'held']
                        runtime_rows.append({'day': day, 'seed': seed, 'method': name, **method_monitor.result,
                                             'gpu_peak_allocated_bytes': 0, 'gpu_used': False})
                        for k in ks:
                            by_role = {s['role']: s for s in summary if s['k'] == k}
                            for metric in ['ndcg', 'recall', 'precision', 'map']:
                                gap_rows.append({'day': day, 'seed': seed, 'method': name, 'k': k, 'metric': metric,
                                                 'fit_minus_selection': by_role['fit'][metric] - by_role['selection'][metric],
                                                 'fit_minus_held': by_role['fit'][metric] - by_role['held'][metric],
                                                 'note': 'descriptive generalization gap; not an independent significance test'})
                    np.savez_compressed(seed_dir / 'candidate_predictions_held.npz', labels=y[roles == 'held'], **saved_predictions)
                    if has_contrast:
                        a, b = [method_frames[m].loc[(method_frames[m].role == 'held') & (method_frames[m].k == 10)].set_index('query') for m in ['hybrid_risk', 'hybrid_no_risk']]
                        delta = a.ndcg - b.ndcg
                        write_json(seed_dir / 'paired_primary_contrast.json', {'contrast': 'hybrid_risk minus hybrid_no_risk',
                                   'metric': 'NDCG@10', 'day': day, 'primary_day': day == 56,
                                   **cluster_interval(delta.to_numpy(), a.id_student.to_numpy(), bootstrap, seed),
                                   'other_windows_exploratory': True})
                    print(f'Recommendation day {day}, seed {seed}: complete', flush=True)
                pd.DataFrame(all_summary).to_csv(output / 'ranking_summary.csv', index=False)
                pd.DataFrame(risk_rows).to_csv(output / 'course_outcome_metrics.csv', index=False)
                pd.DataFrame(probability_rows).to_csv(output / 'recommendation_probability_metrics.csv', index=False)
                pd.DataFrame(bins_rows).to_csv(output / 'reliability_bins.csv', index=False)
                pd.DataFrame(gap_rows).to_csv(output / 'generalization_gaps.csv', index=False)
                pd.DataFrame(runtime_rows).to_csv(output / 'runtime_memory.csv', index=False)
            summary = pd.DataFrame(all_summary)
            metrics = ['precision', 'recall', 'f1', 'ndcg', 'map', 'mrr', 'hit_rate', 'type_diversity', 'novelty_bits']
            summary.groupby(['day', 'role', 'method', 'k'])[metrics].agg(['mean', 'std']).to_csv(output / 'across_seeds.csv')
            outcome = pd.DataFrame(risk_rows)
            outcome_gaps = []
            for keys, part in outcome.groupby(['day', 'seed', 'endpoint', 'stage']):
                by_role = part.set_index('role')
                for metric in ['roc_auc', 'pr_auc', 'f1', 'brier', 'log_loss', 'ece']:
                    outcome_gaps.append(dict(zip(['day', 'seed', 'endpoint', 'stage'], keys), metric=metric,
                        fit_minus_selection=float(by_role.loc['fit', metric] - by_role.loc['selection', metric]) if pd.notna(by_role.loc['fit', metric]) and pd.notna(by_role.loc['selection', metric]) else None,
                        fit_minus_held=float(by_role.loc['fit', metric] - by_role.loc['held', metric]) if pd.notna(by_role.loc['fit', metric]) and pd.notna(by_role.loc['held', metric]) else None))
            pd.DataFrame(outcome_gaps).to_csv(output / 'forecast_generalization_gaps.csv', index=False)
            # Full prediction-stage diagnostics are linked, not recomputed by retraining.
            write_json(output / 'forecast_diagnostic_sources.json', {'root': str(risk_run),
                       'paths': [f'day_{d}/{name}' for d in windows for name in ['fit_diagnostics.csv', 'calibration_before_after.csv', 'paired_comparisons.csv', 'fold_metrics.csv']],
                       'learning_curves': 'day_N/fold_1/seed_S/MODEL/learning_curve.csv',
                       'gpu_memory_and_parameters': 'day_N/fold_1/seed_S/MODEL/training.json'})
            write_report(output, summary, outcome)
        except BaseException as exc:
            write_json(output / 'manifest.json', {**config, 'status': 'failed', 'error': repr(exc)})
            raise
    write_json(output / 'manifest.json', {**config, 'status': 'completed', **total_monitor.result,
               'disk_bytes': sum(p.stat().st_size for p in output.rglob('*') if p.is_file())})
    return output


def write_report(output, summary, outcome):
    """Standalone human report and curves; no modification of source artifacts."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plots = output / 'figures'
    plots.mkdir()
    held = summary.loc[(summary.role == 'held') & (summary.k == 10)]
    for metric in ['ndcg', 'precision', 'recall']:
        fig, ax = plt.subplots(figsize=(9, 5))
        for method, group in held.groupby('method'):
            values = group.groupby('day')[metric].mean()
            ax.plot(values.index, values.values, marker='o', label=method)
        ax.set(xlabel='Observation day', ylabel=metric.upper() + '@10', title='2013 held validation; mean over seeds')
        ax.legend(fontsize=7, loc='best')
        fig.tight_layout()
        fig.savefig(plots / f'{metric}_at_10.png', dpi=160)
        plt.close(fig)
    for endpoint in ['risk', 'success']:
        fig, ax = plt.subplots(figsize=(8, 4))
        subset = outcome.loc[(outcome.role == 'held') & (outcome.endpoint == endpoint) & (outcome.stage == 'calibrated')]
        for metric in ['roc_auc', 'pr_auc', 'precision', 'recall', 'f1']:
            values = subset.groupby('day')[metric].mean()
            ax.plot(values.index, values.values, marker='o', label=metric)
        ax.set(xlabel='Observation day', ylabel='Metric', title=f'Course {endpoint}: fixed threshold 0.5')
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots / f'course_{endpoint}.png', dpi=160)
        plt.close(fig)
    config = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    text = ['# OULAD: обучение и validation гибридной системы', '',
            'Test 2014 не использовался. Все результаты относятся к 2013 году.',
            ('Основная точка — 56 дней; основной рекомендательный контраст — hybrid_risk против hybrid_no_risk по NDCG@10.'
             if config['primary_contrast'] else 'Основная точка — 56 дней; оценивается только система, без сравнительных моделей и абляций.'),
            'Сводки по другим окнам и метрикам исследовательские; опубликованная validation уже просматривалась.', '',
            '## Прогноз завершения курса', '',
            'risk: Fail/Withdrawn; success: Pass/Distinction. Success probability = 1 − risk probability.',
            'PR-AUC обозначает Average Precision; trapezoidal PR-AUC сохранён отдельно.',
            'Precision/Recall/F1 считаются при заранее фиксированном пороге 0.5.', '',
            '| День | Класс | ROC-AUC | PR-AUC | Precision | Recall | F1 | Brier |',
            '|---:|---|---:|---:|---:|---:|---:|---:|']
    metrics = ['roc_auc', 'pr_auc', 'precision', 'recall', 'f1', 'brier']
    table = outcome.loc[(outcome.role == 'held') & (outcome.stage == 'calibrated')].groupby(['day', 'endpoint'])[metrics].mean()
    for (day, endpoint), row in table.iterrows():
        text.append(f'| {day} | {endpoint} | ' + ' | '.join(f'{row[m]:.4f}' for m in metrics) + ' |')
    text += ['', '## Рекомендации', '',
             'Метрики @5/@10/@20 сохранены в ranking_summary.csv. Среднее включает запросы без позитивных будущих кликов (для них top-K метрики равны 0); условные метрики приведены отдельно.',
             'Релевантность — положительный клик в течение 7 дней после среза по доступным кандидатам. Точный горизонт см. manifest.json.',
             'Recall по всем будущим ресурсам и доля достижимых позитивов сохранены отдельно от Recall по допустимым кандидатам.',
             'PR-AUC для взаимодействия считается по полному набору кандидатов без семплирования отрицательных примеров.',
             'Рекомендационный score не считается вероятностью: Brier/ECE/Log loss для эвристик оцениваются только после отдельной Platt-калибровки на calibration.',
             'Функция Platt не меняет ранжирование. Для learned_evidence дополнительно сохранены некалиброванные вероятности логистической модели.', '',
             '## Диагностика и артефакты', '',
             '- generalization_gaps.csv и forecast_generalization_gaps.csv: описательные train–selection/held разрывы.',
             '- recommendation_probability_metrics.csv, course_outcome_metrics.csv, reliability_bins.csv: калибровка, ошибки и матрицы ошибок.',
             '- seed_*/hybrid_weights.csv: фактические веса предпочтений, календаря и поведения для каждого запроса.',
             '- seed_*/METHOD/weights.json и inner_search.csv создаются только при отдельном запуске дополнительных обучаемых ранжировщиков.',
             '- original_ensemble_weights.json: исходные веса ансамбля; рекалибровка ансамбля не выполнялась.',
             '- candidate_features_and_labels.parquet, queries.csv, split_membership.csv: воспроизводимые кандидаты, метки и разбиения.',
             '- candidate_predictions_held.npz: полные кандидатные validation-прогнозы; порядок совпадает с held-строками общего feature-файла.',
             '- coverage_uncertainty_runtime.json: охват, Gini экспозиции, novelty, type diversity, learner-cluster bootstrap CI и latency.',
             '- runtime_memory.csv: wall/CPU time, выборочный peak RSS и общий OS lifetime peak; эти два вида peak памяти не смешиваются.',
             '- Ссылки на нейронные learning curves, GPU-память, число параметров и train gaps: forecast_diagnostic_sources.json.',
             '- Кандидатные признаки вычисляются один раз на окно и переиспользуются для всех seed и методов.', '',
             '## Ограничения', '',
             'Ненаблюдавшийся клик — proxy negative, а не доказанная нерелевантность. Нет логов показов и рандомизированных вмешательств.',
             'Оценка не доказывает улучшение обучения. Веса выбираются только на selection, калибровка — только на calibration.',
             'CI по студентам описывают фиксированный запуск; устойчивость обучения по seed представлена отдельно в across_seeds.csv.',
             ('Основной risk-контраст и его CI находятся в paired_primary_contrast.json; остальные сравнения не являются подтверждающими тестами.'
              if config['primary_contrast'] else 'Сравнительные модели не запускались; парный контраст с отсутствующей базовой моделью не вычисляется.'),
             'Рекомендационные модели обучаются на CPU; GPU-память равна 0. Нейросетевые ресурсы измерены отдельным процессом прогнозирования.']
    (output / 'REPORT_RU.md').write_text('\n'.join(text) + '\n', encoding='utf-8')
