"""Full daily-basket OULAD agent experiment; the forecast source is read-only.

Observed future clicks are proxy relevance. This is exploratory validation,
not a causal test, and the source risk ensemble is never fitted here.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .experiment import (ROLES, load_events, make_candidates, rank_indices,
                         reconstructed_risk, sha256, source_splits, write_json)
from .experiment_metrics import ResourceMonitor, cluster_interval, probability_metrics, ranking_metrics
from .recommendation_signals import ENROLLMENT
from .checkpoints import complete_job, save_job_checkpoint


def validate_source(risk_run, output, windows, seeds, raw):
    """Validate provenance before creating output or loading any model."""
    risk_run, output, raw = map(lambda p: Path(p).resolve(), (risk_run, output, raw))
    manifest = json.loads((risk_run / 'run_manifest.json').read_text(encoding='utf-8'))
    config = manifest['config']
    if manifest.get('status') != 'completed':
        raise ValueError('Source forecast run must be completed')
    if config.get('evaluate_test') or config.get('split_scheme') != 'holdout':
        raise ValueError('Need a 2013-only grouped holdout source run')
    if output == risk_run or output.is_relative_to(risk_run) or risk_run.is_relative_to(output):
        raise ValueError('Output must be separate from the frozen source run')
    if not set(windows).issubset(config['windows']) or not set(seeds).issubset(config['seeds']):
        raise ValueError('Requested windows/seeds are absent from source')
    hashes = {name: sha256(raw / name) for name in ('vle.csv', 'courses.csv', 'studentVle.csv')}
    if any(value != manifest.get('data_sha256', {}).get(name) for name, value in hashes.items()):
        raise ValueError('Raw data differs from the source forecast data')
    return hashes


def replay_baseline(data, query, history, day, risk, k=20):
    """Replay the original rule, keeping full-universe relevance unchanged."""
    scores = (.5 * data.preference.to_numpy() + (.2 + .2 * risk) * data.calendar.to_numpy()
              + (.3 - .2 * risk) * data.engagement.to_numpy())
    recent = set(history.loc[(history.date < day) & (history.date >= day - 7) &
                             (history.sum_click > 0), 'id_site'].astype(int))
    eligible = np.flatnonzero(~data.item.isin(recent).to_numpy())
    order = rank_indices(scores[eligible], data.item.to_numpy()[eligible],
                         data.kind.to_numpy()[eligible], k, penalty=.15)
    return data.item.to_numpy()[eligible][order].tolist()


def _rank(agent, snapshot, k):
    result = agent.rank_snapshot(snapshot, n=k)
    return [int(row['item_id']) for row in result['recommendations']]


def compact_rank(items, kinds, scores, affinities, resource_days, k, mmr_lambda=.8, sasrec_weight=.05):
    """Exact array replay of facade MMR and its bounded activity SAS selector.

    OULAD candidates have activity types and no explicit topic tags. Equal MMR
    utilities prefer smaller item IDs. SAS ties prefer the original slot then
    score/item order, exactly as the facade's stable max over sorted candidates.
    """
    items, kinds = np.asarray(items), np.asarray(kinds)
    scores, affinities = np.asarray(scores), np.asarray(affinities)
    if not len(items):
        return []
    span = float(np.ptp(scores))
    normalized = (scores - scores.min()) / span if span > 1e-12 else np.zeros(len(scores))
    pool = np.lexsort((items, -scores))
    selected, remaining, seen = [], np.ones(len(items), dtype=bool), set()
    for _ in range(min(k, len(items))):
        values = mmr_lambda * normalized - (1 - mmr_lambda) * np.isin(kinds, list(seen))
        values[~remaining] = -np.inf
        maximum = values.max()
        tied = np.flatnonzero(values == maximum)
        at = int(tied[np.argmin(items[tied])])
        selected.append(at)
        remaining[at] = False
        seen.add(kinds[at])
    if sasrec_weight == 0:
        return items[selected].astype(int).tolist()
    groups, affinity = {}, np.zeros(len(items), dtype=float)
    for kind in np.unique(kinds):
        group = pool[kinds[pool] == kind]
        groups[kind] = group
        span = float(np.ptp(affinities[group]))
        if span > 1e-12:
            affinity[group] = (affinities[group] - affinities[group].min()) / span
    exposure = 1 / (1 + np.asarray(resource_days, dtype=float))
    used, output = np.zeros(len(items), dtype=bool), []
    for slot in selected:
        group = groups[kinds[slot]]
        choices = np.concatenate(([slot], group[group != slot]))
        choices = choices[~used[choices]]
        utility = np.minimum(scores[choices], scores[slot]) + sasrec_weight * (affinity[choices] + exposure[choices]) / 2
        at = int(choices[np.argmax(utility)])
        used[at] = True
        output.append(int(items[at]))
    return output


class CompactSnapshots:
    """Store numeric pairs, not millions of nested candidate dictionaries."""
    def __init__(self, agent, data, queries, events, day, risk, max_k=20, inference_batch_size=128):
        from .oulad_agent import FEATURE_NAMES
        self.data, self.queries, self.day = data, queries, day
        self.items = data.item.to_numpy()
        # Group equality is all MMR/SAS require; integer activity codes avoid
        # repeated object-string comparisons over millions of candidates.
        self.kinds = pd.Categorical(data.kind).codes
        self.templates, self.baseline = {}, {}
        self.x = np.empty((len(data), len(FEATURE_NAMES)), dtype=np.float64)
        self.fallback = np.empty(len(data), dtype=float)
        self.affinities = np.empty(len(data), dtype=float)
        self.history_resource_days = np.empty(len(data), dtype=np.int32)
        past = events.loc[events.date < day]
        groups = past.groupby(ENROLLMENT, sort=False).indices
        rows = list(queries.itertuples(index=False))
        started = time.perf_counter()
        prepared_affinities = {}
        print(f'  Personal signals: 0/{len(rows)} queries; SASRec device={getattr(agent.sequential, "device", "none")}, batch={inference_batch_size}', flush=True)
        for position, q in enumerate(rows):
            if position % inference_batch_size == 0 and agent.sequential is not None:
                requests = []
                batch_rows = rows[position:position + inference_batch_size]
                for b in batch_rows:
                    bkey = tuple(getattr(b, c) for c in ENROLLMENT)
                    bhistory = past.iloc[groups[bkey]] if bkey in groups else past.iloc[:0]
                    catalog = agent._available(bkey, day).set_index('id_site', drop=False)
                    catalog = catalog.loc[data.iloc[b.start:b.stop].item.to_numpy()].reset_index(drop=True)
                    requests.append((bkey, bhistory, catalog))
                values = agent.sequential.affinity_batch(requests, day, batch_size=inference_batch_size)
                prepared_affinities = {b.query: value for b, value in zip(batch_rows, values)}
            key = tuple(getattr(q, c) for c in ENROLLMENT)
            history = past.iloc[groups[key]] if key in groups else past.iloc[:0]
            part = data.iloc[q.start:q.stop]
            p = float(risk[q.role][q.forecast_position])
            prepared = {'candidates': part, 'query': q._asdict()}
            if q.query in prepared_affinities:
                prepared['sasrec_affinity'] = prepared_affinities[q.query]
            snapshot = agent.build_snapshot(key, day, p, prepared=prepared, history=history)
            if [c['item_id'] for c in snapshot['candidates']] != part.item.astype(int).tolist():
                raise ValueError('Agent changed the shared candidate order/universe')
            self.x[q.start:q.stop] = snapshot['features']
            self.fallback[q.start:q.stop] = [c['score'] for c in snapshot['candidates']]
            self.affinities[q.start:q.stop] = [c.get('sasrec_affinity', 0.) for c in snapshot['candidates']]
            self.history_resource_days[q.start:q.stop] = [c.get('history_resource_days', 0) for c in snapshot['candidates']]
            self.templates[q.query] = {key: value for key, value in snapshot.items()
                                       if key not in ('features', 'candidates')}
            self.baseline[q.query] = replay_baseline(part, q, history, day, p, k=max(10, max_k))
            if (position + 1) % inference_batch_size == 0 or position + 1 == len(rows):
                elapsed = time.perf_counter() - started
                rate = (position + 1) / max(elapsed, 1e-9)
                print(f'  Personal signals: {position + 1}/{len(rows)} queries; {rate:.2f} queries/s; elapsed={elapsed:.0f}s; ETA={(len(rows) - position - 1) / rate:.0f}s', flush=True)
        if not np.isfinite(self.x).all():
            raise ValueError('Nonfinite candidate feature')

    def snapshot(self, q):
        part = self.data.iloc[q.start:q.stop]
        candidates = []
        for i, row in enumerate(part.itertuples(index=False), q.start):
            candidates.append({'item_id': int(row.item), 'activity_type': str(row.kind),
                'item_type': 'vle_resource', 'score': float(self.fallback[i]),
                'sasrec_affinity': float(self.affinities[i]),
                'history_resource_days': int(self.history_resource_days[i]),
                'strategy': self.templates[q.query]['strategy_selected'],
                'signals': {name: float(getattr(row, name)) for name in
                            ('preference', 'calendar', 'engagement', 'popularity', 'confidence')}})
        return {**self.templates[q.query], 'candidates': candidates,
                'features': self.x[q.start:q.stop]}

    def selection(self, agent, scores=None):
        values = []
        for q in self.queries.itertuples(index=False):
            if q.role == 'selection':
                part = self.data.iloc[q.start:q.stop]
                truth = set(part.loc[part.label == 1, 'item'])
                values.append(ranking_metrics(self.rank(q, agent, 10, scores), truth, 10))
        if not values:
            raise ValueError('Selection queries are required')
        return {name: float(np.mean([r[name] for r in values])) for name in ('ndcg', 'recall')}

    def rank(self, q, agent, k, scores=None):
        sl = slice(q.start, q.stop)
        if scores is None:
            score = (agent.candidate_model.predict_proba(self.x[sl])[:, 1]
                     if agent.candidate_model is not None else self.fallback[sl])
        else:
            score = scores[sl]
        return compact_rank(self.items[sl], self.kinds[sl], score, self.affinities[sl],
                            self.history_resource_days[sl], k, agent.mmr_lambda, agent.sasrec_weight)


def fit_fusion(agent, store, roles, labels, seed, directory):
    """Choose only on selection NDCG, with recall at least replayed baseline."""
    fit = roles == 'fit'
    if not fit.any() or len(np.unique(labels[fit])) != 2:
        raise ValueError('Fit candidates need both proxy classes; enlarge pilot')
    baseline = []
    for q in store.queries.itertuples(index=False):
        if q.role == 'selection':
            part = store.data.iloc[q.start:q.stop]
            baseline.append(ranking_metrics(store.baseline[q.query], set(part.loc[part.label == 1, 'item']), 10))
    if not baseline:
        raise ValueError('Selection queries are required')
    reference = {name: float(np.mean([v[name] for v in baseline])) for name in ('ndcg', 'recall')}
    trials, best = [], None
    selection = roles == 'selection'
    for c in (.1, 1., 10.):
        print(f'  Fusion C={c}: fitting on fit candidates', flush=True)
        model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=500, random_state=seed))
        model.fit(store.x[fit], labels[fit])
        # Batched prediction avoids a scaler/classifier invocation per query.
        selection_scores = np.empty(len(labels), dtype=float)
        selection_scores[selection] = model.predict_proba(store.x[selection])[:, 1]
        agent.candidate_model = None
        result = store.selection(agent, scores=selection_scores)
        print(f'  Fusion C={c}: selection NDCG@10={result["ndcg"]:.6f}, Recall@10={result["recall"]:.6f}', flush=True)
        accepted = result['recall'] + 1e-12 >= reference['recall'] and result['ndcg'] + 1e-12 >= reference['ndcg']
        trials.append({'C': c, **result, 'accepted': accepted,
                       'iterations': model[-1].n_iter_.tolist(),
                       'iteration_limit_reached': bool(np.any(model[-1].n_iter_ >= 500))})
        if accepted and (best is None or result['ndcg'] > best[0]):
            best = (result['ndcg'], model, c)
    agent.candidate_model = None if best is None else best[1]
    report = {'feature_names': list(store.templates[next(iter(store.templates))]['feature_names']),
              'selection_reference': reference, 'trials': trials, 'selected_C': None if best is None else best[2],
              'fallback': best is None, 'mmr_lambda': .8, 'sasrec_weight': .05,
              'nondegradation_guaranteed': False,
              'fallback_note': 'New full-agent fallback differs from legacy replay; it may regress. Consult measured selection/held metrics.',
              'policy': 'preregistered MMR and bounded within-activity SASRec; selection labels only'}
    if best is not None:
        report.update(standardized_coefficients=best[1][-1].coef_[0].tolist(),
                      intercept=float(best[1][-1].intercept_[0]),
                      scaler_mean=best[1][0].mean_.tolist(), scaler_scale=best[1][0].scale_.tolist())
    write_json(directory / 'fusion_search_and_weights.json', report)
    score = store.fallback if best is None else best[1].predict_proba(store.x)[:, 1]
    if best is None:
        report['fallback_selection'] = store.selection(agent)
        report['fallback_selection_minus_reference'] = {
            name: report['fallback_selection'][name] - reference[name] for name in ('ndcg', 'recall')}
        write_json(directory / 'fusion_search_and_weights.json', report)
    return score


def fit_probability_head(x, score, labels, roles, seed):
    mask = roles == 'calibration'
    if not mask.any() or len(np.unique(labels[mask])) != 2:
        raise ValueError('Calibration candidates need both proxy classes; enlarge pilot')
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1., max_iter=500, random_state=seed))
    features = np.column_stack([x, score])
    model.fit(features[mask], labels[mask])
    return model, model.predict_proba(features)[:, 1]


def evaluate(agent, store, labels, probability, ks, seed, bootstrap, directory, scores=None):
    rows, exposure, universe, times = [], set(), set(), []
    for q in store.queries.itertuples(index=False):
        part = store.data.iloc[q.start:q.stop]
        truth = set(part.loc[part.label == 1, 'item'])
        started = time.perf_counter()
        ranked = store.rank(q, agent, max(ks), scores)
        times.append(time.perf_counter() - started)
        if q.role == 'held':
            # Full course/presentation item keys avoid cross-catalog ID collisions.
            exposure.update((q.code_module, q.code_presentation, i) for i in ranked[:10])
            universe.update((q.code_module, q.code_presentation, int(i)) for i in part.item)
        for method, ranking in [('full_agent', ranked), ('replayed_baseline', store.baseline[q.query])]:
            for k in ks:
                rows.append({'query': q.query, 'id_student': q.id_student, 'role': q.role,
                             'method': method, 'k': k, **ranking_metrics(ranking, truth, k)})
    rows = pd.DataFrame(rows)
    rows.to_csv(directory / 'query_metrics.csv', index=False)
    metric_names = ['precision', 'recall', 'f1', 'ndcg', 'map', 'mrr', 'hit_rate', 'fill_rate']
    summary = rows.groupby(['role', 'method', 'k'], as_index=False)[metric_names].mean()
    summary.to_csv(directory / 'ranking_summary.csv', index=False)
    held = rows.loc[(rows.role == 'held') & (rows.k == 10)]
    a, b = [held.loc[held.method == method].set_index('query') for method in ('full_agent', 'replayed_baseline')]
    contrast = {name: cluster_interval((a[name] - b[name]).to_numpy(), a.id_student.to_numpy(), bootstrap, seed)
                for name in ('precision', 'recall', 'ndcg', 'map', 'mrr', 'hit_rate')}
    write_json(directory / 'paired_held_bootstrap.json', {'contrast': 'full_agent minus replayed_baseline',
        'exploratory_validation_already_inspected': True, 'metrics_at_10': contrast})
    write_json(directory / 'coverage_latency.json', {'held_catalog_coverage_at_10': len(exposure) / len(universe) if universe else None,
        'ranking_latency_p50_seconds': float(np.quantile(times, .5)),
        'ranking_latency_p95_seconds': float(np.quantile(times, .95)),
        'latency_excludes_shared_features': True})
    roles = store.data['query'].map(store.queries.set_index('query').role).to_numpy()
    probabilities, bins = [], []
    for role in ROLES:
        mask = roles == role
        result, reliability = probability_metrics(labels[mask], probability[mask])
        probabilities.append({'role': role, **result, 'probability_std': float(np.std(probability[mask])),
                              'degenerate_probability': bool(np.ptp(probability[mask]) < 1e-8)})
        bins.extend({'role': role, **row} for row in reliability)
    pd.DataFrame(probabilities).to_csv(directory / 'contextual_probability_metrics.csv', index=False)
    pd.DataFrame(bins).to_csv(directory / 'reliability_bins.csv', index=False)
    gaps = []
    for (method, k), part in summary.groupby(['method', 'k']):
        part = part.set_index('role')
        for metric in metric_names:
            gaps.append({'method': method, 'k': int(k), 'metric': metric,
                         'fit_minus_selection': float(part.loc['fit', metric] - part.loc['selection', metric]),
                         'fit_minus_held': float(part.loc['fit', metric] - part.loc['held', metric])})
    pd.DataFrame(gaps).to_csv(directory / 'generalization_gaps.csv', index=False)
    return summary


def run_full_agent(risk_run, output, windows=tuple(range(7, 169, 7)), seeds=(42,), *,
                   horizon=7, ks=(5, 10, 20), bootstrap=1000, threads=6, max_learners=None,
                   raw=None, epochs=20, patience=4, device='auto', sasrec_dimension=64, inference_batch_size=128, resume=False):
    previous_workspace = os.environ.get('CUBLAS_WORKSPACE_CONFIG')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    from .oulad_agent import OULADRecommendationAgent, FEATURE_NAMES
    from .retrieval import OULADRetrieval
    from .sequential import DailySASRecService
    import torch
    if device != 'cpu' and torch.cuda.is_available():
        if torch.cuda.is_initialized() and previous_workspace not in (':4096:8', ':16:8'):
            raise RuntimeError('CUDA was initialized without deterministic CUBLAS workspace settings. '
                               'Restart Python with CUBLAS_WORKSPACE_CONFIG=:4096:8 before importing Torch.')
        if os.environ['CUBLAS_WORKSPACE_CONFIG'] not in (':4096:8', ':16:8'):
            raise ValueError('Deterministic CUDA requires CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8')
    root = Path(__file__).resolve().parents[1]
    risk_run, output = Path(risk_run).resolve(), Path(output).resolve()
    raw = Path(raw).resolve() if raw is not None else root / 'data/oulad/raw'
    windows, seeds, ks = list(windows), list(seeds), list(ks)
    if not windows or len(set(windows)) != len(windows) or min(windows) < 1 or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Positive unique windows and unique nonempty seeds required')
    if min(horizon, bootstrap, threads, epochs, patience, inference_batch_size) < 1 or not ks or min(ks) < 1 or 10 not in ks:
        raise ValueError('Positive limits and K=10 required')
    if max_learners is not None and max_learners < 1:
        raise ValueError('Positive pilot limit required')
    hashes = validate_source(risk_run, output, windows, seeds, raw)
    if resume:
        import psutil
        for process in psutil.process_iter(['pid', 'cmdline']):
            if process.info['pid'] != os.getpid() and any(Path(arg).name == 'run_oulad_agent_experiment.py' for arg in (process.info['cmdline'] or [])):
                raise ValueError('Another full-agent experiment is running; stop it before resuming')
    previous = json.loads((output / 'manifest.json').read_text(encoding='utf-8')) if resume else None
    output.mkdir(parents=True, exist_ok=resume)
    code = output / 'source_snapshot'
    code.mkdir(exist_ok=resume)
    sources = [*Path(__file__).parent.glob('*.py'), root / 'run_oulad_agent_experiment.py']
    source_hashes = {}
    for path in sources:
        target = code / path.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not resume:
            shutil.copy2(path, target)
        source_hashes[str(path.relative_to(root))] = sha256(path)
    config = {'risk_source': str(risk_run), 'raw_source': str(raw), 'windows': windows, 'seeds': seeds,
              'horizon': horizon, 'ks': ks, 'bootstrap': bootstrap, 'threads': threads, 'epochs': epochs,
              'patience': patience, 'device': device, 'sasrec_dimension': sasrec_dimension,
              'inference_batch_size': inference_batch_size, 'inference_execution': 'batched SASRec FP32; CPU table processing',
              'max_learners_per_split': max_learners, 'pilot_not_for_publication': max_learners is not None,
              'feature_names': list(FEATURE_NAMES), 'raw_sha256': hashes, 'source_code_sha256': source_hashes,
              'source_manifest_sha256': sha256(risk_run / 'run_manifest.json'),
              'risk_retrained': False, 'external_test_used': False, 'allow_review': True,
              'exploratory_validation_already_inspected': True, 'primary_day': 56,
              'probability_head': 'independent contextual LogisticRegression; not monotone Platt',
              'ranking_engine': 'exact compact-array MMR/activity-type bounded SAS replay; facade parity tested',
              'group_semantics': 'activity_type; explicit topic tags absent from OULAD candidate schema',
              'target': 'positive future click on full available catalog; unclicked is proxy negative'}
    if resume:
        if previous.get('resume_source_code_sha256', source_hashes) != source_hashes:
            raise ValueError('Code changed since the last resume')
        for key, value in config.items():
            if key != 'source_code_sha256' and previous.get(key) != value:
                raise ValueError(f'Resume configuration mismatch: {key}')
        allowed = {'recommendations/full_agent_experiment.py': '345a87c3e63b6675eda6533314a70df0728ffb07603d2de726b640dd3aff9391',
                   'run_oulad_agent_experiment.py': '2efe1c2cc4dd2621d6c9bedb4e298e770704c22f58b5e0476989d4aa6482bffa'}
        for name, old_hash in previous['source_code_sha256'].items():
            normalized = name.replace('\\', '/')
            if sha256(code / name) != old_hash:
                raise ValueError(f'Corrupt source snapshot: {name}')
            if source_hashes.get(name) != old_hash and allowed.get(normalized) != old_hash:
                import ast
                if normalized != 'recommendations/sequential.py' or ast.dump(ast.parse((code / name).read_text(encoding='utf-8')), include_attributes=False) != ast.dump(ast.parse((root / name).read_text(encoding='utf-8')), include_attributes=False):
                    raise ValueError(f'Resume source mismatch: {name}')
        # Preserve the original source identity for repeated resumes.
        config['source_code_sha256'] = previous['source_code_sha256']
    config['resume_source_code_sha256'] = source_hashes
    write_json(output / 'manifest.json', {**config, 'status': 'running'})
    try:
        with ResourceMonitor() as total, threadpool_limits(limits=threads):
            torch.set_num_threads(threads)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
            torch.use_deterministic_algorithms(True)
            cohorts = {day: source_splits(risk_run / f'day_{day}') for day in windows}
            keys = pd.concat([frame[ENROLLMENT] for frame, _ in cohorts.values()]).drop_duplicates()
            events = load_events(raw, keys, max(windows) + horizon)
            resources, courses = pd.read_csv(raw / 'vle.csv'), pd.read_csv(raw / 'courses.csv')
            summaries = []
            for day in windows:
                frame, indices = cohorts[day]
                directory = output / f'day_{day}'
                directory.mkdir(exist_ok=resume)
                print(f'Full agent day {day}: shared full candidates', flush=True)
                data, queries, splits, _ = make_candidates(resources, events, frame, indices, day, horizon,
                    courses, max_learners, include_no_decay=False, allow_review=True,
                    feature_engine='batched', progress=True)
                if data.empty or any(not (queries.role == role).any() for role in ROLES):
                    raise ValueError('Every role needs eligible candidates/queries')
                data.to_parquet(directory / 'candidate_features_and_labels.parquet', index=False)
                queries.to_csv(directory / 'queries.csv', index=False)
                splits.to_csv(directory / 'split_membership.csv', index=False)
                roles = data['query'].map(queries.set_index('query').role).to_numpy()
                labels = data.label.to_numpy(dtype=int)
                for seed in seeds:
                    destination = directory / f'seed_{seed}'
                    if resume and complete_job(destination, day, seed):
                        if not (destination / 'checkpoint.json').exists():
                            save_job_checkpoint(destination, day, seed)
                        summaries.append(pd.read_csv(destination / 'ranking_summary.csv').assign(day=day, seed=seed))
                        print(f'Full agent day {day}, seed {seed}: reusing completed job', flush=True)
                        continue
                    if destination.exists():
                        archive = destination.with_name(destination.name + '_interrupted_' + str(time.time_ns()))
                        destination.rename(archive)
                        print(f'Archived incomplete job: {archive}; restarting this day/seed', flush=True)
                    destination.mkdir()
                    risk, _, ensemble = reconstructed_risk(risk_run / f'day_{day}', seed, indices)
                    write_json(destination / 'frozen_ensemble_config.json', ensemble)
                    query_risk = queries[['query', *ENROLLMENT, 'role', 'forecast_position']].copy()
                    query_risk['forecast_probability'] = [float(risk[q.role][q.forecast_position]) for q in queries.itertuples()]
                    query_risk.to_csv(destination / 'frozen_forecasts.csv', index=False)
                    # Restrict all service fitting to exact fit enrollments, with past selection only for early stop.
                    fit_keys = frame.iloc[indices['fit']][ENROLLMENT]
                    selection_keys = frame.iloc[indices['selection']][ENROLLMENT]
                    if max_learners is not None:
                        fit_keys = fit_keys.loc[fit_keys.id_student.isin(sorted(fit_keys.id_student.unique())[:max_learners])]
                        selection_keys = selection_keys.loc[selection_keys.id_student.isin(sorted(selection_keys.id_student.unique())[:max_learners])]
                    with ResourceMonitor() as monitor:
                        if torch.cuda.is_available() and device != 'cpu':
                            torch.cuda.reset_peak_memory_stats()
                        print(f'Full agent day {day}, seed {seed}: fitting SVD retrieval', flush=True)
                        retrieval = OULADRetrieval(seed=seed).fit(resources, events, fit_keys, day)
                        print(f'  SVD complete: {retrieval.fit_metadata_["elapsed_seconds"]:.1f}s; preparing SASRec on {device}', flush=True)
                        sequential = DailySASRecService(seed=seed, dimension=sasrec_dimension)
                        sequential.fit(resources, events, fit_keys, day, selection_keys=selection_keys,
                                       epochs=epochs, patience=patience, device=device)
                        write_json(destination / 'sasrec_training.json', sequential.metadata)
                        pd.DataFrame(sequential.metadata.get('loss_curve', [])).to_csv(destination / 'sasrec_learning_curve.csv', index=False)
                        agent = OULADRecommendationAgent(resources, events, retrieval=retrieval,
                            sequential=sequential, random_seed=seed, allow_review=True, mmr_lambda=.8, sasrec_weight=.05)
                        print(f'Full agent day {day}: computing personal signals', flush=True)
                        store = CompactSnapshots(agent, data, queries, events, day, risk, max_k=max(ks), inference_batch_size=inference_batch_size)
                        query_diagnostics = []
                        for q in queries.itertuples(index=False):
                            template = store.templates[q.query]
                            profile = template.get('profile', {})
                            query_diagnostics.append({'query': q.query, 'role': q.role,
                                'strategy_selected': template.get('strategy_selected'),
                                'history_resource_days': q.history_resource_days,
                                'cold_start': bool(q.history_resource_days == 0),
                                'profile_confidence': profile.get('confidence', q.preference_confidence)})
                        pd.DataFrame(query_diagnostics).to_csv(destination / 'query_strategy_diagnostics.csv', index=False)
                        write_json(destination / 'query_diagnostic_summary.json', {
                            'cold_start_fraction': float(np.mean([r['cold_start'] for r in query_diagnostics])),
                            'n_queries': len(query_diagnostics), 'n_candidate_pairs': len(data)})
                        np.save(destination / 'full_candidate_features.npy', store.x, allow_pickle=False)
                        np.savez_compressed(destination / 'candidate_utilities.npz', fallback=store.fallback,
                                            sasrec_affinity=store.affinities,
                                            history_resource_days=store.history_resource_days, labels=labels)
                        print(f'Full agent day {day}: fitting fusion and checking selection', flush=True)
                        score = fit_fusion(agent, store, roles, labels, seed, destination)
                        print(f'Full agent day {day}: fitting contextual probability head', flush=True)
                        agent.probability_model, probability = fit_probability_head(store.x, score, labels, roles, seed)
                        write_json(destination / 'probability_head.json', {'type': 'contextual LogisticRegression',
                            'fit_role': 'calibration', 'features': list(FEATURE_NAMES) + ['fusion_probability_or_fallback_utility'],
                            'standardized_coefficients': agent.probability_model[-1].coef_[0].tolist(),
                            'intercept': float(agent.probability_model[-1].intercept_[0]),
                            'zero_clamped': False, 'changes_ranking': False,
                            'degenerate_probability': bool(np.ptp(probability) < 1e-8)})
                        agent.save(destination / 'agent_bundle')
                        print(f'Full agent day {day}: evaluation and {bootstrap} bootstrap replicates', flush=True)
                        summary = evaluate(agent, store, labels, probability, ks, seed, bootstrap, destination, scores=score)
                        summaries.append(summary.assign(day=day, seed=seed))
                        np.savez_compressed(destination / 'held_candidate_predictions.npz',
                            labels=labels[roles == 'held'], score=score[roles == 'held'], probability=probability[roles == 'held'])
                    write_json(destination / 'runtime_memory.json', {**monitor.result,
                        'gpu_used': str(sequential.device).startswith('cuda'),
                        'gpu_peak_allocated_bytes': int(torch.cuda.max_memory_allocated()) if str(sequential.device).startswith('cuda') else 0,
                        'gpu_peak_reserved_bytes': int(torch.cuda.max_memory_reserved()) if str(sequential.device).startswith('cuda') else 0})
                    save_job_checkpoint(destination, day, seed)
                    print(f'Checkpoint saved: {destination / "checkpoint.json"}', flush=True)
                    del store, agent, sequential, retrieval
                    print(f'Full agent day {day}, seed {seed}: complete', flush=True)
                pd.concat(summaries, ignore_index=True).to_csv(output / 'ranking_summary.csv', index=False)
        write_json(output / 'manifest.json', {**config, 'status': 'completed', **total.result})
        (output / 'REPORT_RU.md').write_text(
            '# Полный рекомендательный агент OULAD\n\n'
            'Обучены SVD retrieval, daily-basket SASRec и согласование кандидатов; исходный ансамбль риска только прочитан.\n\n'
            'Использованы learner-disjoint fit/selection/calibration/held 2013; 2014 не используется. '
            'Полная доступная вселенная включает полезные повторы. Эталон исключает недавно просмотренное, '
            'но оценивается на той же полной целевой релевантности.\n\n'
            'C выбирается только по selection NDCG@10 при Recall@10 не ниже эталона; при отсутствии '
            'допустимого варианта сохраняется fallback. MMR=.8 и SASRec=.05 зафиксированы заранее. '
            'Отдельная contextual probability head обучена только на calibration и не меняет ранжирование.\n\n'
            'ranking_summary.csv содержит P/R/F1/NDCG/MAP/MRR/HitRate @5/10/20. В day_*/seed_* '
            'сохранены learner-cluster bootstrap, охват, задержки, память, калибровка, разрывы fit/held, '
            'полные признаки, веса и загружаемый agent_bundle. Проверяйте fallback и degenerate_probability.\n\n'
            'Будущий клик — proxy relevance; отсутствие клика не доказывает нерелевантность. '
            'Held уже просматривалась: сравнение исследовательское, улучшение обучения не доказано. '
            'Пилот с max_learners не предназначен для публикации.\n', encoding='utf-8')
    except BaseException as exc:
        write_json(output / 'manifest.json', {**config, 'status': 'failed', 'error': repr(exc)})
        raise
    return output
