"""Controlled fusion ablations and frozen-versus-updated preference profiles."""
import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from article_experiment_utils import session, sha, ready, checkpoint, atomic_csv, atomic_json
from recommendations.full_agent_experiment import compact_rank
from recommendations.experiment_metrics import ranking_metrics, cluster_interval, ResourceMonitor

VARIANTS = ['full_refit', 'without_explicit_preference', 'without_risk', 'without_cf',
            'without_content', 'without_sasrec', 'without_mmr', 'frozen_profile_day7']
KEY = ['id_student', 'code_module', 'code_presentation']


def get_input_paths(source, day, seed):
    base = source / f'day_{day}'; job = base / f'seed_{seed}'
    if not (job / 'runtime_memory.json').exists():
        raise ValueError(f'Source job incomplete: {job}')
    return [base / 'candidate_features_and_labels.parquet', base / 'queries.csv',
            job / 'full_candidate_features.npy', job / 'candidate_utilities.npz',
            job / 'query_strategy_diagnostics.csv', job / 'agent_bundle/bundle.json']


def make_variant(name, original, names, data, queries, diagnostics, frozen):
    x = np.array(original, copy=True)
    column = {n: i for i, n in enumerate(names)}
    strategy = data['query'].map(diagnostics.set_index('query').strategy_selected).to_numpy()
    if name == 'without_explicit_preference':
        x[:, [column['preference'], column['preference_confidence']]] = 0
    elif name == 'without_risk':
        x[:, column['risk_calendar']] = 0
    elif name in ('without_cf', 'without_content'):
        signal = 'collaborative' if name == 'without_cf' else 'content_based'
        x[:, column[signal]] = 0
        x[strategy == signal, column['selected_strategy_score']] = 0
    elif name == 'frozen_profile_day7':
        # Identity join, never target or future-interaction joins.
        membership = queries[['query', *KEY]]
        current = data[['query', 'kind']].merge(membership, on='query', how='left', validate='many_to_one', sort=False)
        mapped = current.merge(frozen, on=[*KEY, 'kind'], how='left', validate='many_to_one', sort=False)
        x[:, column['preference']] = mapped.frozen_preference.fillna(0).to_numpy()
        x[:, column['preference_confidence']] = mapped.frozen_confidence.fillna(0).to_numpy()
    return x


def rank_queries(data, queries, scores, affinity, days, variant, role, ks):
    items = data.item.to_numpy(); kinds = pd.Categorical(data.kind).codes; labels = data.label.to_numpy()
    records = []; subset = queries.loc[queries.role == role]; started = time.perf_counter()
    for index, q in enumerate(subset.itertuples(index=False), 1):
        sl = slice(q.start, q.stop)
        ranked = compact_rank(items[sl], kinds[sl], scores[sl], affinity[sl], days[sl], max(ks),
                              1. if variant == 'without_mmr' else .8,
                              0. if variant == 'without_sasrec' else .05)
        truth = set(items[sl][labels[sl] == 1])
        for k in ks:
            records.append({'query': q.query, 'id_student': q.id_student, 'k': k,
                            **ranking_metrics(ranked, truth, k)})
        if index % 512 == 0 or index == len(subset):
            elapsed = time.perf_counter() - started
            print(f'  {variant} {role}: {index}/{len(subset)}; {index/max(elapsed,1e-9):.1f} queries/s; ETA={elapsed/index*(len(subset)-index):.0f}s', flush=True)
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--agent-run', type=Path, required=True)
    parser.add_argument('--windows', type=int, nargs='+', default=[7, 14, 21, 28])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=VARIANTS)
    parser.add_argument('--ks', type=int, nargs='+', default=[5, 10, 20])
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--output', type=Path); parser.add_argument('--resume', type=Path)
    args = parser.parse_args(); source = args.agent_run.resolve()
    if 10 not in args.ks or min(args.ks + [args.bootstrap, args.threads]) < 1:
        parser.error('Positive limits and K=10 required')
    variants = list(dict.fromkeys(['full_refit', *args.variants]))
    input_paths = [p for d in sorted(set([7, *args.windows])) for p in get_input_paths(source, d, args.seed)]
    print('Verifying frozen input hashes (no training started yet)', flush=True)
    inputs = {str(p.relative_to(source)): sha(p) for p in input_paths}
    config = {'source': str(source), 'inputs': inputs, 'windows': args.windows, 'seed': args.seed,
              'variants': variants, 'ks': args.ks, 'bootstrap': args.bootstrap, 'test_used': False,
              'exploratory_validation': True, 'intervals': 'unadjusted learner-cluster bootstrap; multiple contrasts exploratory',
              'protocol': 'refit fusion on fit; select C by selection NDCG@10 then recall; evaluate held',
              'preference_comparison': 'profile frozen at day 7 versus current profile; all other histories stay current',
              'training': 'fusion only; frozen SASRec/SVD/risk reused',
              'cold_frozen_preferences': 'zero signal if student/type absent on day7',
              'candidate_policy': 'same full available catalog and repeat eligibility for every variant',
              'ranking_source_sha256': sha(Path(__file__).parent / 'recommendations/full_agent_experiment.py')}
    with session('recommendation_controls', config, __file__, args.output, args.resume) as (output, identity), threadpool_limits(limits=args.threads):
        old_data = pd.read_parquet(source / 'day_7/candidate_features_and_labels.parquet', columns=['query', 'kind', 'preference', 'confidence'])
        old_queries = pd.read_csv(source / 'day_7/queries.csv', usecols=['query', *KEY])
        frozen = old_data.drop_duplicates(['query', 'kind']).merge(old_queries, on='query', validate='many_to_one')
        frozen = frozen[[*KEY, 'kind', 'preference', 'confidence']].rename(columns={'preference': 'frozen_preference', 'confidence': 'frozen_confidence'})
        summaries, differences = [], []
        for day in args.windows:
            paths = get_input_paths(source, day, args.seed); job = paths[2].parent
            data = pd.read_parquet(paths[0]); queries = pd.read_csv(paths[1]); original = np.load(paths[2], mmap_mode='r')
            settings = json.loads(paths[5].read_text()); names = settings['feature_names']
            diagnostics = pd.read_csv(paths[4])
            with np.load(paths[3], allow_pickle=False) as saved:
                affinity = saved['sasrec_affinity'].copy(); history_days = saved['history_resource_days'].copy()
            roles = data['query'].map(queries.set_index('query').role).to_numpy(); labels = data.label.to_numpy()
            fit = roles == 'fit'
            score_cache = None
            for variant in variants:
                directory = output / f'day_{day}' / variant; directory.mkdir(parents=True, exist_ok=True)
                if ready(directory, identity):
                    print(f'Day {day}: {variant} checkpoint reused', flush=True)
                    results = pd.read_csv(directory / 'held_query_metrics.csv')
                else:
                    print(f'Day {day}: {variant} fitting controlled fusion', flush=True)
                    with ResourceMonitor() as monitor:
                        x = make_variant(variant, original, names, data, queries, diagnostics, frozen)
                        if variant == 'frozen_profile_day7' and day == 7 and not np.allclose(x, original, rtol=1e-12, atol=1e-12):
                            raise ValueError('Day-7 frozen profile must equal the day-7 updated profile')
                        cacheable = variant in ('full_refit', 'without_sasrec', 'without_mmr')
                        candidates = score_cache if cacheable and score_cache is not None else []
                        if not candidates:
                            for c in [.1, 1., 10.]:
                                print(f'  {variant}: C={c}, fit pairs={int(fit.sum())}', flush=True)
                                model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=500, random_state=args.seed))
                                model.fit(x[fit], labels[fit])
                                scores = np.empty(len(data))
                                for start in range(0, len(data), 65536):
                                    scores[start:start+65536] = model.predict_proba(x[start:start+65536])[:, 1]
                                candidates.append((c, model, scores))
                            if cacheable:
                                score_cache = candidates
                        trials = []; best = None
                        for c, model, scores in candidates:
                            selection = rank_queries(data, queries, scores, affinity, history_days, variant, 'selection', [10])
                            quality = (float(selection.ndcg.mean()), float(selection.recall.mean()))
                            trials.append({'C': c, 'selection_ndcg10': quality[0], 'selection_recall10': quality[1],
                                           'iteration_limit_reached': bool(np.any(model[-1].n_iter_ >= 500))})
                            if best is None or quality > best[0]:
                                best = (quality, c, model, scores)
                        results = rank_queries(data, queries, best[3], affinity, history_days, variant, 'held', args.ks)
                        joblib.dump(best[2], directory / 'fusion.joblib')
                        atomic_json(directory / 'selection.json', {'trials': trials, 'selected_C': best[1], 'feature_names': names,
                                     'standardized_coefficients': best[2][-1].coef_[0].tolist()})
                        atomic_csv(directory / 'held_query_metrics.csv', results)
                        del x
                    atomic_json(directory / 'runtime_memory.json', monitor.result)
                    checkpoint(directory, identity)
                for k, part in results.groupby('k'):
                    interval = cluster_interval(part.ndcg.to_numpy(), part.id_student.to_numpy(), args.bootstrap, args.seed)
                    summaries.append({'day': day, 'variant': variant, 'k': int(k),
                                      **{n: float(part[n].mean()) for n in ['precision', 'recall', 'f1', 'ndcg', 'map', 'mrr', 'hit_rate']},
                                      'ndcg_ci_low': interval['low'], 'ndcg_ci_high': interval['high']})
                if variant == 'full_refit':
                    reference = results
                else:
                    paired = results.merge(reference, on=['query', 'id_student', 'k'], suffixes=('_variant', '_full'), validate='one_to_one')
                    for k, part in paired.groupby('k'):
                        for metric in ['ndcg', 'recall', 'precision']:
                            interval = cluster_interval((part[metric+'_variant'] - part[metric+'_full']).to_numpy(), part.id_student.to_numpy(), args.bootstrap, args.seed)
                            differences.append({'day': day, 'variant_minus_full': variant, 'k': int(k), 'metric': metric, **interval})
                atomic_csv(output / 'summary.csv', pd.DataFrame(summaries))
                atomic_csv(output / 'paired_differences.csv', pd.DataFrame(differences))
            del data, original, affinity, history_days, score_cache


if __name__ == '__main__':
    main()
