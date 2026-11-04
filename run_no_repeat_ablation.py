"""Frozen-model ablation: exclude all previously accessed materials, no training."""
import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from recommendations.full_agent_experiment import compact_rank
from recommendations.experiment_metrics import ranking_metrics, cluster_interval
from run_oulad_agent_experiment import _Tee
from contextlib import redirect_stdout, redirect_stderr
import sys


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--agent-run', type=Path, required=True)
    parser.add_argument('--windows', nargs='+', type=int, default=[7, 14, 21, 28])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ks', nargs='+', type=int, default=[5, 10, 20])
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    source = args.agent_run.resolve()
    output = (args.resume or args.output or source.parent / ('no_repeat_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))).resolve()
    if output == source or output.is_relative_to(source):
        parser.error('Output must be separate from source')
    if min(args.ks + [args.bootstrap, args.threads]) < 1:
        parser.error('Positive K, threads and bootstrap required')
    config = {'source': str(source), 'windows': args.windows, 'seed': args.seed, 'ks': args.ks,
              'bootstrap': args.bootstrap, 'script_sha256': sha(Path(__file__)),
              'ranking_sha256': sha(Path(__file__).parent / 'recommendations/full_agent_experiment.py'),
              'policy': 'exclude every item with positive historical resource_days before cutoff',
              'primary_truth': 'same full available-catalog future relevance for both policies',
              'training': False, 'test_used': False}
    if args.resume:
        if json.loads((output / 'config.json').read_text(encoding='utf-8')) != config:
            raise ValueError('Resume configuration/code mismatch')
    output.mkdir(parents=True, exist_ok=bool(args.resume))
    (output / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    log = output / ('log_' + str(time.time_ns()) + '.txt')
    with log.open('x', encoding='utf-8') as logfile, redirect_stdout(_Tee(sys.stdout, logfile)), redirect_stderr(_Tee(sys.stderr, logfile)), threadpool_limits(limits=args.threads):
        summaries, differences = [], []
        for day in args.windows:
            base = source / f'day_{day}'
            job = base / f'seed_{args.seed}'
            required = [base / 'candidate_features_and_labels.parquet', base / 'queries.csv',
                        job / 'candidate_utilities.npz', job / 'full_candidate_features.npy',
                        job / 'agent_bundle/ranking.joblib', job / 'agent_bundle/bundle.json']
            if not (job / 'runtime_memory.json').is_file():
                raise ValueError(f'Source job not complete: {job}')
            print(f'Day {day}: verifying frozen inputs', flush=True)
            hashes = {str(p.relative_to(source)): sha(p) for p in required}
            checkpoint = output / f'day_{day}.json'
            table = output / f'day_{day}_queries.csv'
            if args.resume and checkpoint.exists():
                saved = json.loads(checkpoint.read_text())
                if saved['inputs'] != hashes or saved['output_sha256'] != sha(table):
                    raise ValueError('Checkpoint checksum mismatch')
                rows = pd.read_csv(table)
                print(f'Day {day}: reusing checkpoint', flush=True)
            else:
                data = pd.read_parquet(required[0])
                queries = pd.read_csv(required[1]); queries = queries.loc[queries.role == 'held']
                with np.load(required[2], allow_pickle=False) as utilities:
                    history = utilities['history_resource_days'].copy()
                    affinity = utilities['sasrec_affinity'].copy()
                    fallback = utilities['fallback'].copy()
                bundle = joblib.load(required[4])
                settings = json.loads(required[5].read_text())
                if sha(required[4]) != settings['artifacts_sha256']['ranking.joblib']:
                    raise ValueError('Frozen ranking bundle checksum mismatch')
                x = np.load(required[3], mmap_mode='r', allow_pickle=False)
                if list(bundle['feature_names']) != settings['feature_names']:
                    raise ValueError('Feature schema mismatch')
                items, kinds = data.item.to_numpy(), pd.Categorical(data.kind).codes
                labels = data.label.to_numpy()
                records = []; started = time.perf_counter()
                for index, q in enumerate(queries.itertuples(index=False), 1):
                    sl = slice(q.start, q.stop)
                    score = bundle['candidate_model'].predict_proba(x[sl])[:, 1] if bundle['candidate_model'] is not None else fallback[sl]
                    truth = set(items[sl][labels[sl] == 1])
                    for policy, mask in [('repeats_allowed', np.ones(q.stop-q.start, dtype=bool)), ('never_seen_only', history[sl] == 0)]:
                        ranked = compact_rank(items[sl][mask], kinds[sl][mask], score[mask], affinity[sl][mask], history[sl][mask], max(args.ks), settings['mmr_lambda'], settings['sasrec_weight'])
                        unseen_truth = truth & set(items[sl][history[sl] == 0])
                        for k in args.ks:
                            values = ranking_metrics(ranked, truth, k)
                            conditional = ranking_metrics(ranked, unseen_truth, k) if policy == 'never_seen_only' else {}
                            records.append({'day': day, 'query': q.query, 'id_student': q.id_student, 'policy': policy,
                                            'k': k, 'n_candidates': int(mask.sum()), 'n_relevant_all': len(truth),
                                            'n_relevant_unseen': len(unseen_truth), **values,
                                            **{'conditional_unseen_' + key: value for key, value in conditional.items()}})
                    if index % 128 == 0 or index == len(queries):
                        elapsed = time.perf_counter()-started
                        print(f'Day {day}: {index}/{len(queries)} queries; {index/max(elapsed,1e-9):.1f}/s; elapsed={elapsed:.0f}s; ETA={elapsed/index*(len(queries)-index):.0f}s', flush=True)
                rows = pd.DataFrame(records)
                temporary = table.with_suffix('.tmp'); rows.to_csv(temporary, index=False); temporary.replace(table)
                temporary = checkpoint.with_suffix('.tmp')
                temporary.write_text(json.dumps({'inputs': hashes, 'output_sha256': sha(table)}), encoding='utf-8'); temporary.replace(checkpoint)
            for (policy, k), part in rows.groupby(['policy', 'k']):
                values = {name: float(part[name].mean()) for name in ['precision', 'recall', 'ndcg', 'f1', 'map', 'mrr', 'hit_rate']}
                interval = cluster_interval(part.ndcg.to_numpy(), part.id_student.to_numpy(), repeats=args.bootstrap, seed=args.seed)
                summaries.append({'day': day, 'policy': policy, 'k': int(k), **values, 'ndcg_cluster_interval': json.dumps(interval)})
            for k in args.ks:
                allowed = rows.loc[(rows.k == k) & (rows.policy == 'repeats_allowed')].sort_values('query')
                excluded = rows.loc[(rows.k == k) & (rows.policy == 'never_seen_only')].sort_values('query')
                if not np.array_equal(allowed['query'], excluded['query']):
                    raise ValueError('Paired query alignment failed')
                for metric in ['precision', 'recall', 'ndcg']:
                    delta = excluded[metric].to_numpy() - allowed[metric].to_numpy()
                    interval = cluster_interval(delta, allowed.id_student.to_numpy(), repeats=args.bootstrap, seed=args.seed)
                    differences.append({'day': day, 'k': k, 'metric': metric, 'contrast': 'never_seen_only minus repeats_allowed', **interval})
            pd.DataFrame(summaries).to_csv(output / 'summary.csv', index=False)
            pd.DataFrame(differences).to_csv(output / 'paired_policy_differences.csv', index=False)
        print(f'Finished frozen no-repeat ablation: {output}', flush=True)


if __name__ == '__main__':
    main()
