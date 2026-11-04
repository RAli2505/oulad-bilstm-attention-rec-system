"""Capacity-constrained risk ranking and subgroup reports, without training."""
import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

from article_experiment_utils import session, sha, ready, checkpoint, atomic_csv
from recommendations.experiment import source_splits, reconstructed_risk
from recommendations.experiment_metrics import cluster_interval

MODEL = 'ensemble_lstm_attention_lgbm_weighted'


def scores(y, p, selected):
    y = np.asarray(y, dtype=int); selected = np.asarray(selected, dtype=bool)
    tp = int(y[selected].sum()); fp = int(selected.sum()) - tp; fn = int(y.sum()) - tp
    precision = tp / (tp + fp) if tp + fp else 0.
    recall = tp / (tp + fn) if tp + fn else 0.
    return {'n': len(y), 'positive': int(y.sum()), 'selected': int(selected.sum()),
            'fraction_selected': float(selected.mean()), 'precision': precision, 'recall': recall,
            'f1': 2 * precision * recall / (precision + recall) if precision + recall else 0.,
            'roc_auc': float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
            'pr_auc': float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else None,
            'brier': float(np.mean((np.asarray(p) - y) ** 2))}


def top_fraction(p, fraction):
    count = max(1, int(np.ceil(len(p) * fraction)))
    mask = np.zeros(len(p), dtype=bool)
    # Fixed saved enrollment row order is the prespecified tie-break.
    mask[np.argsort(-np.asarray(p), kind='stable')[:count]] = True
    return mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--risk-run', type=Path, required=True)
    parser.add_argument('--windows', type=int, nargs='+', default=[7, 14, 21, 28, 56, 84, 112])
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 13, 7, 23, 101])
    parser.add_argument('--capacities', type=float, nargs='+', default=[.1, .2, .3])
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--output', type=Path); parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    if any(c <= 0 or c >= 1 for c in args.capacities) or min(args.bootstrap, args.threads) < 1:
        parser.error('Capacities in (0,1) and positive limits required')
    source = args.risk_run.resolve(); manifest = json.loads((source / 'run_manifest.json').read_text())
    if manifest['status'] != 'completed' or manifest['config']['evaluate_test']:
        raise ValueError('Completed 2013-only source required')
    if not set(args.windows) <= set(manifest['config']['windows']) or not set(args.seeds) <= set(manifest['config']['seeds']):
        raise ValueError('Windows/seeds missing from source')
    raw = Path(__file__).parent / 'data/oulad/raw/studentInfo.csv'
    if sha(raw) != manifest['data_sha256']['studentInfo.csv']:
        raise ValueError('Student metadata changed')
    config = {'source': str(source), 'manifest_sha256': sha(source / 'run_manifest.json'),
              'windows': args.windows, 'seeds': args.seeds, 'capacities': args.capacities,
              'bootstrap': args.bootstrap, 'student_metadata_sha256': sha(raw),
              'training': False, 'test_used': False, 'exploratory_validation': True,
              'capacity_unit': 'student-course enrollment; students taking two courses may have two records',
              'threshold_selection': 'selection only; held is diagnostic, never used to choose a threshold'}
    with session('risk_practical', config, __file__, args.output, args.resume) as (output, identity), threadpool_limits(limits=args.threads):
        metadata = pd.read_csv(raw)
        metadata = metadata[['id_student', 'code_module', 'code_presentation', 'gender', 'age_band', 'disability', 'imd_band', 'highest_education']]
        budgets, subgroups = [], []
        for day in args.windows:
            directory = output / f'day_{day}'; directory.mkdir(exist_ok=True)
            if ready(directory, identity):
                budgets.append(pd.read_csv(directory / 'capacity.csv')); subgroups.append(pd.read_csv(directory / 'subgroups.csv')); continue
            print(f'Day {day}: reconstructing fixed per-seed forecasts and selection-only thresholds', flush=True)
            frame, indices = source_splits(source / f'day_{day}')
            # Reconstruction checks saved held forecasts against frozen calibrators.
            all_risk = [reconstructed_risk(source / f'day_{day}', seed, indices)[0] for seed in args.seeds]
            selection_p = np.mean([v['selection'] for v in all_risk], axis=0)
            held_p = np.mean([v['held'] for v in all_risk], axis=0)
            y = frame.iloc[indices['held']].target.to_numpy(dtype=int)
            held = frame.iloc[indices['held']][['id_student', 'code_module', 'code_presentation']].reset_index(drop=True)
            held = held.merge(metadata, on=['id_student', 'code_module', 'code_presentation'], validate='one_to_one', how='left')
            held['target'], held['probability'] = y, held_p
            rows = []; masks = {'p_ge_0.5': held_p >= .5}
            for capacity in args.capacities:
                # The numerical threshold is frozen on selection; capacity can drift on held.
                threshold = float(np.sort(selection_p)[-max(1, int(np.ceil(len(selection_p) * capacity)))])
                for mode, mask in [('selection_threshold', held_p >= threshold), ('fixed_capacity_ranking', top_fraction(held_p, capacity))]:
                    name = f'{mode}_{capacity:g}'; masks[name] = mask
                    result = scores(y, held_p, mask)
                    # Recall is an average selected indicator among true failures.
                    ci = cluster_interval(mask[y == 1].astype(float), held.id_student.to_numpy()[y == 1], repeats=args.bootstrap)
                    rows.append({'day': day, 'capacity': capacity, 'mode': mode,
                                 'threshold': threshold if mode == 'selection_threshold' else None,
                                 **result, 'recall_ci_low': ci['low'], 'recall_ci_high': ci['high']})
            group_rows = []
            for column in ['code_module', 'code_presentation', 'gender', 'age_band', 'disability', 'imd_band', 'highest_education']:
                for value, positions in held.groupby(column, dropna=False).indices.items():
                    at = np.asarray(positions)
                    for policy, mask in masks.items():
                        group_rows.append({'day': day, 'group_variable': column, 'group': str(value),
                                           'policy': policy, **scores(y[at], held_p[at], mask[at]),
                                           'small_group': len(at) < 100 or int(y[at].sum()) < 20})
            atomic_csv(directory / 'capacity.csv', pd.DataFrame(rows)); atomic_csv(directory / 'subgroups.csv', pd.DataFrame(group_rows))
            checkpoint(directory, identity)
            budgets.append(pd.DataFrame(rows)); subgroups.append(pd.DataFrame(group_rows))
        atomic_csv(output / 'capacity_summary.csv', pd.concat(budgets, ignore_index=True))
        atomic_csv(output / 'subgroup_summary.csv', pd.concat(subgroups, ignore_index=True))


if __name__ == '__main__':
    main()
