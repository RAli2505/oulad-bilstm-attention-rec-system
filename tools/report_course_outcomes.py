"""Fast read-only post-hoc class metrics from completed 2013 validation forecasts."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support, roc_auc_score)


def class_metrics(y, p):
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Expected binary targets and finite probabilities in [0, 1]')
    predicted = (p >= .5).astype(int)
    precision, recall, f1, support = precision_recall_fscore_support(y, predicted, labels=[0, 1], zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    result = {'n': len(y), 'accuracy': accuracy_score(y, predicted),
              'macro_f1': f1_score(y, predicted, average='macro', zero_division=0),
              'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}
    for label, name in [(0, 'success'), (1, 'failure')]:
        target, probability = (1 - y, 1 - p) if label == 0 else (y, p)
        result.update({f'precision_{name}': float(precision[label]), f'recall_{name}': float(recall[label]),
                       f'f1_{name}': float(f1[label]), f'support_{name}': int(support[label]),
                       f'pr_auc_ap_{name}': float(average_precision_score(target, probability)),
                       f'roc_auc_{name}': float(roc_auc_score(target, probability))})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--risk-run', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    source = args.risk_run.resolve()
    output = args.output or source.parent / ('outcomes_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    output = output.resolve()
    if output == source or output.is_relative_to(source):
        raise ValueError('Report must be outside the immutable source run')
    manifest_path = source / 'run_manifest.json'
    before = manifest_path.read_bytes()
    manifest = json.loads(before)
    if manifest['status'] != 'completed' or manifest['config']['evaluate_test']:
        raise ValueError('Expected completed validation-only source run')
    config = manifest['config']
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    rows, audit, across_seeds = [], [], []
    with (output / 'report.log').open('x', encoding='utf-8') as log:
        for index, day in enumerate(config['windows'], 1):
            directory = source / f'day_{day}'
            if json.loads((directory / 'status.json').read_text())['status'] != 'completed':
                raise ValueError(f'Incomplete day {day}')
            for seed in config['seeds']:
                for model in config['models']:
                    job = directory / 'fold_1' / f'seed_{seed}' / model
                    status = json.loads((job / 'status.json').read_text())
                    if status['status'] != 'completed' or not (job / 'predictions.npz').is_file():
                        raise ValueError(f'Incomplete job: {job}')
                    audit.append({'day': day, 'seed': seed, 'model': model, 'status': 'completed'})
            data = pd.read_csv(directory / 'validation_2013_predictions.csv')
            for model in config['models']:
                for stage, column in [('calibrated', f'{model}_probability'), ('raw', f'{model}_raw_probability')]:
                    rows.append({'day': day, 'model': model, 'stage': stage,
                                 'aggregation': 'probabilities averaged across seeds',
                                 **class_metrics(data.target.to_numpy(), data[column].to_numpy())})
            across_seeds.append(pd.read_csv(directory / 'validation_across_seeds.csv').assign(day=day))
            elapsed = time.perf_counter() - started
            message = f'Outcomes: {index}/{len(config["windows"])} windows; day={day}; elapsed={elapsed:.1f}s; ETA={elapsed / index * (len(config["windows"]) - index):.1f}s'
            print(message, flush=True)
            log.write(message + '\n'); log.flush()
    pd.DataFrame(rows).to_csv(output / 'class_metrics.csv', index=False)
    pd.DataFrame(audit).to_csv(output / 'completed_jobs.csv', index=False)
    pd.concat(across_seeds, ignore_index=True).to_csv(output / 'existing_seed_metric_summary.csv', index=False)
    if manifest_path.read_bytes() != before:
        raise RuntimeError('Source manifest changed during reporting')
    (output / 'manifest.json').write_text(json.dumps({
        'source': str(source), 'source_manifest_sha256': hashlib.sha256(before).hexdigest(),
        'status': 'completed', 'training': False, 'test_used': False,
        'completed_jobs': len(audit), 'windows': config['windows'], 'seeds': config['seeds'],
        'threshold': 'failure if p_failure >= 0.5; success otherwise',
        'success': 'Pass + Distinction', 'failure': 'Fail + Withdrawn',
        'aggregation': 'class_metrics uses probability averages, not mean per-seed metrics',
        'elapsed_seconds': time.perf_counter() - started}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Finished report: {output}', flush=True)


if __name__ == '__main__':
    main()
