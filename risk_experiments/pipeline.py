from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.model_selection import StratifiedGroupKFold

from .features import FoldTransform, build_window, temporal_split, KEYS
from .metrics import metrics, bootstrap_auc, make_plots, calibration_stats
from .networks import MODELS
from .training import fit_predict
from .enhanced_features import enhanced_window, EnhancedTransform
from .tuning import ENSEMBLES, MATCHED_ABLATIONS, tune_and_predict, fit_ensemble


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str), encoding='utf-8')


def grouped_folds(y, groups, seed, n=5):
    splitter = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed)
    for fit, held in splitter.split(np.zeros(len(y)), y, groups):
        assert not set(groups[fit]) & set(groups[held])
        if len(np.unique(y[fit])) != 2 or len(np.unique(y[held])) != 2:
            raise ValueError('A fold has only one class; choose a sufficiently large cohort')
        yield fit, held


def holdout_student_groups(target, seed):
    """Fixed 70/30 learner split before landmark eligibility filtering."""
    labels = target.final_result.isin(['Fail', 'Withdrawn']).to_numpy(dtype=int)
    groups = target.id_student.to_numpy()
    partitions = list(grouped_folds(labels, groups, seed, n=10))
    validation_rows = np.concatenate([held for _, held in partitions[:3]])
    validation_groups = np.unique(groups[validation_rows])
    return validation_groups


def permutation_p(y, proposed, reference, groups, repeats, seed=42):
    from sklearn.metrics import roc_auc_score
    _, inverse = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(seed)
    observed = roc_auc_score(y, proposed) - roc_auc_score(y, reference)
    extreme = 0
    for _ in range(repeats):
        swap = rng.integers(0, 2, inverse.max() + 1)[inverse].astype(bool)
        p, q = np.where(swap, reference, proposed), np.where(swap, proposed, reference)
        extreme += abs(roc_auc_score(y, p) - roc_auc_score(y, q)) >= abs(observed)
    return (extreme + 1) / (repeats + 1)


def run_window(target, daily, assess, config, output: Path, signature: str, prepare_only=False):
    output.mkdir(parents=True, exist_ok=True)
    improved = config.get('improved', False)
    holdout = config.get('split_scheme') == 'holdout'
    names = list(config.get('models', MODELS))
    evaluate_test = config.get('evaluate_test', False)
    if not evaluate_test:
        # Do not construct 2014 features, predictions, or label-based reports.
        presentations = ['2013B', '2013J']
        target = target[target.code_presentation.isin(presentations)]
        daily = daily[daily.code_presentation.isin(presentations)]
        assess = assess[assess.code_presentation.isin(presentations)]
    fixed_validation_groups = holdout_student_groups(target, config['split_seed']) if holdout else None
    builder = enhanced_window if config.get('feature_profile') == 'rich' else build_window
    window = builder(target, daily, assess, config['days'], config['score_lag'], config['score_policy'])
    dev, test, overlap = temporal_split(window)
    if not len(dev) or (evaluate_test and not len(test)):
        raise ValueError('Development cohort is required; test cohort is required only when enabled')
    roles = np.full(len(window.frame), 'excluded_other_or_overlap', dtype=object)
    roles[dev], roles[test] = 'development_2013', 'test_2014'
    manifest = window.frame[KEYS].copy()
    manifest['target'], manifest['time_from_landmark'], manifest['event_withdrawal'] = window.y, window.time, window.event
    manifest['role'] = roles
    manifest['oof_fold'] = -1
    if holdout:
        validation_mask = np.isin(window.groups[dev], fixed_validation_groups)
        outer = [(np.flatnonzero(~validation_mask), np.flatnonzero(validation_mask))]
        if len(np.unique(window.y[dev][outer[0][1]])) != 2:
            raise ValueError('Validation cohort must contain both outcome classes')
        manifest.loc[dev[outer[0][0]], 'role'] = 'train_pool_2013'
        manifest.loc[dev[outer[0][1]], 'role'] = 'validation_2013'
    else:
        outer = list(grouped_folds(window.y[dev], window.groups[dev], config['split_seed']))
    evaluation_rel = np.sort(np.concatenate([held for _, held in outer]))
    evaluation_ids = dev[evaluation_rel]
    for fold, (_, held_rel) in enumerate(outer, 1):
        manifest.loc[dev[held_rel], 'oof_fold'] = fold
    manifest.to_csv(output / 'cohort_and_folds.csv', index=False)
    characteristics = manifest.copy()
    characteristics['zero_history'] = window.counts.sum(axis=(1, 2)) == 0
    characteristics.groupby(['role', 'code_module', 'code_presentation']).agg(
        enrollments=('id_student', 'size'), students=('id_student', 'nunique'),
        positive_outcomes=('target', 'sum'), withdrawals=('event_withdrawal', 'sum'),
        zero_history=('zero_history', 'sum')).to_csv(output / 'cohort_characteristics.csv')
    numeric = ['num_of_prev_attempts', 'studied_credits', 'date_registration', 'n_submitted', 'mean_score']
    window.frame[numeric].describe().to_csv(output / 'numeric_descriptives.csv')
    window.frame.isna().sum().rename('missing_count').to_csv(output / 'missingness.csv')
    write_json(output / 'cohort_audit.json', {**window.audit, 'development': len(dev), 'external_test': len(test),
                                            'test_removed_repeated_2013_students': overlap,
                                            '2013_prevalence': float(window.y[dev].mean()),
                                            'split_scheme': 'single_grouped_holdout' if holdout else 'five_fold_grouped_cv',
                                            'evaluation_validation_records': len(evaluation_ids),
                                            'actual_validation_fraction': len(evaluation_ids) / len(dev),
                                            'nominal_validation_fraction': .3 if holdout else 1.,
                                            'external_test_evaluated': evaluate_test})
    print(f"Day {window.days}: development={len(dev)}, external test={len(test)}, zero-history={window.audit['zero_history_enrollments']}", flush=True)
    if prepare_only:
        return
    nseed = len(config['seeds'])
    oof = {name: np.full((nseed, len(dev)), np.nan) for name in names}
    test_sum = {name: np.zeros((nseed, len(test))) for name in names}
    oof_raw = {name: np.full((nseed, len(dev)), np.nan) for name in names}
    test_raw = {name: np.zeros((nseed, len(test))) for name in names}
    oof_risk, test_risk = {}, {}
    fold_rows = []
    diagnostic_rows = []
    for fold, (outer_fit_rel, held_rel) in enumerate(outer, 1):
        outer_fit, held_idx = dev[outer_fit_rel], dev[held_rel]
        fit_rel, val_rel = next(grouped_folds(window.y[outer_fit], window.groups[outer_fit], config['split_seed'] + fold))
        fit_idx, val_idx = outer_fit[fit_rel], outer_fit[val_rel]
        calibration_idx = None
        if improved:
            selection_rel, calibration_rel = next(grouped_folds(window.y[val_idx], window.groups[val_idx],
                                                                config['split_seed'] + 100 + fold, n=2))
            calibration_idx, val_idx = val_idx[calibration_rel], val_idx[selection_rel]
            assert not set(window.groups[val_idx]) & set(window.groups[calibration_idx])
        fold_dir = output / f'fold_{fold}'
        fold_dir.mkdir(exist_ok=True)
        np.savez_compressed(fold_dir / 'indices.npz', fit=fit_idx, validation=val_idx, held=held_idx,
                            calibration=calibration_idx if improved else np.array([], dtype=int), external_test=test)
        transform_type = EnhancedTransform if config.get('feature_profile') == 'rich' else FoldTransform
        transform = transform_type().fit(window, fit_idx)
        joblib.dump(transform, fold_dir / 'preprocessing.joblib')
        static_names = transform.static.get_feature_names_out().tolist()
        write_json(fold_dir / 'features.json', {'activity_channels': transform.channel_names,
                   'static_features': static_names,
                   'classical_feature_names': static_names + [f'{stat}:{channel}' for stat in
                        ['log_total', 'log_mean', 'log_std', 'log_max', 'log_last7', 'log_previous7', 'log_last7_minus_log_previous7']
                        for channel in transform.channel_names],
                   'sequence_transform': 'log1p, per-channel train-only mean/std',
                   'classical_features': 'same static + per-channel total/mean/std/max/last7/previous7/trend',
                   'paper_features': 'same static + observed typed channels summed into weekly total counts',
                   'normalization_fitted_on': 'inner_fit_only'})
        partitions = [transform.transform(window, idx) for idx in (fit_idx, val_idx, held_idx)]
        partitions.append(transform.transform(window, test) if evaluate_test else None)
        calibration_partition = transform.transform(window, calibration_idx) if improved else None
        for si, seed in enumerate(config['seeds']):
            fold_predictions = {}
            selected_main = None
            order = ['proposed'] + [n for n in names if n != 'proposed'] if improved else names
            for name in order:
                directory = fold_dir / f'seed_{seed}' / name
                status_path = directory / 'status.json'
                complete = False
                if status_path.exists():
                    previous = json.loads(status_path.read_text())
                    complete = previous.get('signature') == signature and previous.get('status') == 'completed'
                    if complete:
                        complete = all((directory / p).exists() and sha256(directory / p) == h
                                       for p, h in previous['artifacts_sha256'].items())
                start = time.perf_counter()
                if complete:
                    pred = dict(np.load(directory / 'predictions.npz'))
                    report = json.loads((directory / 'training.json').read_text())
                    print(f'Day {window.days} fold {fold} seed {seed} {name}: reuse', flush=True)
                else:
                    directory.mkdir(parents=True, exist_ok=True)
                    write_json(status_path, {'status': 'running', 'signature': signature})
                    print(f'Day {window.days} fold {fold} seed {seed} {name}: training', flush=True)
                    if improved and name in ENSEMBLES:
                        pred, report = fit_ensemble(name, fold_predictions, partitions[1]['y'],
                                                   calibration_partition['y'], partitions[2]['y'], directory)
                    elif improved:
                        matched = selected_main if name in MATCHED_ABLATIONS else None
                        if name in MATCHED_ABLATIONS and matched is None:
                            raise RuntimeError('Matched ablations require the selected main configuration')
                        pred, report = tune_and_predict(name, partitions[0], partitions[1], calibration_partition,
                                                       partitions[2], directory, seed, config, matched=matched)
                    else:
                        pred, report = fit_predict(name, *partitions, directory, seed, config)
                    artifacts = {p.relative_to(directory).as_posix(): sha256(p) for p in directory.rglob('*')
                                 if p.is_file() and p != status_path}
                    write_json(status_path, {'status': 'completed', 'signature': signature, 'artifacts_sha256': artifacts})
                    print(f'  completed in {time.perf_counter() - start:.1f}s', flush=True)
                fold_predictions[name] = pred
                if improved and name == 'proposed':
                    selected_main = report['selected_configuration']
                oof[name][si, held_rel] = pred['held_probability']
                if evaluate_test:
                    test_sum[name][si] += pred['test_probability'] / len(outer)
                oof_raw[name][si, held_rel] = expit(pred['held_score'])
                if evaluate_test:
                    test_raw[name][si] += expit(pred['test_score']) / len(outer)
                diagnostic_rows.append({'model': name, 'fold': fold, 'seed': seed,
                    **{k: v for k, v in report['diagnostics'].items() if not isinstance(v, dict)}})
                pd.DataFrame(diagnostic_rows).to_csv(output / 'fit_diagnostics.csv', index=False)
                if 'held_risk' in pred:
                    if name not in oof_risk:
                        oof_risk[name] = np.full((nseed, len(dev)), np.nan)
                        test_risk[name] = np.zeros((nseed, len(test)))
                    oof_risk[name][si, held_rel] = pred['held_risk']
                    if evaluate_test:
                        test_risk[name][si] += pred['test_risk'] / len(outer)
                fold_rows.append({'model': name, 'fold': fold, 'seed': seed,
                                  **metrics(window.y[held_idx], pred['held_probability'], pred.get('held_risk'),
                                            window.time[held_idx], window.event[held_idx]),
                                  'raw_roc_auc': metrics(window.y[held_idx], expit(pred['held_score']))['roc_auc'],
                                  'training_seconds': report['seconds'],
                                  'parameter_count': report['parameter_count'],
                                  'gpu_peak_bytes': report['gpu_peak_allocated_bytes']})
                pd.DataFrame(fold_rows).to_csv(output / 'fold_metrics.csv', index=False)
        del partitions
    for name in names:
        assert np.isfinite(oof[name][:, evaluation_rel]).all(), f'Incomplete validation coverage for {name}'
    pd.DataFrame(fold_rows).groupby('model').agg(
        roc_auc_mean=('roc_auc', 'mean'), roc_auc_std=('roc_auc', 'std'),
        pr_auc_mean=('pr_auc', 'mean'), pr_auc_std=('pr_auc', 'std'),
        log_loss_mean=('log_loss', 'mean'), log_loss_std=('log_loss', 'std'),
        mean_training_seconds=('training_seconds', 'mean')).to_csv(
            output / ('validation_across_seeds.csv' if holdout else 'cv_mean_std.csv'))
    # External labels are used for metrics only after all predeclared models are fitted.
    all_summary, comparisons, calibration_rows, bin_rows = [], [], [], []
    validation_split = 'validation_2013' if holdout else 'oof_2013'
    evaluation_sets = [(validation_split, evaluation_ids,
                        {k: v[:, evaluation_rel] for k, v in oof.items()},
                        {k: v[:, evaluation_rel] for k, v in oof_risk.items()},
                        {k: v[:, evaluation_rel] for k, v in oof_raw.items()})]
    if evaluate_test:
        evaluation_sets.append(('test_2014', test, test_sum, test_risk, test_raw))
    for split_name, ids, probabilities, risks, raw_probabilities in evaluation_sets:
        means = {name: a.mean(axis=0) for name, a in probabilities.items()}
        prediction_table = window.frame.iloc[ids][KEYS].copy()
        prediction_table['target'], prediction_table['time'], prediction_table['event'] = window.y[ids], window.time[ids], window.event[ids]
        for name, p in means.items():
            prediction_table[f'{name}_probability'] = p
            raw_probability = raw_probabilities[name].mean(axis=0)
            prediction_table[f'{name}_raw_probability'] = raw_probability
            for stage, probability in [('raw', raw_probability), ('calibrated', p)]:
                measures, bins = calibration_stats(window.y[ids], probability)
                calibration_rows.append({'split': split_name, 'model': name, 'stage': stage, **measures})
                bin_rows.extend({'split': split_name, 'model': name, 'stage': stage, **b} for b in bins)
            risk = risks[name].mean(axis=0) if name in risks else None
            if risk is not None:
                prediction_table[f'{name}_withdrawal_risk'] = risk
            result = metrics(window.y[ids], p, risk, window.time[ids], window.event[ids])
            ci = bootstrap_auc(window.y[ids], p, window.groups[ids], config['bootstrap'])
            all_summary.append({'split': split_name, 'model': name, **result,
                                'roc_auc_ci_low': ci[0], 'roc_auc_ci_high': ci[1]})
            if name != 'proposed':
                ci_delta = bootstrap_auc(window.y[ids], means['proposed'], window.groups[ids],
                                         config['bootstrap'], comparator=p)
                comparison = {'split': split_name, 'reference': name,
                              'delta_auc_proposed_minus_reference': metrics(window.y[ids], means['proposed'])['roc_auc'] - result['roc_auc'],
                              'delta_auc_ci_low': ci_delta[0], 'delta_auc_ci_high': ci_delta[1],
                              'confirmatory': bool(split_name == 'test_2014' and name == 'no_cox' and window.days == config['primary_day'])}
                if comparison['confirmatory']:
                    comparison['learner_cluster_permutation_p'] = permutation_p(window.y[ids], means['proposed'], p,
                                                              window.groups[ids], config['permutations'])
                comparisons.append(comparison)
        prediction_table.to_csv(output / f'{split_name}_predictions.csv', index=False)
        plot_dir = output / split_name
        plot_dir.mkdir(exist_ok=True)
        make_plots(window.y[ids], means, plot_dir, f'Day {window.days}: {split_name}')
    pd.DataFrame(all_summary).to_csv(output / 'summary.csv', index=False)
    pd.DataFrame(comparisons).to_csv(output / 'paired_comparisons.csv', index=False)
    pd.DataFrame(calibration_rows).to_csv(output / 'calibration_before_after.csv', index=False)
    pd.DataFrame(bin_rows).to_csv(output / 'reliability_bins.csv', index=False)
    write_json(output / 'selection_on_development.json', {
        'criterion': 'fixed validation ROC-AUC' if holdout else 'pooled OOF ROC-AUC, averaged across predeclared seeds',
        'ranking': sorted([r for r in all_summary if r['split'] == validation_split], key=lambda r: r['roc_auc'], reverse=True),
        'note': 'Descriptive development ranking; the primary proposed-vs-no_cox contrast remains predeclared regardless of rank.'})
    write_json(output / 'status.json', {'status': 'completed', 'signature': signature})


def synthetic_data():
    """Synthetic full pipeline proof; never mixed with publication results."""
    rng = np.random.default_rng(3)
    rows, daily, assessments = [], [], []
    for i in range(400):
        sid = i // 2
        presentation = '2013J' if i < 280 else '2014J'
        module = 'AAA' if i % 2 else 'BBB'
        result = ['Pass', 'Fail', 'Withdrawn', 'Distinction'][sid % 4]
        rows.append(dict(id_student=sid, code_module=module, code_presentation=presentation,
                         final_result=result, date_registration=-10., date_unregistration=80. if result == 'Withdrawn' else np.nan,
                         module_presentation_length=240., gender='M' if sid % 2 else 'F', region='R1',
                         highest_education='HE', imd_band='0-10%', age_band='0-35', disability='N',
                         num_of_prev_attempts=sid % 3, studied_credits=60))
        for day in (0, 5, 20, 40):
            if sid % 9:
                daily.append(dict(id_student=sid, code_module=module, code_presentation=presentation,
                                  date=day, activity_type='forumng' if day % 2 else 'oucontent', sum_click=int(rng.integers(1, 10))))
        for aid, day in enumerate((10, 45)):
            assessments.append(dict(id_student=sid, code_module=module, code_presentation=presentation,
                                    id_assessment=i * 2 + aid, date_submitted=day, date=day,
                                    is_banked=0, score=float(rng.integers(30, 100))))
    return pd.DataFrame(rows), pd.DataFrame(daily), pd.DataFrame(assessments)
