"""Inner-only model selection, independent calibration, convex ensembles."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
import shutil
import time

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score, log_loss
import torch

from .networks import CLASSICAL, COX, make_model
from .training import Calibration, fit_predict, _score, _tensors, _predict, apply_dropout
from .metrics import metrics, calibration_stats

ENSEMBLE_MEMBERS = ['bilstm', 'lightgbm', 'transformer']
ENSEMBLE_GROUPS = {
    'ensemble_equal': ENSEMBLE_MEMBERS,
    'ensemble_inner_weighted': ENSEMBLE_MEMBERS,
    'ensemble_lstm_attention_lgbm_equal': ['lstm', 'proposed', 'lightgbm'],
    'ensemble_lstm_attention_lgbm_weighted': ['lstm', 'proposed', 'lightgbm'],
}
ENSEMBLES = tuple(ENSEMBLE_GROUPS)
MATCHED_ABLATIONS = ('no_attention', 'no_static', 'no_cox')


def candidates(name):
    if name in COX:
        return [{'cox_weight': weight, 'learning_rate': lr, 'weight_decay': wd, 'dropout_scale': 1.}
                for weight, lr, wd in [(1., .001, 0.), (.001, .001, .0001), (.01, .001, .0001),
                                      (.1, .001, .0001), (.01, .0003, .001), (.1, .0003, .001)]]
    if name not in CLASSICAL:
        return [{'learning_rate': lr, 'weight_decay': wd, 'dropout_scale': dropout}
                for lr, wd, dropout in [(.001, 0., 1.), (.001, .0001, 1.), (.001, .001, 1.25),
                                       (.0003, .0001, 1.), (.0003, .001, 1.25), (.001, .001, .75)]]
    if name == 'lr':
        return [{'model_params': {'C': c}} for c in [.01, .1, 1., 10., 100., .03]]
    if name == 'svm':
        return [{'model_params': {'C': c, 'gamma': gamma}} for c, gamma in
                [(.1, 'scale'), (1., 'scale'), (10., 'scale'), (3., 'scale'), (1., .001), (10., .001)]]
    if name == 'rf':
        return [{'model_params': {'min_samples_leaf': leaf, 'max_features': features, 'max_depth': depth}}
                for leaf, features, depth in [(2, 'sqrt', None), (5, .7, None), (10, .7, None),
                                              (5, .7, 12), (10, 1., 8), (20, 1., 12)]]
    if name == 'mlp':
        return [{'model_params': {'alpha': alpha, 'hidden_layer_sizes': hidden, 'learning_rate_init': lr}}
                for alpha, hidden, lr in [(.01, (128, 64), .001), (.1, (128, 64), .001), (1., (128, 64), .001),
                                         (10., (128, 64), .001), (1., (64, 32), .0003), (10., (64, 32), .0003)]]
    if name == 'xgboost':
        return [{'model_params': {'max_depth': depth, 'min_child_weight': child, 'reg_lambda': reg,
                                  'subsample': .85, 'colsample_bytree': .85}}
                for depth, child, reg in [(3, 5, 5), (4, 5, 5), (6, 5, 10), (3, 10, 10), (4, 10, 20), (6, 10, 20)]]
    if name == 'lightgbm':
        return [{'model_params': {'num_leaves': leaves, 'min_child_samples': child, 'reg_lambda': reg,
                                  'colsample_bytree': .85}}
                for leaves, child, reg in [(15, 30, 1), (31, 30, 5), (15, 60, 5), (31, 60, 10), (63, 60, 10), (15, 100, 10)]]
    raise ValueError(name)


def score_trial(pred, labels):
    # Only the inner selection partition is allowed to rank configurations.
    return (float(roc_auc_score(labels, pred['val_score'])),
            -float(log_loss(labels, np.clip(expit(pred['val_score']), 1e-7, 1 - 1e-7))))


def tune_and_predict(name, fit, selection, calibration_data, held, directory, seed, config, matched=None):
    start = time.perf_counter()
    directory.mkdir(parents=True, exist_ok=True)
    choices = [matched] if matched is not None else candidates(name)[:config['tuning_trials']]
    records, refined = [], []

    def trial(options, stage, number, budget):
        path = directory / 'search' / f'{stage}_{number}'
        settings = {**config, **options, 'epochs': budget, 'plot_learning_curves': stage != 'screen'}
        if config['synthetic']:
            settings['epochs'] = 1
            settings['mlp_iterations'] = 2
        elif stage == 'screen' and name == 'mlp':
            settings['mlp_iterations'] = 60
        marker = path / 'trial_complete.json'
        def checksum(p):
            h = hashlib.sha256()
            with p.open('rb') as f:
                for block in iter(lambda: f.read(1024 * 1024), b''):
                    h.update(block)
            return h.hexdigest()
        complete = False
        if marker.exists():
            saved = json.loads(marker.read_text())
            complete = all((path / p).exists() and checksum(path / p) == h
                           for p, h in saved.get('artifacts', {}).items()) and bool(saved.get('artifacts'))
        if complete:
            report = json.loads((path / 'training.json').read_text())
            pred = dict(np.load(path / 'predictions.npz'))
        else:
            # Outer held labels/features and calibration labels are not passed to search.
            pred, report = fit_predict(name, fit, selection, selection, None, path, seed, settings)
            report.update(trial_role='inner_search_only', held_partition_role='inner_selection_copy_not_outer_held',
                          independent_calibration_used=False)
            (path / 'training.json').write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')
            marker.write_text(json.dumps({'stage': stage, 'options': options,
                'artifacts': {p.name: checksum(p) for p in path.iterdir() if p.is_file() and p != marker}}), encoding='utf-8')
        auc, negative_loss = score_trial(pred, selection['y'])
        record = {'stage': stage, 'trial': number, 'selection_auc': auc, 'selection_log_loss': -negative_loss,
                  'options': options, 'directory': str(path), 'settings': settings}
        records.append(record)
        pd.json_normalize(records).to_csv(directory / 'inner_search.csv', index=False)
        return record

    if name in CLASSICAL or len(choices) == 1:
        for i, options in enumerate(choices):
            refined.append(trial(options, 'full', i, config['epochs']))
    else:
        screened = [trial(options, 'screen', i, min(config['screen_epochs'], config['epochs']))
                    for i, options in enumerate(choices)]
        shortlist = sorted(screened, key=lambda r: (r['selection_auc'], -r['selection_log_loss']), reverse=True)[:2]
        for i, record in enumerate(shortlist):
            refined.append(trial(record['options'], 'full', i, config['epochs']))
    winner = max(refined, key=lambda r: (r['selection_auc'], -r['selection_log_loss']))
    source = Path(winner['directory'])
    for p in source.iterdir():
        if p.is_file() and p.name != 'trial_complete.json':
            shutil.copy2(p, directory / p.name)
    pred, report = finalize_selected(name, directory, fit, selection, calibration_data, held, seed, winner['settings'])
    report.update(seconds=time.perf_counter() - start, selected_configuration=winner['options'],
                  inner_selection={'criterion': 'selection ROC-AUC, log loss breaks ties',
                                   'winner': winner, 'trials': len(records), 'held_used_for_search': False,
                                   'calibration_used_for_search': False})
    (directory / 'training.json').write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')
    return pred, report


def finalize_selected(name, directory, fit, selection, calibration_data, held, seed, config):
    report = json.loads((directory / 'training.json').read_text())
    source_pred = dict(np.load(directory / 'predictions.npz'))
    score_fit, score_selection = source_pred['fit_score'], source_pred['val_score']
    risk_held = None
    if name in CLASSICAL:
        model = joblib.load(directory / 'model.joblib')
        score_cal, score_held = _score(model, calibration_data['flat']), _score(model, held['flat'])
    else:
        device = torch.device(config['device'])
        checkpoint = torch.load(directory / 'model.pt', map_location=device, weights_only=False)
        model = make_model(name, fit['x'].shape[2], fit['s'].shape[1], fit['x'].shape[1]).to(device)
        apply_dropout(model, config.get('dropout_scale', 1.))
        model.load_state_dict(checkpoint['state_dict'])
        cal_gpu = _tensors(calibration_data, device)
        score_cal, _ = _predict(model, cal_gpu)
        del cal_gpu
        held_gpu = _tensors(held, device)
        score_held, risk = _predict(model, held_gpu)
        del held_gpu
        if name in COX:
            stats = checkpoint['parameters']['risk_standardization']
            risk_held = (risk - stats['training_mean']) / stats['training_std']
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    calibrator = Calibration().fit(score_cal, calibration_data['y'])
    joblib.dump(calibrator, directory / 'calibration.joblib')
    pred = {'held_score': score_held, 'held_probability': calibrator.predict(score_held),
            'val_score': score_selection, 'val_target': selection['y'],
            'calibration_score': score_cal, 'calibration_target': calibration_data['y'],
            'fit_score': score_fit, 'fit_target': fit['y']}
    if risk_held is not None:
        pred['held_risk'] = risk_held
    np.savez_compressed(directory / 'predictions.npz', **pred)
    rows, bins = [], []
    for split, labels, score in [('train', fit['y'], score_fit), ('selection_validation', selection['y'], score_selection),
                                  ('calibration_fit', calibration_data['y'], score_cal), ('held', held['y'], score_held)]:
        for stage, p in [('raw', expit(score)), ('calibrated', calibrator.predict(score))]:
            measures, reliability = calibration_stats(labels, p)
            rows.append({'split': split, 'stage': stage, **measures})
            bins.extend({'split': split, 'stage': stage, **b} for b in reliability)
    pd.DataFrame(rows).to_csv(directory / 'calibration_metrics.csv', index=False)
    pd.DataFrame(bins).to_csv(directory / 'calibration_bins.csv', index=False)
    report['n_held'] = len(held['y'])
    report['calibration'].update(fit_partition='independent_inner_calibration', n=len(calibration_data['y']),
        coefficient=float(calibrator.model.coef_[0, 0]), intercept=float(calibrator.model.intercept_[0]),
        score_mean=float(calibrator.scaler.mean_[0]), score_std=float(calibrator.scaler.scale_[0]))
    report['validation_role'] = 'selection_only; independent of calibration and outer held'
    report.update(trial_role='selected_final_model', held_partition_role='outer_held_validation',
                  independent_calibration_used=True)
    return pred, report


def fit_ensemble(name, predictions, selection_y, calibration_y, held_y, directory):
    """Weights learned only on inner selection; Platt fit only on separate calibration."""
    start = time.perf_counter()
    directory.mkdir(parents=True, exist_ok=True)
    members = ENSEMBLE_GROUPS[name]
    selection = np.column_stack([expit(predictions[m]['val_score']) for m in members])
    cal = np.column_stack([expit(predictions[m]['calibration_score']) for m in members])
    held = np.column_stack([expit(predictions[m]['held_score']) for m in members])
    equal = np.full(len(members), 1 / len(members))
    weights = equal
    if name.endswith('weighted'):
        objective = lambda w: log_loss(selection_y, np.clip(selection @ w, 1e-7, 1 - 1e-7)) + .01 * ((w - equal) ** 2).sum()
        result = minimize(objective, equal, method='SLSQP', bounds=[(0, 1)] * len(equal),
                          constraints=[{'type': 'eq', 'fun': lambda w: w.sum() - 1}], options={'maxiter': 100})
        if not result.success:
            raise RuntimeError(f'Ensemble optimization failed: {result.message}')
        weights = np.maximum(result.x, 0)
        weights /= weights.sum()
    score_cal, score_held = [logit(np.clip(a @ weights, 1e-6, 1 - 1e-6)) for a in (cal, held)]
    calibration = Calibration().fit(score_cal, calibration_y)
    pred = {'held_score': score_held, 'held_probability': calibration.predict(score_held),
            'val_score': logit(np.clip(selection @ weights, 1e-6, 1 - 1e-6)),
            'calibration_score': score_cal, 'calibration_target': calibration_y}
    np.savez_compressed(directory / 'predictions.npz', **pred)
    joblib.dump(calibration, directory / 'calibration.joblib')
    report = {'model': name, 'seconds': 0., 'parameter_count': len(weights), 'gpu_peak_allocated_bytes': 0,
              'members': members, 'weights': weights.tolist(), 'diagnostics': {},
              'weights_fitted_on': 'inner_selection', 'calibration_fitted_on': 'independent_inner_calibration',
              'outer_held_used_for_weights_or_calibration': False, 'seed': None, 'warnings': []}
    (directory / 'training.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    rows, bins = [], []
    for stage, p in [('raw', expit(score_held)), ('calibrated', pred['held_probability'])]:
        measures, reliability = calibration_stats(held_y, p)
        rows.append({'split': 'held', 'stage': stage, **measures})
        bins.extend({'split': 'held', 'stage': stage, **b} for b in reliability)
    pd.DataFrame(rows).to_csv(directory / 'calibration_metrics.csv', index=False)
    pd.DataFrame(bins).to_csv(directory / 'calibration_bins.csv', index=False)
    report['seconds'] = time.perf_counter() - start
    (directory / 'training.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return pred, report
