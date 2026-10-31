"""Tables 1-5 from Frontiers AI 9:1811886."""
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score)

from config import (TABLES_DIR, SEED, BATCH_TRAIN, BATCH_TEST,
                    KFOLD_SPLITS, KFOLD_EPOCHS)
from data import build_flat_features
from models import StaticBaseline, DynamicBaseline, HybridModel
from training import train_and_evaluate, truncate_dynamic
from protocol import folds


def classifier_scores(clf, X):
    """AUC needs ranking scores, so SVM does not need internal Platt CV."""
    if isinstance(clf, SVC):
        return clf.decision_function(X), clf.predict(X), 'decision_function'
    probability = clf.predict_proba(X)[:, 1]
    return probability, (probability >= 0.5).astype(int), 'probability'


def _save_csv(df, name):
    path = os.path.join(TABLES_DIR, name)
    df.to_csv(path, index=False)
    print(f"   saved {path}")


def _print(title, df):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
    print(df.to_string(index=False))


def table1_comparative(metrics_static, metrics_dynamic, metrics_hybrid):
    df = pd.DataFrame({
        'Metric': ['Accuracy', 'F1-Score (W)', 'ROC-AUC', 'Train Time (s)'],
        'Static Vector':     [metrics_static['Accuracy'],  metrics_static['F1-Score'],
                              metrics_static['ROC-AUC'],   metrics_static['Time']],
        'Behavioral Vector': [metrics_dynamic['Accuracy'], metrics_dynamic['F1-Score'],
                              metrics_dynamic['ROC-AUC'],  metrics_dynamic['Time']],
        'Hybrid Model':      [metrics_hybrid['Accuracy'],  metrics_hybrid['F1-Score'],
                              metrics_hybrid['ROC-AUC'],   metrics_hybrid['Time']],
    }).round(4)
    _print("TABLE 1: Comparative analysis of predictive vectors", df)
    _save_csv(df, 'table1_comparative.csv')
    return df


def table2_baselines(splits, hybrid_metrics):
    """Classical ML baselines on flat features (static + clickstream summaries)."""
    Xtr = build_flat_features(splits['X_s_raw_tr'], splits['X_d_raw_tr'])
    Xte = build_flat_features(splits['X_s_raw_te'], splits['X_d_raw_te'])
    y_tr, y_te = splits['y_tr'], splits['y_te']

    sc = StandardScaler()
    Xtr = sc.fit_transform(Xtr)
    Xte = sc.transform(Xte)

    try:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(eval_metric='logloss', n_estimators=300,
                            random_state=SEED, n_jobs=-1, verbosity=0)
    except ImportError:
        xgb = None
    try:
        from lightgbm import LGBMClassifier
        lgbm = LGBMClassifier(n_estimators=300, random_state=SEED, n_jobs=-1, verbose=-1)
    except ImportError:
        lgbm = None

    candidates = [
        ('Logistic Regression', LogisticRegression(max_iter=2000, n_jobs=-1, random_state=SEED)),
        ('Random Forest',       RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED)),
        ('SVM',                 SVC(kernel='rbf', probability=False, cache_size=1024, random_state=SEED)),
    ]
    if xgb  is not None: candidates.append(('XGBoost',  xgb))
    if lgbm is not None: candidates.append(('LightGBM', lgbm))
    candidates.append(('MLP', MLPClassifier(hidden_layer_sizes=(128, 64),
                                            max_iter=200, random_state=SEED)))

    rows = []
    for name, clf in candidates:
        t0 = time.time()
        clf.fit(Xtr, y_tr)
        probs, preds, score_type = classifier_scores(clf, Xte)
        import joblib
        model_dir = os.path.join(TABLES_DIR, '..', 'checkpoints')
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump({'model': clf, 'scaler': sc}, os.path.join(model_dir, f'{name}-holdout.joblib'))
        pd.DataFrame({'target': y_te, 'score': probs, 'score_type': score_type,
                      'prediction': preds}).to_csv(
            os.path.join(TABLES_DIR, f'{name}-holdout-predictions.csv'), index=False)
        rows.append({
            'Model':        name,
            'ROC-AUC':      roc_auc_score(y_te, probs),
            'Weighted F1':  f1_score(y_te, preds, average='weighted'),
            'Accuracy':     accuracy_score(y_te, preds),
            'Time (s)':     time.time() - t0,
        })

    rows.append({
        'Model':       'Proposed Hybrid (deep)',
        'ROC-AUC':     hybrid_metrics['ROC-AUC'],
        'Weighted F1': hybrid_metrics['F1-Score'],
        'Accuracy':    hybrid_metrics['Accuracy'],
        'Time (s)':    hybrid_metrics['Time'],
    })
    df = pd.DataFrame(rows).round(4)
    _print("TABLE 2: Comparison with baseline ML models", df)
    _save_csv(df, 'table2_baselines.csv')
    return df


def _make_loader(X_s, X_d, y_, batch, shuffle=False):
    ds = TensorDataset(torch.FloatTensor(X_s),
                       torch.FloatTensor(X_d),
                       torch.LongTensor(y_))
    return DataLoader(ds, batch_size=batch, shuffle=shuffle)


def _kfold_classical(name, clf_factory, X_flat, y, splits_n, groups):
    aucs, f1s = [], []
    for fold, (tr, te) in enumerate(folds(y, groups, splits_n, SEED), 1):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X_flat[tr])
        Xte = sc.transform(X_flat[te])
        clf = clf_factory()
        clf.fit(Xtr, y[tr])
        probs, preds, score_type = classifier_scores(clf, Xte)
        aucs.append(roc_auc_score(y[te], probs))
        f1s.append(f1_score(y[te], preds, average='weighted'))
        import joblib
        model_dir = os.path.join(TABLES_DIR, '..', 'checkpoints')
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump({'model': clf, 'scaler': sc}, os.path.join(model_dir, f'{name}-fold{fold}.joblib'))
        pd.DataFrame({'development_row': te, 'target': y[te], 'score': probs,
                      'score_type': score_type, 'prediction': preds}).to_csv(
            os.path.join(TABLES_DIR, f'{name}-fold{fold}-predictions.csv'), index=False)
    return name, np.array(aucs), np.array(f1s)


def _kfold_hybrid(static_dim, seq_len, X_s, X_d, y, splits_n, epochs, groups):
    aucs, f1s = [], []
    for fold, (tr, te) in enumerate(folds(y, groups, splits_n, SEED), 1):
        print(f"  fold {fold}/{splits_n}")
        sc_s = StandardScaler(); sc_d = RobustScaler()
        # Split first, then fit transforms only on the inner training learners.
        fit_rel, val_rel = next(folds(y[tr], groups[tr], 7, SEED + fold))
        fit_idx, val_idx = tr[fit_rel], tr[val_rel]
        Xstr2 = sc_s.fit_transform(X_s[fit_idx]); Xsva = sc_s.transform(X_s[val_idx])
        Xste = sc_s.transform(X_s[te])
        Xdtr2 = sc_d.fit_transform(X_d[fit_idx]); Xdva = sc_d.transform(X_d[val_idx])
        Xdte = sc_d.transform(X_d[te])
        ytr2, yva = y[fit_idx], y[val_idx]
        torch.manual_seed(SEED + fold)
        import joblib
        joblib.dump({'static_scaler': sc_s, 'dynamic_scaler': sc_d,
                     'fit_idx': fit_idx, 'val_idx': val_idx, 'test_idx': te},
                    os.path.join(TABLES_DIR, f'hybrid-fold{fold}-preprocessing.joblib'))

        train_loader = _make_loader(Xstr2, Xdtr2, ytr2, BATCH_TRAIN, shuffle=True)
        val_loader   = _make_loader(Xsva,  Xdva,  yva,  BATCH_TEST)
        test_loader  = _make_loader(Xste,  Xdte,  y[te], BATCH_TEST)

        m = train_and_evaluate(
            f'Hybrid (Balanced)-fold{fold}',
            HybridModel(static_dim, seq_len),
            train_loader, val_loader, test_loader, ytr2,
            epochs=epochs, use_weights=True)
        aucs.append(m['ROC-AUC'])
        f1s.append(m['F1-Score'])
    return 'Proposed Hybrid', np.array(aucs), np.array(f1s)


def table3_kfold(X_s_raw, X_d_raw, y, static_dim, seq_len,
                 splits_n=KFOLD_SPLITS, include_hybrid=True,
                 hybrid_epochs=KFOLD_EPOCHS, groups=None, row_ids=None):
    if groups is None:
        raise ValueError('Student groups are required for article cross-validation')
    fold_id = np.full(len(y), -1)
    for fold, (_, te) in enumerate(folds(y, groups, splits_n, SEED), 1):
        fold_id[te] = fold
    pd.DataFrame({'row': np.arange(len(y)) if row_ids is None else row_ids,
                  'student': groups, 'fold': fold_id, 'target': y}).to_csv(
        os.path.join(TABLES_DIR, 'fold_manifest.csv'), index=False)
    X_flat = build_flat_features(X_s_raw, X_d_raw)
    print(f"\n[k-fold CV with {splits_n} folds] ...")

    try:
        from xgboost import XGBClassifier
        def _xgb(): return XGBClassifier(eval_metric='logloss', n_estimators=300,
                                         random_state=SEED, n_jobs=-1, verbosity=0)
    except ImportError:
        _xgb = None
    try:
        from lightgbm import LGBMClassifier
        def _lgbm(): return LGBMClassifier(n_estimators=300, random_state=SEED,
                                           n_jobs=-1, verbose=-1)
    except ImportError:
        _lgbm = None

    classical = [
        ('Logistic Regression', lambda: LogisticRegression(max_iter=2000, n_jobs=-1, random_state=SEED)),
        ('Random Forest',       lambda: RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED)),
        ('SVM',                 lambda: SVC(kernel='rbf', probability=False, cache_size=1024, random_state=SEED)),
    ]
    if _xgb  is not None: classical.append(('XGBoost',  _xgb))
    if _lgbm is not None: classical.append(('LightGBM', _lgbm))
    classical.append(('MLP', lambda: MLPClassifier(hidden_layer_sizes=(128, 64),
                                                   max_iter=200, random_state=SEED)))

    rows = []
    for name, factory in classical:
        n, aucs, f1s = _kfold_classical(name, factory, X_flat, y, splits_n, groups)
        pd.DataFrame({'fold': np.arange(1, splits_n + 1), 'ROC-AUC': aucs,
                      'Weighted F1': f1s}).to_csv(os.path.join(TABLES_DIR, f'{name}-fold-metrics.csv'), index=False)
        rows.append({'Model': n,
                     'ROC-AUC (mean ± std)': f"{aucs.mean():.4f} ± {aucs.std():.4f}",
                     'Weighted F1 (mean ± std)': f"{f1s.mean():.4f} ± {f1s.std():.4f}"})

    if include_hybrid:
        n, aucs, f1s = _kfold_hybrid(static_dim, seq_len,
                                     X_s_raw, X_d_raw, y, splits_n, hybrid_epochs, groups)
        pd.DataFrame({'fold': np.arange(1, splits_n + 1), 'ROC-AUC': aucs,
                      'Weighted F1': f1s}).to_csv(os.path.join(TABLES_DIR, 'hybrid-fold-metrics.csv'), index=False)
        rows.append({'Model': n,
                     'ROC-AUC (mean ± std)': f"{aucs.mean():.4f} ± {aucs.std():.4f}",
                     'Weighted F1 (mean ± std)': f"{f1s.mean():.4f} ± {f1s.std():.4f}"})

    df = pd.DataFrame(rows)
    _print(f"TABLE 3: Statistical validation ({splits_n}-fold CV)", df)
    _save_csv(df, 'table3_kfold.csv')
    return df


def table4_balancing(metrics_imbal, metrics_hybrid):
    df = pd.DataFrame({
        'Processing Pipeline': ['Baseline (Imbalanced Loss)',
                                'OULAD-Ultimate (Weighted Loss)'],
        'Recall (Class 0 - Fail)': [metrics_imbal['Recall_Risk_0'],
                                    metrics_hybrid['Recall_Risk_0']],
        'Weighted F1-Score':       [metrics_imbal['F1-Score'],
                                    metrics_hybrid['F1-Score']],
        'Class 0 Support':         [metrics_imbal['Full_Report']['0']['support'],
                                    metrics_hybrid['Full_Report']['0']['support']],
    }).round(4)
    _print("TABLE 4: Impact of class balancing (recall analysis)", df)
    _save_csv(df, 'table4_balancing.csv')
    return df


def table5_ablation(splits, full_metrics, static_dim, seq_len,
                    weeks_kept=(18, 23)):
    """Retrain hybrid with truncated dynamic input (weeks 0-15, 0-20).

    Note: course week 0 starts at index ~3 (we have 3 pre-course weeks),
    so 'weeks 0-15' ≈ indices 0..18, 'weeks 0-20' ≈ indices 0..23.
    """
    rows = []

    train_loader = _make_loader(splits['X_s_tr'], splits['X_d_tr'], splits['y_tr'],
                                BATCH_TRAIN, shuffle=True)
    val_loader   = _make_loader(splits['X_s_va'], splits['X_d_va'], splits['y_va'],
                                BATCH_TEST)

    for n_weeks in weeks_kept:
        torch.manual_seed(SEED)
        Xd_tr_t = truncate_dynamic(splits['X_d_tr'], n_weeks)
        Xd_va_t = truncate_dynamic(splits['X_d_va'], n_weeks)
        Xd_te_t = truncate_dynamic(splits['X_d_te'], n_weeks)
        tr = _make_loader(splits['X_s_tr'], Xd_tr_t, splits['y_tr'],
                          BATCH_TRAIN, shuffle=True)
        va = _make_loader(splits['X_s_va'], Xd_va_t, splits['y_va'], BATCH_TEST)
        te = _make_loader(splits['X_s_te'], Xd_te_t, splits['y_te'], BATCH_TEST)
        m = train_and_evaluate(
            f'Ablation-w{n_weeks}',
            HybridModel(static_dim, seq_len),
            tr, va, te, splits['y_tr'], use_weights=True)
        rows.append({
            'Model':       f'Temporal input: {n_weeks} bins (static unchanged)',
            'Weeks used':  '0-15' if n_weeks == 18 else '0-20',
            'ROC-AUC':     m['ROC-AUC'],
            'Weighted F1': m['F1-Score'],
        })

    rows.append({
        'Model':       'Full model',
        'Weeks used':  '0-32',
        'ROC-AUC':     full_metrics['ROC-AUC'],
        'Weighted F1': full_metrics['F1-Score'],
    })

    df = pd.DataFrame(rows).round(4)
    _print("TABLE 5: Temporal-input ablation (retrospective static features)", df)
    _save_csv(df, 'table5_ablation.csv')
    return df
