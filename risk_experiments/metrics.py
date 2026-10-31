from __future__ import annotations

import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score, log_loss,
                             brier_score_loss, f1_score, precision_score, recall_score,
                             balanced_accuracy_score, roc_curve, precision_recall_curve)


def concordance(risk, time, event):
    """Chunked Harrell C, no quadratic allocation for the external test cohort."""
    event_ids = np.flatnonzero(event)
    numerator = denominator = 0.
    for chunk in np.array_split(event_ids, max(1, (len(event_ids) + 127) // 128)):
        comparable = time[chunk, None] < time[None, :]
        denominator += comparable.sum()
        numerator += ((risk[chunk, None] > risk[None, :]) * comparable).sum()
        numerator += .5 * ((risk[chunk, None] == risk[None, :]) * comparable).sum()
    return float(numerator / denominator) if denominator else None


def metrics(y, probability, risk=None, time=None, event=None):
    p = np.clip(probability, 1e-7, 1 - 1e-7)
    classes = p >= .5
    ece = 0.
    for i in range(10):
        mask = (p >= i / 10) & (p < (i + 1) / 10)
        if mask.any():
            ece += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return {'roc_auc': float(roc_auc_score(y, p)), 'pr_auc': float(average_precision_score(y, p)),
            'log_loss': float(log_loss(y, p)), 'brier': float(brier_score_loss(y, p)),
            'f1_risk': float(f1_score(y, classes, zero_division=0)),
            'precision_risk': float(precision_score(y, classes, zero_division=0)),
            'recall_risk': float(recall_score(y, classes, zero_division=0)),
            'balanced_accuracy': float(balanced_accuracy_score(y, classes)), 'ece_10': float(ece),
            'c_index_withdrawal': concordance(risk, time, event) if risk is not None else None,
            'n': len(y), 'prevalence': float(np.mean(y))}


def bootstrap_auc(y, p, groups, repeats, seed=42, comparator=None):
    """Learner-cluster percentile CI, paired when comparator is provided."""
    _, inverse = np.unique(groups, return_inverse=True)
    n_groups = inverse.max() + 1
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        weights = np.bincount(rng.integers(n_groups, size=n_groups), minlength=n_groups)[inverse]
        if not (weights[y == 0].sum() and weights[y == 1].sum()):
            continue
        value = roc_auc_score(y, p, sample_weight=weights)
        if comparator is not None:
            value -= roc_auc_score(y, comparator, sample_weight=weights)
        values.append(value)
    return [float(v) for v in np.quantile(values, [.025, .975])] if values else [None, None]


def calibration_stats(y, probability):
    """Diagnostic intercept/slope, never applied back to predictions."""
    from sklearn.linear_model import LogisticRegression
    p = np.clip(probability, 1e-6, 1 - 1e-6)
    logits = np.log(p / (1 - p))
    if np.std(logits) < 1e-8:
        intercept, slope = None, None
    else:
        diagnostic = LogisticRegression(C=1e6, max_iter=2000).fit(logits[:, None], y)
        intercept, slope = float(diagnostic.intercept_[0]), float(diagnostic.coef_[0, 0])
    bins = []
    for i in range(10):
        mask = (p >= i / 10) & (p < (i + 1) / 10)
        bins.append({'bin_lower': i / 10, 'bin_upper': (i + 1) / 10, 'n': int(mask.sum()),
                     'predicted_mean': float(p[mask].mean()) if mask.any() else None,
                     'observed_frequency': float(y[mask].mean()) if mask.any() else None})
    measures = metrics(y, p)
    return {k: measures[k] for k in ('brier', 'log_loss', 'ece_10')} | {
        'calibration_intercept': intercept, 'calibration_slope': slope}, bins


def make_plots(y, predictions, directory, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    for kind in ('roc', 'pr', 'calibration'):
        fig, ax = plt.subplots(figsize=(9, 7))
        for name, p in predictions.items():
            if kind == 'roc':
                x, v, _ = roc_curve(y, p)
            elif kind == 'pr':
                v, x, _ = precision_recall_curve(y, p)
            else:
                v, x = calibration_curve(y, p, n_bins=10, strategy='quantile')
            ax.plot(x, v, label=name)
        ax.set_title(title)
        ax.set_xlabel('False positive rate' if kind == 'roc' else 'Recall' if kind == 'pr' else 'Predicted probability')
        ax.set_ylabel('True positive rate' if kind == 'roc' else 'Precision' if kind == 'pr' else 'Observed risk')
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(directory / f'{kind}.png', dpi=180)
        plt.close(fig)
