"""Explicit ranking, probability, uncertainty and resource-use measurements."""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
import threading
import time

import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, roc_auc_score, log_loss,
                             precision_recall_curve, auc, confusion_matrix)


def ranking_metrics(ranked, relevant, k):
    if k < 1 or len(set(ranked)) != len(ranked):
        raise ValueError('Positive K and unique recommendations required')
    truth = set(relevant)
    hit = np.array([int(i in truth) for i in ranked[:k]], dtype=float)
    n = len(truth)
    hits = int(hit.sum())
    precision, recall = hits / k, hits / n if n else 0.
    positions = np.arange(1, len(hit) + 1)
    ideal = np.sum(1 / np.log2(np.arange(1, min(n, k) + 1) + 1))
    dcg = float(np.sum(hit / np.log2(positions + 1)))
    first = np.flatnonzero(hit)
    return {'precision': precision, 'recall': recall,
            'f1': 2 * precision * recall / (precision + recall) if precision + recall else 0.,
            'ndcg': dcg / ideal if ideal else 0.,
            'map': float(np.sum(np.cumsum(hit) / positions * hit) / min(n, k)) if n else 0.,
            'mrr': 1 / (int(first[0]) + 1) if len(first) else 0.,
            'hit_rate': float(hits > 0), 'n_relevant': n, 'hits': hits,
            'fill_rate': min(len(ranked), k) / k}


def score_metrics(y, score):
    y, score = np.asarray(y), np.asarray(score, dtype=float)
    if y.ndim != 1 or score.shape != y.shape or not len(y) or not np.isin(y, [0, 1]).all() or not np.isfinite(score).all():
        raise ValueError('Finite, aligned binary outcomes and scores required')
    both = len(np.unique(y)) == 2
    precision, recall, _ = precision_recall_curve(y, score) if both else (None, None, None)
    return {'n': len(y), 'prevalence': float(y.mean()),
            'roc_auc': float(roc_auc_score(y, score)) if both else None,
            'pr_auc': float(average_precision_score(y, score)) if both else None,
            'pr_auc_trapezoid': float(auc(recall, precision)) if both else None,
            'pr_auc_definition': 'average_precision; trapezoidal area reported separately'}


def probability_metrics(y, probability, bins=10):
    y, p = np.asarray(y), np.asarray(probability, dtype=float)
    result = score_metrics(y, p)
    if (p < 0).any() or (p > 1).any() or bins < 1:
        raise ValueError('Probabilities must lie in [0,1]')
    tn, fp, fn, tp = confusion_matrix(y, p >= .5, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.
    recall = tp / (tp + fn) if tp + fn else 0.
    specificity = tn / (tn + fp) if tn + fp else None
    reliability = []
    bucket = np.minimum((p * bins).astype(int), bins - 1)
    for index in range(bins):
        mask = bucket == index
        if mask.any():
            predicted, observed = float(p[mask].mean()), float(y[mask].mean())
            reliability.append({'bin': index, 'n': int(mask.sum()), 'predicted': predicted,
                                'observed': observed, 'absolute_error': abs(predicted - observed)})
    slope = intercept = None
    if len(np.unique(y)) == 2 and np.std(p) > 1e-12:
        # Post-hoc diagnostic only; this fit never changes published predictions.
        diagnostic = LogisticRegression(C=1e6, max_iter=1000).fit(logit(np.clip(p, 1e-6, 1 - 1e-6))[:, None], y)
        slope, intercept = float(diagnostic.coef_[0, 0]), float(diagnostic.intercept_[0])
    result.update(brier=float(np.mean((p - y) ** 2)),
                  log_loss=float(log_loss(y, np.clip(p, 1e-15, 1 - 1e-15), labels=[0, 1])),
                  precision=float(precision), recall=float(recall),
                  f1=float(2 * precision * recall / (precision + recall)) if precision + recall else 0.,
                  specificity=float(specificity) if specificity is not None else None,
                  balanced_accuracy=float((recall + specificity) / 2) if specificity is not None and tp + fn else None,
                  accuracy=float((tp + tn) / len(y)), tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn), threshold=.5,
                  ece=sum(b['n'] * b['absolute_error'] for b in reliability) / len(y),
                  mce=max(b['absolute_error'] for b in reliability),
                  calibration_slope=slope, calibration_intercept=intercept)
    return result, reliability


def outcome_metrics(failure_target, failure_probability):
    y, p = np.asarray(failure_target), np.asarray(failure_probability)
    return {'risk': probability_metrics(y, p)[0], 'success': probability_metrics(1 - y, 1 - p)[0]}


def cluster_interval(values, groups, repeats=1000, seed=42):
    values, groups = np.asarray(values, dtype=float), np.asarray(groups)
    if not len(values) or values.shape != groups.shape or not np.isfinite(values).all() or repeats < 1:
        raise ValueError('Nonempty finite clustered observations required')
    unique, inverse = np.unique(groups, return_inverse=True)
    totals = np.bincount(inverse, weights=values)
    counts = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    estimates = []
    # Bound bootstrap memory; average queries while resampling whole learners.
    for start in range(0, repeats, 64):
        draw = rng.integers(0, len(unique), size=(min(64, repeats - start), len(unique)))
        estimates.extend((totals[draw].sum(axis=1) / counts[draw].sum(axis=1)).tolist())
    return {'mean': float(values.mean()), 'low': float(np.quantile(estimates, .025)),
            'high': float(np.quantile(estimates, .975)), 'clusters': len(unique), 'repeats': repeats}


def memory_bytes(pid=None):
    """Current RSS/working set and OS lifetime peak; no optional dependency."""
    pid = os.getpid() if pid is None else pid
    if os.name == 'nt':
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in ['PeakWorkingSetSize', 'WorkingSetSize',
                'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
                'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage']]
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        psapi = ctypes.WinDLL('psapi', use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        handle = kernel.OpenProcess(0x410, False, pid)
        if not handle:
            return None, None
        try:
            counter = Counters()
            counter.cb = ctypes.sizeof(counter)
            if psapi.GetProcessMemoryInfo(handle, ctypes.byref(counter), counter.cb):
                return int(counter.WorkingSetSize), int(counter.PeakWorkingSetSize)
        finally:
            kernel.CloseHandle(handle)
    else:
        try:
            fields = {line.split(':')[0]: line.split(':')[1].strip().split()[0]
                      for line in Path(f'/proc/{pid}/status').read_text().splitlines() if ':' in line}
            return int(fields['VmRSS']) * 1024, int(fields['VmHWM']) * 1024
        except (OSError, KeyError, ValueError):
            pass
    return None, None


class ResourceMonitor:
    """Sample process RSS every 50ms; OS peak is cumulative and labeled so."""
    def __init__(self, pid=None):
        self.pid = pid
        self.peak = self.os_peak = None
        self._stop = threading.Event()

    def _sample(self):
        while not self._stop.is_set():
            current, peak = memory_bytes(self.pid)
            if current is not None:
                self.peak = max(self.peak or 0, current)
                self.os_peak = max(self.os_peak or 0, peak)
            self._stop.wait(.05)

    def __enter__(self):
        self.started = time.perf_counter()
        self.cpu = time.process_time()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self._stop.set()
        self.thread.join()
        self.result = {'wall_seconds': time.perf_counter() - self.started,
                       'cpu_seconds_current_process': time.process_time() - self.cpu,
                       'sampled_peak_rss_bytes': self.peak, 'os_lifetime_peak_rss_bytes': self.os_peak,
                       'sample_interval_seconds': .05}
