"""Reproducible learner-disjoint splits used by all comparison models."""
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def folds(y, groups, n_splits=5, seed=42):
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train, test in splitter.split(np.zeros(len(y)), y, groups):
        assert not set(groups[train]) & set(groups[test])
        yield train, test


def outer_split(y, groups, seed=42):
    development, test = next(folds(y, groups, seed=seed))
    train_rel, val_rel = next(folds(y[development], groups[development], seed=seed + 1))
    return development, test, development[train_rel], development[val_rel]
