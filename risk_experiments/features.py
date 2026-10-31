from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data.oulad_loader import OULADLoader

KEYS = ['id_student', 'code_module', 'code_presentation']
CAT = ['gender', 'region', 'highest_education', 'imd_band', 'age_band', 'disability']
NUM = ['num_of_prev_attempts', 'studied_credits', 'date_registration',
       'submitted_any', 'banked_any', 'n_submitted', 'mean_score', 'score_observed']


def fingerprint(raw: Path) -> dict:
    result = {}
    for p in sorted(raw.glob('*.csv')):
        h = hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda: f.read(1024 * 1024), b''):
                h.update(block)
        result[p.name] = h.hexdigest()
    return result


def load_data(raw: Path, cache: Path, hashes: dict):
    """Cache unfiltered daily counts; never reuse the legacy active-only cohort."""
    cache.mkdir(parents=True, exist_ok=True)
    signature = {'inputs': hashes, 'builder_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 'loader_sha256': hashlib.sha256((Path(__file__).resolve().parents[1] / 'data/oulad_loader.py').read_bytes()).hexdigest()}
    marker = cache / 'fingerprint.json'
    if (marker.exists() and json.loads(marker.read_text()) == signature
            and (cache / 'target.parquet').exists() and (cache / 'daily.parquet').exists()):
        target = pd.read_parquet(cache / 'target.parquet')
        daily = pd.read_parquet(cache / 'daily.parquet')
    else:
        loader = OULADLoader(raw)
        target, daily = loader.load_target(), loader.load_daily_vle()
        target.to_parquet(cache / 'target.parquet', index=False)
        daily.to_parquet(cache / 'daily.parquet', index=False)
        marker.write_text(json.dumps(signature, indent=2), encoding='utf-8')
    catalog = pd.read_csv(raw / 'assessments.csv')
    assess = pd.read_csv(raw / 'studentAssessment.csv').merge(
        catalog,
        on='id_assessment', how='left', validate='many_to_one')
    assess.attrs['assessment_catalog'] = catalog.to_dict('records')
    return target, daily, assess


@dataclass
class Window:
    frame: pd.DataFrame
    counts: np.ndarray
    channels: list[str]
    days: int
    audit: dict

    @property
    def y(self):
        return self.frame['final_result'].isin(['Fail', 'Withdrawn']).to_numpy(dtype=np.int64)

    @property
    def event(self):
        return (self.frame['final_result'] == 'Withdrawn').to_numpy(dtype=np.float32)

    @property
    def time(self):
        f = self.frame
        return (np.where(self.event, f.date_unregistration, f.module_presentation_length) - self.days).astype('float32')

    @property
    def groups(self):
        return self.frame.id_student.to_numpy()


def build_window(target, daily, assess, days: int, score_lag: int = 7, score_policy: str = 'omit') -> Window:
    """Predict at start of day D, using events strictly before D."""
    if days < 1 or score_lag < 0 or score_policy not in ('omit', 'proxy'):
        raise ValueError('Positive observation window and nonnegative score lag required')
    f = target.copy()
    valid = f.final_result.isin(['Pass', 'Distinction', 'Fail', 'Withdrawn'])
    known_withdrawal = (f.final_result != 'Withdrawn') | f.date_unregistration.notna()
    active = (f.date_unregistration.isna() | (f.date_unregistration > days))
    registered = f.date_registration.notna() & (f.date_registration < days)
    course_open = f.module_presentation_length > days
    eligible = valid & known_withdrawal & active & registered & course_open
    audit = {'input_enrollments': len(f), 'eligible': int(eligible.sum()),
             'unknown_withdrawal_time': int((~known_withdrawal).sum()),
             'not_active_at_landmark': int((~active).sum()),
             'not_registered_at_landmark': int((~registered).sum()),
             'course_ended': int((~course_open).sum()), 'days': days, 'score_lag': score_lag, 'score_policy': score_policy}
    f = f.loc[eligible].reset_index(drop=True)
    f['row_id'] = np.arange(len(f))
    assert not f.duplicated(KEYS).any()
    a = assess.merge(f[KEYS], on=KEYS, how='inner', validate='many_to_one')
    a = a[a.date_submitted < days].copy()
    if len(a):
        # Submission is observable. Score release is not recorded by OULAD;
        # max(due, submission)+lag is an explicit conservative availability proxy.
        a['score_available_proxy'] = np.maximum(a.date_submitted, a['date'].fillna(a.date_submitted)) + score_lag
        a['visible_score'] = a.score.where(a.score_available_proxy < days)
        grouped = a.groupby(KEYS, observed=True).agg(
            n_submitted=('id_assessment', 'size'), banked_any=('is_banked', 'max'),
            mean_score=('visible_score', 'mean'), score_observed=('visible_score', 'count')).reset_index()
        f = f.merge(grouped, on=KEYS, how='left', validate='one_to_one')
    else:
        for col in ['n_submitted', 'banked_any', 'mean_score', 'score_observed']:
            f[col] = np.nan
    f[['n_submitted', 'banked_any', 'score_observed']] = f[['n_submitted', 'banked_any', 'score_observed']].fillna(0)
    f['submitted_any'] = (f.n_submitted > 0).astype(float)
    f['score_observed'] = (f.score_observed > 0).astype(float)
    if score_policy == 'omit':
        # OULAD has submission timestamps, but no score-release timestamps.
        # Primary analysis uses observable submission counts and no grade values.
        f['mean_score'] = np.nan
        f['score_observed'] = 0.
    d = daily[(daily.date >= 0) & (daily.date < days)].merge(f[KEYS + ['row_id']], on=KEYS, how='inner')
    channels = sorted(d.activity_type.dropna().astype(str).unique())
    x = np.zeros((len(f), days, len(channels)), dtype='float32')
    mapping = {c: i for i, c in enumerate(channels)}
    if len(d):
        np.add.at(x, (d.row_id.to_numpy(), d.date.to_numpy(dtype=int), d.activity_type.astype(str).map(mapping).to_numpy()),
                  d.sum_click.to_numpy(dtype='float32'))
    audit['zero_history_enrollments'] = int((x.sum(axis=(1, 2)) == 0).sum())
    return Window(f, x, channels, days, audit)


class FoldTransform:
    def fit(self, window: Window, indices):
        self.cat_cols = getattr(self, 'cat_cols', CAT)
        self.num_cols = getattr(self, 'num_cols', NUM)
        self.channels = np.flatnonzero(window.counts[indices].sum(axis=(0, 1)) > 0)
        if not len(self.channels):
            raise ValueError('Training fold contains no observable activity types')
        self.channel_names = [window.channels[i] for i in self.channels]
        self.static = ColumnTransformer([
            ('cat', make_pipeline(SimpleImputer(strategy='most_frequent', keep_empty_features=True),
                                  OneHotEncoder(handle_unknown='ignore', sparse_output=False)), self.cat_cols),
            ('num', make_pipeline(SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True),
                                  StandardScaler()), self.num_cols)], sparse_threshold=0)
        self.static.fit(window.frame.iloc[indices][self.cat_cols + self.num_cols])
        x = np.log1p(window.counts[indices][:, :, self.channels])
        self.mean = x.mean(axis=(0, 1), keepdims=True)
        self.std = np.maximum(x.std(axis=(0, 1), keepdims=True), 1e-5)
        self.flat_scaler = StandardScaler().fit(self._flat(window, indices))
        self.week_scaler = StandardScaler().fit(self._weekly(window, indices))
        return self

    def _weekly(self, window, indices):
        x = window.counts[indices][:, :, self.channels].sum(axis=2)
        pad = (-window.days) % 7
        return np.log1p(np.pad(x, ((0, 0), (0, pad))).reshape(len(indices), -1, 7).sum(axis=2))

    def _flat(self, window, indices):
        x = window.counts[indices][:, :, self.channels]
        static = self.static.transform(window.frame.iloc[indices][self.cat_cols + self.num_cols])
        last = x[:, -7:].sum(axis=1)
        prev = x[:, -14:-7].sum(axis=1) if window.days >= 14 else np.zeros_like(last)
        chunks = [np.log1p(x.sum(axis=1)), np.log1p(x.mean(axis=1)), np.log1p(x.std(axis=1)),
                  np.log1p(x.max(axis=1)), np.log1p(last), np.log1p(prev), np.log1p(last) - np.log1p(prev)]
        return np.concatenate([static, *chunks], axis=1).astype('float32')

    def transform(self, window, indices):
        indices = np.asarray(indices)
        return {'x': ((np.log1p(window.counts[indices][:, :, self.channels]) - self.mean) / self.std).astype('float32'),
                's': self.static.transform(window.frame.iloc[indices][self.cat_cols + self.num_cols]).astype('float32'),
                'flat': self.flat_scaler.transform(self._flat(window, indices)).astype('float32'),
                'weekly': self.week_scaler.transform(self._weekly(window, indices)).astype('float32'),
                'y': window.y[indices], 'time': window.time[indices], 'event': window.event[indices]}


def temporal_split(window):
    f = window.frame
    dev = np.flatnonzero(f.code_presentation.isin(['2013B', '2013J']))
    candidates = np.flatnonzero(f.code_presentation.isin(['2014B', '2014J']))
    test = candidates[~np.isin(window.groups[candidates], window.groups[dev])]
    assert not set(window.groups[dev]) & set(window.groups[test])
    return dev, test, len(candidates) - len(test)
