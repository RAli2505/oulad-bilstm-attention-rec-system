"""Causal behavioral, course-context and deadline features, without grade values."""
import numpy as np
import pandas as pd

from .features import CAT, NUM, KEYS, FoldTransform, build_window

EXTRA_CAT = ['code_module', 'presentation_season']
EXTRA_NUM = ['course_days', 'course_progress', 'days_to_course_end', 'active_days', 'active_fraction',
             'days_since_first_activity', 'days_since_last_activity', 'longest_inactive_streak',
             'active_week_fraction', 'weekly_log_trend', 'weekly_variability', 'daily_activity_entropy',
             'activity_type_entropy', 'log_precourse_clicks', 'precourse_active_days',
             'n_due', 'overdue_count', 'due_completion_fraction', 'late_submissions', 'mean_submission_delay',
             'max_overdue_days', 'missing_due_weight', 'days_to_next_deadline'] + [
                 f'{stat}_{period}' for period in (7, 14, 28) for stat in ('log_recent', 'log_previous', 'log_change')]


def entropy(counts):
    p = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    return -(p * np.log(np.maximum(p, 1e-12))).sum(axis=1)


def enhanced_window(target, daily, assess, days, score_lag=7, score_policy='omit'):
    w = build_window(target, daily, assess, days, score_lag, score_policy)
    f, x = w.frame, w.counts
    totals = x.sum(axis=2)
    active = totals > 0
    any_activity = active.any(axis=1)
    f['presentation_season'] = f.code_presentation.astype(str).str[-1]
    f['course_days'] = f.module_presentation_length
    f['course_progress'] = days / f.module_presentation_length
    f['days_to_course_end'] = f.module_presentation_length - days
    f['active_days'], f['active_fraction'] = active.sum(axis=1), active.mean(axis=1)
    first = np.where(any_activity, active.argmax(axis=1), days)
    last = np.where(any_activity, days - 1 - active[:, ::-1].argmax(axis=1), -1)
    f['days_since_first_activity'] = np.where(any_activity, days - first, days)
    f['days_since_last_activity'] = np.where(any_activity, days - last, days)
    streak = np.zeros(len(f))
    longest = np.zeros(len(f))
    for day in range(days):
        streak = np.where(active[:, day], 0, streak + 1)
        longest = np.maximum(longest, streak)
    f['longest_inactive_streak'] = longest
    weekly = np.pad(totals, ((0, 0), (0, (-days) % 7))).reshape(len(f), -1, 7).sum(axis=2)
    f['active_week_fraction'] = (weekly > 0).mean(axis=1)
    position = np.arange(weekly.shape[1], dtype=float)
    centered = position - position.mean()
    f['weekly_log_trend'] = (np.log1p(weekly) * centered).sum(axis=1) / max(float((centered ** 2).sum()), 1)
    f['weekly_variability'] = weekly.std(axis=1) / np.maximum(weekly.mean(axis=1), 1)
    f['daily_activity_entropy'], f['activity_type_entropy'] = entropy(totals), entropy(x.sum(axis=1))
    for period in (7, 14, 28):
        recent = totals[:, max(0, days - period):].sum(axis=1)
        previous = totals[:, max(0, days - 2 * period):max(0, days - period)].sum(axis=1)
        f[f'log_recent_{period}'] = np.log1p(recent)
        f[f'log_previous_{period}'] = np.log1p(previous)
        f[f'log_change_{period}'] = np.log1p(recent) - np.log1p(previous)
    pre = daily[(daily.date < 0) & (daily.date >= -90)].merge(f[KEYS + ['date_registration']], on=KEYS)
    pre = pre[pre.date >= pre.date_registration]
    pre_agg = pre.groupby(KEYS, observed=True).agg(
        precourse_clicks=('sum_click', 'sum'), precourse_active_days=('date', 'nunique')).reset_index()
    f = f.merge(pre_agg, on=KEYS, how='left', validate='one_to_one')
    f['log_precourse_clicks'] = np.log1p(f.precourse_clicks.fillna(0))
    f['precourse_active_days'] = f.precourse_active_days.fillna(0)
    catalog = pd.DataFrame(assess.attrs.get('assessment_catalog', []))
    if catalog.empty:
        catalog = assess[['id_assessment', 'code_module', 'code_presentation', 'date']].drop_duplicates()
    catalog = catalog.dropna(subset=['date']).copy()
    if 'weight' not in catalog:
        catalog['weight'] = 1.
    calendar = f[KEYS + ['row_id']].merge(catalog[['id_assessment', 'code_module', 'code_presentation', 'date', 'weight']],
                                          on=['code_module', 'code_presentation'], how='left')
    submissions = assess[assess.date_submitted < days].sort_values('date_submitted').drop_duplicates(KEYS + ['id_assessment'])
    calendar = calendar.merge(submissions[KEYS + ['id_assessment', 'date_submitted']],
                              on=KEYS + ['id_assessment'], how='left', validate='many_to_one')
    due = calendar[calendar.date < days].copy()
    due['missing'] = due.date_submitted.isna()
    due['late'] = due.date_submitted.notna() & (due.date_submitted > due.date)
    due['delay'] = due.date_submitted - due.date
    due['overdue_days'] = np.where(due['missing'], days - due.date, 0)
    due['missing_weight'] = due['missing'] * due.weight.fillna(0)
    agg = due.groupby('row_id').agg(n_due=('id_assessment', 'size'), overdue_count=('missing', 'sum'),
                                   late_submissions=('late', 'sum'), mean_submission_delay=('delay', 'mean'),
                                   max_overdue_days=('overdue_days', 'max'), missing_due_weight=('missing_weight', 'sum'))
    for col in ['n_due', 'overdue_count', 'late_submissions', 'mean_submission_delay', 'max_overdue_days', 'missing_due_weight']:
        f[col] = f.row_id.map(agg[col]).fillna(0)
    f['due_completion_fraction'] = (f.n_due - f.overdue_count) / np.maximum(f.n_due, 1)
    upcoming = calendar[calendar.date >= days].groupby('row_id').date.min()
    f['days_to_next_deadline'] = (f.row_id.map(upcoming) - days).fillna(f.days_to_course_end)
    w.frame = f
    w.audit['feature_profile'] = 'behavior_deadlines_context_v2'
    w.audit['schedule_assumption'] = 'Published course deadlines and weights are known at prediction time'
    return w


class EnhancedTransform(FoldTransform):
    def __init__(self):
        self.cat_cols = CAT + EXTRA_CAT
        self.num_cols = NUM + EXTRA_NUM
