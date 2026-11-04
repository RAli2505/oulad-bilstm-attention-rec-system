"""Observational association of repeat access with subsequent OULAD assessments."""
import argparse
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from run_oulad_agent_experiment import _Tee

KEY = ['id_student', 'code_module', 'code_presentation']


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def adjusted(frame, outcome):
    # Fixed assessment effects plus pre-exposure activity; student-cluster sandwich.
    controls = ['log_clicks', 'log_sites', 'log_days', 'log_recent_clicks', 'log_prior_submissions', 'tenure']
    fixed = pd.get_dummies(frame.assessment_group, drop_first=True, dtype=float).to_numpy()
    x = np.column_stack([np.ones(len(frame)), frame.repeat_access.to_numpy(), frame[controls].to_numpy(), fixed])
    others = np.delete(x, 1, axis=1)
    exposure_variance = float(np.mean((x[:, 1] - others @ np.linalg.lstsq(others, x[:, 1], rcond=None)[0]) ** 2))
    if exposure_variance < 1e-10 or len(frame) <= np.linalg.matrix_rank(x) + 1:
        raise ValueError('Insufficient exposure overlap or degrees of freedom for adjusted association')
    y = frame[outcome].to_numpy(dtype=float)
    inverse = np.linalg.pinv(x.T @ x)
    beta = inverse @ x.T @ y
    residual = y - x @ beta
    _, groups = np.unique(frame.id_student, return_inverse=True)
    n_groups = int(groups.max()) + 1
    scores = np.zeros((n_groups, x.shape[1])); np.add.at(scores, groups, x * residual[:, None])
    correction = n_groups / max(n_groups - 1, 1) * (len(frame) - 1) / max(len(frame) - np.linalg.matrix_rank(x), 1)
    variance = inverse @ (scores.T @ scores) @ inverse * correction
    se = float(np.sqrt(max(variance[1, 1], 0)))
    return {'outcome': outcome, 'adjusted_repeat_association': float(beta[1]),
            'ci_low': float(beta[1] - 1.96 * se), 'ci_high': float(beta[1] + 1.96 * se),
            'n': len(frame), 'student_clusters': n_groups,
            'exposure_residual_variance': exposure_variance}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--windows', type=int, nargs='+', default=[7, 14, 21, 28])
    parser.add_argument('--raw', type=Path, default=Path(__file__).parent / 'data/oulad/raw')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--threads', type=int, default=6)
    args = parser.parse_args()
    if min(args.windows) < 1 or args.threads < 1:
        parser.error('Positive windows and threads required')
    output = args.resume or args.output or Path(__file__).parent / 'runs' / ('repeat_outcomes_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    output = output.resolve(); raw = args.raw.resolve()
    if output == raw or output.is_relative_to(raw):
        parser.error('Output must be separate from raw data')
    config = {'windows': args.windows, 'exposure_days': 7, 'max_outcome_days_after_landmark': 35,
              'raw': str(raw), 'files': {n: digest(raw / n) for n in ['studentRegistration.csv', 'studentVle.csv', 'assessments.csv', 'studentAssessment.csv']},
              'script_sha256': digest(Path(__file__)), 'causal_effect': False, 'test_2014_used': False}
    if args.resume and json.loads((output / 'config.json').read_text()) != config:
        raise ValueError('Resume data/configuration mismatch')
    output.mkdir(parents=True, exist_ok=bool(args.resume))
    (output / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    with (output / f'log_{time.time_ns()}.txt').open('x', encoding='utf-8') as logfile, redirect_stdout(_Tee(sys.stdout, logfile)), redirect_stderr(_Tee(sys.stderr, logfile)), threadpool_limits(limits=args.threads):
        print('Loading 2013 registrations and assessments', flush=True)
        reg = pd.read_csv(raw / 'studentRegistration.csv'); reg = reg.loc[reg.code_presentation.isin(['2013B', '2013J'])]
        assessments = pd.read_csv(raw / 'assessments.csv')
        assessments = assessments.loc[assessments.code_presentation.isin(['2013B', '2013J']) & assessments.assessment_type.isin(['TMA', 'CMA'])]
        submissions = pd.read_csv(raw / 'studentAssessment.csv')
        submissions = submissions.loc[submissions.is_banked == 0].copy()
        submissions = submissions.merge(assessments[['id_assessment', 'code_module', 'code_presentation']], on='id_assessment', how='inner', validate='many_to_one')
        parts = []
        for i, chunk in enumerate(pd.read_csv(raw / 'studentVle.csv', chunksize=500000), 1):
            parts.append(chunk.loc[chunk.code_presentation.isin(['2013B', '2013J']) & (chunk.date < max(args.windows) + 7) & (chunk.sum_click > 0)])
            print(f'Loading activity chunk {i}', flush=True)
        events = pd.concat(parts, ignore_index=True)
        reports = []
        for day in args.windows:
            checkpoint = output / f'day_{day}.json'
            if args.resume and checkpoint.exists():
                saved = json.loads(checkpoint.read_text())
                if saved['cohort_sha256'] != digest(output / f'day_{day}_cohort.csv'):
                    raise ValueError('Checkpoint cohort checksum mismatch')
                reports.extend(saved['results']); print(f'Day {day}: checkpoint reused', flush=True); continue
            print(f'Day {day}: exposure and subsequent assessment cohorts', flush=True)
            end = day + 7
            next_assessment = assessments.loc[(assessments.date >= end) & (assessments.date <= day + 35)].sort_values(['date', 'id_assessment']).drop_duplicates(['code_module', 'code_presentation'])
            cohort = reg.loc[(reg.date_registration < day) & (reg.date_unregistration.isna() | (reg.date_unregistration >= day))].merge(next_assessment, on=['code_module', 'code_presentation'], validate='many_to_one')
            past = events.loc[events.date < day]
            future = events.loc[(events.date >= day) & (events.date < end)]
            seen = past[KEY + ['id_site']].drop_duplicates()
            repeated = future.merge(seen, on=KEY + ['id_site'], how='inner', validate='many_to_one')[KEY].drop_duplicates().assign(repeat_access=1)
            history = past.groupby(KEY).agg(clicks=('sum_click', 'sum'), sites=('id_site', 'nunique'), days=('date', 'nunique')).reset_index()
            recent = past.loc[past.date >= day - 7].groupby(KEY).sum_click.sum().rename('recent_clicks').reset_index()
            prior = submissions.loc[submissions.date_submitted < day].groupby(KEY).size().rename('prior_submissions').reset_index()
            for values in [repeated, history, recent, prior]:
                cohort = cohort.merge(values, on=KEY, how='left', validate='many_to_one')
            for column in ['repeat_access', 'clicks', 'sites', 'days', 'recent_clicks', 'prior_submissions']:
                cohort[column] = cohort[column].fillna(0)
            for column in ['clicks', 'sites', 'days', 'recent_clicks', 'prior_submissions']:
                cohort['log_' + column] = np.log1p(cohort[column])
            cohort['tenure'] = day - cohort.date_registration
            # Fixed deadline selects outcomes without looking at scores or submission.
            actual = submissions[['id_student', 'id_assessment', 'date_submitted', 'score']]
            if actual.duplicated(['id_student', 'id_assessment']).any():
                raise ValueError('Duplicate student assessments require an explicit policy')
            cohort = cohort.merge(actual, on=['id_student', 'id_assessment'], how='left', validate='many_to_one')
            cohort['submitted_by_deadline'] = (cohort.date_submitted.notna() & (cohort.date_submitted <= cohort.date)).astype(int)
            cohort['score_by_deadline_zero_if_missing'] = cohort.score.where(cohort.submitted_by_deadline.eq(1), 0).fillna(0)
            # An assessment already submitted before exposure ends is not a subsequent outcome.
            cohort = cohort.loc[cohort.date_submitted.isna() | (cohort.date_submitted >= end)].copy()
            cohort['assessment_group'] = cohort.id_assessment.astype(str)
            if cohort.repeat_access.nunique() != 2:
                raise ValueError('Both exposure groups required')
            table = output / f'day_{day}_cohort.csv'; cohort.to_csv(table, index=False)
            cohort.groupby('repeat_access')[['submitted_by_deadline', 'score_by_deadline_zero_if_missing']].agg(['mean', 'count']).to_csv(output / f'day_{day}_unadjusted.csv')
            result = [{'day': day, **adjusted(cohort, outcome)} for outcome in ['submitted_by_deadline', 'score_by_deadline_zero_if_missing']]
            temporary = checkpoint.with_suffix('.tmp'); temporary.write_text(json.dumps({'results': result, 'cohort_sha256': digest(table)}, indent=2)); temporary.replace(checkpoint)
            reports.extend(result); pd.DataFrame(reports).to_csv(output / 'adjusted_associations.csv', index=False)
            print(f'Day {day}: checkpoint saved; n={len(cohort)}', flush=True)
        pd.DataFrame(reports).to_csv(output / 'adjusted_associations.csv', index=False)
        (output / 'LIMITATIONS_RU.txt').write_text('Наблюдательная связь, не причинный эффект. Повторы — фактические обращения, показы рекомендаций отсутствуют. Возможны смешение мотивацией и выборочное раннее выполнение заданий. Исключение уже сдавших до конца окна экспозиции ограничивает целевую популяцию. Оценки до окна не использованы как признаки: даты публикации оценок неизвестны. Балл с нулём при отсутствии своевременной сдачи — составной исход, а не оценка усвоения. Сопоставление по тематике VLE и задания не установлено. Интервалы — асимптотические, с кластеризацией по студенту.\n', encoding='utf-8')
        print(f'Finished observational analysis: {output}', flush=True)


if __name__ == '__main__':
    main()
