"""Compare reference/batched features and top-K on the same bounded real sample.

Does not train models or modify a running experiment. Timings concern feature
construction only; sampled benchmark results are not recommendation quality.
"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from recommendations.experiment import load_events, make_candidates, rank_indices, source_splits
from recommendations.recommendation_signals import ENROLLMENT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--risk-run', type=Path, required=True)
    parser.add_argument('--windows', type=int, nargs='+', default=[7, 56, 168])
    parser.add_argument('--max-learners', type=int, default=12)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if min(args.max_learners, args.repeats) < 1:
        parser.error('Positive sample and repeat counts required')
    splits = {d: source_splits(args.risk_run / f'day_{d}') for d in args.windows}
    selected = []
    for frame, indices in splits.values():
        for positions in indices.values():
            cohort = frame.iloc[positions]
            selected.append(cohort.loc[cohort.id_student.isin(sorted(cohort.id_student.unique())[:args.max_learners]), ENROLLMENT])
    raw = ROOT / 'data/oulad/raw'
    events = load_events(raw, pd.concat(selected).drop_duplicates(), max(args.windows) + 7)
    resources, courses = pd.read_csv(raw / 'vle.csv'), pd.read_csv(raw / 'courses.csv')
    reports = []
    with threadpool_limits(limits=1):
        for day in args.windows:
            frame, indices = splits[day]
            durations = {'reference': [], 'batched': []}
            values = {}
            for repeat in range(args.repeats):
                for engine in (['reference', 'batched'] if repeat % 2 == 0 else ['batched', 'reference']):
                    start = time.perf_counter()
                    values[engine] = make_candidates(resources, events, frame, indices, day, 7, courses,
                        max_learners=args.max_learners, include_no_decay=False, feature_engine=engine)
                    durations[engine].append(time.perf_counter() - start)
            old, new = values['reference'], values['batched']
            for a, b in zip(old[:3], new[:3]):
                pd.testing.assert_frame_equal(a, b, check_dtype=False, atol=1e-12, rtol=1e-12)
            for q in old[1].itertuples():
                ranks = []
                risk = (q.query % 11) / 10
                for value in [old, new]:
                    part = value[0].iloc[q.start:q.stop]
                    score = (.5 * part.preference + (.2 + .2 * risk) * part.calendar + (.3 - .2 * risk) * part.engagement).to_numpy()
                    ranks.append(part.item.to_numpy()[rank_indices(score, part.item.to_numpy(), part.kind.to_numpy(), 20, .15)].tolist())
                if ranks[0] != ranks[1]:
                    raise AssertionError(f'Top-K differs for query {q.query}')
            floats = ['preference', 'calendar', 'engagement', 'popularity', 'confidence', 'no_decay']
            difference = np.abs(old[0][floats].to_numpy() - new[0][floats].to_numpy())
            medians = {name: float(np.median(times)) for name, times in durations.items()}
            report = {'day': day, 'queries': len(new[1]), 'candidate_pairs': len(new[0]),
                      'seconds': durations, 'median_seconds': medians,
                      'feature_speedup': medians['reference'] / medians['batched'],
                      'maximum_absolute_feature_difference': float(difference.max()) if difference.size else 0.,
                      'candidate_label_query_parity': True, 'top_20_parity': True}
            reports.append(report)
            print(json.dumps(report), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as output:
        json.dump({'risk_run': str(args.risk_run.resolve()), 'max_learners_per_split': args.max_learners,
                   'threads': 1, 'scope': 'feature construction; no training; sample is not full-study quality',
                   'results': reports}, output, indent=2)


if __name__ == '__main__':
    main()
