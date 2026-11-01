"""Read OULAD logs and a saved risk forecast, then create a new JSON snapshot."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import pandas as pd

from .preferences import PreferenceModel
from .recommendation_agent import RecommendationAgent
from .recommendation_signals import ENROLLMENT, finite_number

DEFAULT_PROBABILITY_COLUMN = 'ensemble_lstm_attention_lgbm_weighted_probability'


def read_forecast(path, enrollment, as_of_day, *, prediction_day, column=DEFAULT_PROBABILITY_COLUMN):
    """Select exactly one forecast without reading outcome/assessment columns.

    CSVs do not carry a cutoff column, so require the caller to declare it.
    If a day_N directory exists, additionally verify the declared cutoff.
    """
    cutoff = finite_number(as_of_day, 'as_of_day')
    source_day = finite_number(prediction_day, 'prediction_day')
    if cutoff != int(cutoff) or source_day != cutoff:
        raise ValueError('Prediction day must equal recommendation cutoff')
    path = Path(path)
    for parent in path.parents:
        match = re.fullmatch(r'day_(\d+)', parent.name)
        if match:
            if int(match.group(1)) != source_day:
                raise ValueError('Prediction directory disagrees with declared day')
            break
    # Only a model probability column is accepted, never a label or raw risk.
    if not column.endswith('_probability') or column.endswith('_raw_probability'):
        raise ValueError('Select a calibrated model probability column')
    frame = pd.read_csv(path, usecols=ENROLLMENT + [column], float_precision='round_trip')
    row = frame.loc[(frame['id_student'] == enrollment[0])
                    & (frame['code_module'] == enrollment[1])
                    & (frame['code_presentation'] == enrollment[2]), column]
    if len(row) != 1:
        raise ValueError(f'Expected exactly one forecast for enrollment; found {len(row)}')
    return finite_number(row.iloc[0], 'forecast_probability', maximum=1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True, help='Directory with raw vle.csv and studentVle.csv')
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--prediction-day', type=int, required=True, help='Cutoff used to generate saved forecasts')
    parser.add_argument('--probability-column', default=DEFAULT_PROBABILITY_COLUMN)
    parser.add_argument('--student', type=int, required=True)
    parser.add_argument('--module', required=True)
    parser.add_argument('--presentation', required=True)
    parser.add_argument('--day', type=int, required=True)
    parser.add_argument('--n', type=int, default=5)
    parser.add_argument('--half-life-days', type=float, default=14.)
    parser.add_argument('--allow-review', action='store_true', help='Allow resources seen during the previous seven days')
    parser.add_argument('--output', type=Path, required=True, help='New JSON file; existing files are never overwritten')
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f'Output already exists: {args.output}')
    enrollment = (args.student, args.module, args.presentation)
    probability = read_forecast(args.predictions, enrollment, args.day,
                                prediction_day=args.prediction_day, column=args.probability_column)
    resources = pd.read_csv(args.data_dir / 'vle.csv')
    columns = ENROLLMENT + ['id_site', 'date', 'sum_click']
    parts = []
    # Retain individual resources while avoiding a full raw-log load in RAM.
    for chunk in pd.read_csv(args.data_dir / 'studentVle.csv', usecols=columns, chunksize=500_000):
        selected = chunk.loc[(chunk['id_student'] == args.student)
                              & (chunk['code_module'] == args.module)
                              & (chunk['code_presentation'] == args.presentation)
                              & (chunk['date'] < args.day), columns]
        if len(selected):
            parts.append(selected)
    events = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)
    agent = RecommendationAgent(resources, events, PreferenceModel(half_life_days=args.half_life_days))
    result = agent.recommend(enrollment, args.day, probability, args.n, allow_review=args.allow_review)
    with args.predictions.open('rb') as source:
        source_hash = hashlib.file_digest(source, 'sha256').hexdigest()
    result['forecast_source'] = {'path': str(args.predictions.resolve()), 'sha256': source_hash,
                                 'column': args.probability_column, 'prediction_day': args.prediction_day}
    result['interaction_source'] = str((args.data_dir / 'studentVle.csv').resolve())
    result['catalog_source'] = str((args.data_dir / 'vle.csv').resolve())
    payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as output:
        output.write(payload + '\n')
    print(f'Created {args.output}: {len(result["recommendations"])} recommendations; forecast unchanged ({probability:.6f}).')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
