"""Retrain forecasting and evaluate adaptive recommendations, without 2014 test."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

WINDOWS = [7, 14, 21, 28, 56, 84, 112, 140, 168]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--windows', type=int, nargs='+', default=WINDOWS.copy())
    parser.add_argument('--seeds', type=int, nargs='+', default=[42])
    parser.add_argument('--threads', type=int, default=min(6, os.cpu_count() or 1))
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--tuning-trials', type=int, default=4)
    parser.add_argument('--screen-epochs', type=int, default=15)
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.add_argument('--permutations', type=int, default=1000)
    parser.add_argument('--horizon', type=int, default=7)
    parser.add_argument('--ks', type=int, nargs='+', default=[5, 10, 20])
    parser.add_argument('--feature-engine', choices=['batched', 'reference'], default='batched')
    parser.add_argument('--feature-batch-size', type=int, default=256, help='Memory batch, not a sample; all queries/candidates are retained')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--risk-run', type=Path, help='Skip forecast training and read this completed source run')
    parser.add_argument('--resume-risk', type=Path, help='Resume forecast training with matching original settings')
    parser.add_argument('--max-learners', type=int, help='Pilot only: maximum learners per recommendation split')
    parser.add_argument('--check', action='store_true', help='Verify dependencies, arguments and paths without training')
    return parser


def risk_command(args):
    root = Path(__file__).resolve().parent
    command = [sys.executable, '-X', 'utf8', '-u', str(root / 'run_improved_article.py'), '--improved', '--system-only',
               '--windows', *map(str, args.windows), '--seeds', *map(str, args.seeds),
               '--threads', str(args.threads), '--device', args.device, '--epochs', str(args.epochs),
               '--patience', str(args.patience), '--batch-size', str(args.batch_size),
               '--tuning-trials', str(args.tuning_trials), '--screen-epochs', str(args.screen_epochs),
               '--bootstrap', str(args.bootstrap), '--permutations', str(args.permutations),
               '--score-policy', 'omit', '--primary-day', '56']
    if args.resume_risk:
        command.extend(['--resume', str(args.resume_risk.resolve())])
    if args.check:
        command.append('--check')
    return command


def check_forecast_exit(code, risk_run, args):
    """Recover only known native shutdown failure with fully verified outputs.

    Exit 0xC0000409 is never treated as success on its own. A completed run,
    matching configuration, every window/job, saved hashes and finite aligned
    forecasts are required. Cancellation and other exit codes remain failures.
    """
    if code == 0:
        return {'exit_code': 0, 'recovered_after_verified_completion': False}
    if code & 0xFFFFFFFF != 0xC0000409:
        raise RuntimeError(f'Forecast stage failed with exit {code}; see forecast_stage.log')
    from recommendations.experiment import sha256, source_splits
    from run_risk_article import system_model_names
    import numpy as np
    import pandas as pd
    root = Path(risk_run).resolve()
    manifest = json.loads((root / 'run_manifest.json').read_text(encoding='utf-8'))
    if manifest.get('status') != 'completed':
        raise RuntimeError('Native exit: forecast manifest is not completed; refusing recovery')
    config = manifest['config']
    expected = {name: getattr(args, name) for name in ['windows', 'seeds', 'threads', 'epochs', 'patience',
                'batch_size', 'tuning_trials', 'screen_epochs', 'bootstrap', 'permutations']}
    expected.update(models=system_model_names(), system_only=True, evaluate_test=False,
                    split_scheme='holdout', synthetic=False, precision='FP32', deterministic=True,
                    score_policy='omit', improved=True)
    if args.device != 'auto':
        expected['device'] = args.device
    if any(config.get(name) != value for name, value in expected.items()):
        raise RuntimeError('Native exit: forecast configuration differs from requested run; refusing recovery')
    signature = manifest['signature']
    verified_jobs, verified_files, verified_bytes = 0, 0, 0
    for day in args.windows:
        day_dir = root / f'day_{day}'
        status = json.loads((day_dir / 'status.json').read_text(encoding='utf-8'))
        if status.get('status') != 'completed' or status.get('signature') != signature:
            raise RuntimeError(f'Native exit: day {day} is not completed')
        frame, indices = source_splits(day_dir)
        published = pd.read_csv(day_dir / 'validation_2013_predictions.csv')
        keys = ['id_student', 'code_module', 'code_presentation', 'target']
        if not published[keys].reset_index(drop=True).equals(frame.iloc[indices['held']][keys].reset_index(drop=True)):
            raise RuntimeError(f'Native exit: day {day} validation identity mismatch')
        summary = pd.read_csv(day_dir / 'summary.csv')
        if len(summary) != len(config['models']) or set(summary.model) != set(config['models']):
            raise RuntimeError(f'Native exit: day {day} summary is incomplete')
        for seed in args.seeds:
            for model in config['models']:
                job = day_dir / f'fold_1/seed_{seed}' / model
                record = json.loads((job / 'status.json').read_text(encoding='utf-8'))
                hashes = record.get('artifacts_sha256', {})
                required = {'predictions.npz', 'training.json', 'calibration.joblib'}
                if record.get('status') != 'completed' or record.get('signature') != signature or not required.issubset(hashes):
                    raise RuntimeError(f'Native exit: incomplete checkpoint {job}')
                for relative, digest in hashes.items():
                    artifact = (job / relative).resolve()
                    if not artifact.is_relative_to(job.resolve()) or not artifact.is_file() or sha256(artifact) != digest:
                        raise RuntimeError(f'Native exit: checkpoint checksum mismatch: {artifact}')
                    verified_files += 1
                    verified_bytes += artifact.stat().st_size
                with np.load(job / 'predictions.npz', allow_pickle=False) as saved:
                    fields = {'held_probability': 'held', 'held_score': 'held', 'val_score': 'selection', 'calibration_score': 'calibration'}
                    if model != 'ensemble_lstm_attention_lgbm_weighted':
                        fields['fit_score'] = 'fit'
                    for name, role in fields.items():
                        values = saved[name]
                        if values.shape != (len(indices[role]),) or not np.isfinite(values).all():
                            raise RuntimeError(f'Native exit: invalid saved predictions: {job}/{name}')
                    p = saved['held_probability']
                    if ((p < 0) | (p > 1)).any():
                        raise RuntimeError(f'Native exit: invalid probability: {job}')
                verified_jobs += 1
    return {'exit_code': code, 'exit_hex': '0xC0000409', 'recovered_after_verified_completion': True,
            'verified_model_jobs': verified_jobs, 'verified_artifacts': verified_files,
            'verified_bytes': verified_bytes, 'risk_run': str(root),
            'warning': 'Native process failed after completed forecast output; all saved checkpoints verified. Native runtime cause remains unresolved.'}


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.windows != WINDOWS:
        parser.error('This protocol requires exactly: 7 14 21 28 56 84 112 140 168')
    if args.risk_run and args.resume_risk:
        parser.error('Choose --risk-run or --resume-risk')
    if len(set(args.seeds)) != len(args.seeds) or not args.seeds or min(args.ks) < 1 or 10 not in args.ks:
        parser.error('Use unique seeds and positive K including 10')
    if min(args.threads, args.epochs, args.patience, args.batch_size, args.tuning_trials,
           args.screen_epochs, args.bootstrap, args.permutations, args.horizon, args.feature_batch_size) < 1:
        parser.error('All numeric budgets must be positive')
    if args.max_learners is not None and args.max_learners < 1:
        parser.error('--max-learners must be positive')
    root = Path(__file__).resolve().parent
    for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        os.environ[name] = str(args.threads)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    from recommendations.experiment import run_recommendations, write_json
    from recommendations.experiment_metrics import ResourceMonitor
    if args.check:
        if args.risk_run:
            source = json.loads((args.risk_run / 'run_manifest.json').read_text(encoding='utf-8'))
            if source['config'].get('evaluate_test') or not set(args.windows).issubset(source['config']['windows']) or not set(args.seeds).issubset(source['config']['seeds']):
                parser.error('Source run does not match requested training/validation protocol')
            print('Recommendation source manifest OK; no training started.')
            return 0
        return subprocess.run(risk_command(args), cwd=root, check=False).returncode
    output = args.output.resolve() if args.output else root / 'runs' / ('system_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    output.mkdir(parents=True, exist_ok=False)
    source = output / 'source'
    shutil.copytree(root / 'recommendations', source / 'recommendations', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(__file__, source / Path(__file__).name)
    state = {**vars(args), 'output': str(output), 'status': 'running', 'external_test_used': False}
    state = json.loads(json.dumps(state, default=str))
    write_json(output / 'system_manifest.json', state)
    risk_run = args.risk_run.resolve() if args.risk_run else None
    try:
        if risk_run is None:
            with (output / 'forecast_stage.log').open('x', encoding='utf-8') as log:
                process = subprocess.Popen(risk_command(args), cwd=root, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True, encoding='utf-8', bufsize=1)
                with ResourceMonitor(process.pid) as monitor:
                    for line in process.stdout:
                        print(line, end='', flush=True)
                        log.write(line)
                        if line.startswith('Output:'):
                            risk_run = Path(line.split(':', 1)[1].strip()).resolve()
                    code = process.wait()
                write_json(output / 'forecast_process_resources.json', monitor.result)
                if risk_run is None:
                    raise RuntimeError('Forecast runner did not identify its output directory')
                audit = check_forecast_exit(code, risk_run, args)
                write_json(output / 'forecast_completion_audit.json', audit)
                state['forecast_exit_code'] = code
                if audit['recovered_after_verified_completion']:
                    state['forecast_shutdown_warning'] = audit['warning']
                    print(f'WARNING: {audit["warning"]}', flush=True)
        state['risk_run'] = str(risk_run)
        write_json(output / 'system_manifest.json', state)
        recommendations = run_recommendations(risk_run, output / 'recommendations', args.windows, args.seeds,
                          horizon=args.horizon, ks=tuple(args.ks), bootstrap=args.bootstrap,
                          threads=args.threads, max_learners=args.max_learners, methods=['hybrid_risk'],
                          feature_engine=args.feature_engine, feature_batch_size=args.feature_batch_size)
        state.update(status='completed', recommendations=str(recommendations))
    except BaseException as exc:
        state.update(status='failed', error=repr(exc))
        write_json(output / 'system_manifest.json', state)
        raise
    write_json(output / 'system_manifest.json', state)
    print(f'Finished full system: {output}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
