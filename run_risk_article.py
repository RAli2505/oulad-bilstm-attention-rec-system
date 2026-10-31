"""All 14 models, three landmarks, five learner-grouped folds; BCE+Cox is primary."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback


class Tee:
    def __init__(self, terminal, file):
        self.terminal, self.file = terminal, file

    def write(self, value):
        self.terminal.write(value)
        self.file.write(value)

    def flush(self):
        self.terminal.flush()
        self.file.flush()

    def isatty(self):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--windows', type=int, nargs='+', default=[28, 56, 112])
    parser.add_argument('--primary-day', type=int, default=56)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 13, 7])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=.001)
    parser.add_argument('--cox-weight', type=float, default=1.)
    parser.add_argument('--score-lag', type=int, default=7)
    parser.add_argument('--score-policy', choices=['omit', 'proxy'], default='omit',
                        help='Primary analysis omits scores without release timestamps; proxy is a separate sensitivity analysis')
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.add_argument('--permutations', type=int, default=1000)
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--smoke', action='store_true', help='Synthetic data, all 14 models and 5 folds, one epoch, not for publication')
    args = parser.parse_args()
    if (min(args.windows) < 1 or args.epochs < 1 or args.patience < 1 or args.batch_size < 2 or
            args.threads < 1 or args.bootstrap < 1 or args.permutations < 1 or args.score_lag < 0 or
            args.learning_rate <= 0 or args.cox_weight <= 0):
        parser.error('Invalid nonpositive parameter or negative score lag')
    if len(set(args.windows)) != len(args.windows) or len(set(args.seeds)) != len(args.seeds):
        parser.error('Windows and seeds must be unique')
    if args.primary_day not in args.windows and not args.smoke:
        parser.error('--primary-day must be included in --windows')
    root = Path(__file__).resolve().parent
    resume = args.resume.resolve() if args.resume else None
    os.chdir(root)
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = str(args.threads)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    from threadpoolctl import threadpool_limits
    from risk_experiments.features import fingerprint, load_data
    from risk_experiments.pipeline import run_window, sha256, synthetic_data, write_json
    from risk_experiments.networks import MODELS
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Math attention avoids nondeterministic fused Transformer backward kernels.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.use_deterministic_algorithms(True)
    device = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else 'cpu' if args.device == 'auto' else args.device
    if device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable; choose --device cpu')
    required = ['numpy', 'pandas', 'pyarrow', 'scipy', 'scikit-learn', 'torch', 'xgboost', 'lightgbm', 'joblib', 'matplotlib', 'threadpoolctl']
    versions = {name: importlib.metadata.version(name) for name in required}
    raw = root / 'data' / 'oulad' / 'raw'
    required_files = ['studentInfo.csv', 'studentRegistration.csv', 'courses.csv', 'assessments.csv',
                      'studentAssessment.csv', 'studentVle.csv', 'vle.csv']
    if not args.smoke:
        for name in required_files:
            if not (raw / name).is_file():
                raise FileNotFoundError(raw / name)
    config = {'windows': [28] if args.smoke else args.windows, 'primary_day': 28 if args.smoke else args.primary_day,
              'seeds': [42] if args.smoke else args.seeds, 'epochs': 1 if args.smoke else args.epochs, 'patience': args.patience,
              'batch_size': args.batch_size, 'learning_rate': args.learning_rate, 'cox_weight': args.cox_weight,
              'score_lag': args.score_lag, 'score_policy': args.score_policy, 'bootstrap': 10 if args.smoke else args.bootstrap,
              'permutations': 10 if args.smoke else args.permutations, 'threads': args.threads, 'device': device,
              'split_seed': 42, 'folds': 5, 'min_delta': 1e-4, 'precision': 'FP32', 'deterministic': True,
              'inference_batch_size': 512, 'train_probe_max_rows': 512,
              'optimizer': 'Adam', 'optimizer_fused': device == 'cuda',
              'mlp_iterations': 2 if args.smoke else 200, 'synthetic': args.smoke,
              'models': MODELS, 'protocol': 'oulad_landmark_composite_risk_bce_cox_v1'}
    print(f"Device: {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}", flush=True)
    print(f"{len(MODELS)} models; windows {config['windows']}; five folds; seeds {config['seeds']}; FP32.", flush=True)
    if args.check:
        print('Dependencies and input paths OK. No training started.', flush=True)
        return
    hashes = {'synthetic': 'seed3_n400'} if args.smoke else fingerprint(raw)
    paths = [Path(__file__), *root.glob('risk_experiments/*.py'), root / 'models/pred_oulad.py',
             root / 'models/oulad_hybrid_attention.py', root / 'data/oulad_loader.py', root / 'evaluation_metrics.py']
    sources = {str(p.relative_to(root)): sha256(p) for p in paths}
    identity = {'config': config, 'data_sha256': hashes, 'source_sha256': sources, 'versions': versions}
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    output = resume or root / 'runs' / ('risk_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    if resume:
        previous = json.loads((output / 'run_manifest.json').read_text())
        if previous['signature'] != signature:
            raise ValueError('Cannot resume: code, data, packages, device or experiment settings differ. Use the original flags.')
    else:
        output.mkdir(parents=True, exist_ok=False)
    for relative in sources:
        destination = output / 'source' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / relative, destination)
    shutil.copy2(root / 'docs/RISK_ARTICLE_PROTOCOL_RU.md', output / 'protocol.md')
    environment = {'python': sys.version, 'platform': platform.platform(), 'processor': platform.processor(),
                   'logical_cpus': os.cpu_count(), 'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
                   'gpu': str(torch.cuda.get_device_properties(0)) if device == 'cuda' else None,
                   'packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions() if d.metadata['Name']}}
    write_json(output / 'environment.json', environment)
    manifest = {**identity, 'signature': signature, 'status': 'running', 'command': sys.argv,
                'output': str(output), 'started_utc': datetime.now(timezone.utc).isoformat(),
                'primary_contrast': 'proposed vs no_cox, external 2014 test, day 56 (or selected primary day)',
                'score_availability_assumption': 'Scores omitted in primary analysis. Optional proxy=max(submission_date, due_date)+score_lag; not an observed release timestamp.',
                'cox_objective': 'minibatch partial likelihood, Breslow ties, lambda=cox_weight; event=withdrawal only',
                'selection': 'fixed hyperparameters; neural early stop on inner validation raw BCE; no test-driven model selection'}
    write_json(output / 'run_manifest.json', manifest)
    (output / 'experiment_card.md').write_text(
        '# OULAD BCE + Cox experiment\n\n'
        'Primary endpoint: Fail + Withdrawn vs Pass + Distinction. Survival event: withdrawal.\n\n'
        'Learner-disjoint 2013 development CV; unseen-learner 2014 external test. '
        'Primary contrast: proposed vs no_cox at the predeclared primary day. '
        'Numerical assessment scores are omitted unless an availability proxy was explicitly requested.\n\n'
        'Resolved settings:\n\n```json\n' + json.dumps(config, indent=2) + '\n```\n\n'
        'See run_manifest.json, environment.json, source/, day_*/cohort_audit.json, '
        'fold metrics, learning curves, fit diagnostics and calibration reports for complete details.\n', encoding='utf-8')
    terminal_out, terminal_err = sys.stdout, sys.stderr
    start = time.perf_counter()
    with (output / 'run.log').open('a', encoding='utf-8', buffering=1) as log:
        sys.stdout, sys.stderr = Tee(terminal_out, log), Tee(terminal_err, log)
        try:
            print('Output:', output, flush=True)
            target, daily, assess = synthetic_data() if args.smoke else load_data(raw, root / '.risk-cache', hashes)
            with threadpool_limits(limits=args.threads):
                for day in config['windows']:
                    run_window(target, daily, assess, {**config, 'days': day}, output / f'day_{day}', signature, args.prepare_only)
            manifest['status'] = 'prepared' if args.prepare_only else 'completed'
        except BaseException as exc:
            manifest.update(status='failed', error=repr(exc))
            traceback.print_exc()
            raise
        finally:
            manifest.update(wall_seconds=time.perf_counter() - start, finished_utc=datetime.now(timezone.utc).isoformat())
            write_json(output / 'run_manifest.json', manifest)
            sys.stdout, sys.stderr = terminal_out, terminal_err
    print('Finished:', output, flush=True)


if __name__ == '__main__':
    main()
