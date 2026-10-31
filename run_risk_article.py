"""OULAD risk experiments with learner-grouped validation and a closed external test."""
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
        self.file.write(value)
        return self.terminal.write(value)

    def flush(self):
        self.terminal.flush()
        self.file.flush()

    def isatty(self):
        return False


def experiment_signature(identity):
    common_config = {k: v for k, v in identity['config'].items() if k != 'seeds'}
    return hashlib.sha256(json.dumps({**identity, 'config': common_config}, sort_keys=True).encode()).hexdigest()


def configure_utf8_output():
    """RTK's Windows console bridge expects UTF-8, including Cyrillic paths."""
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='backslashreplace')


def logging_only_resume_compatible(previous, identity, output):
    """Accept only this display-only patch; preserve numerical source guards."""
    import ast
    for key in ('data_sha256', 'versions'):
        if previous[key] != identity[key]:
            return False
    strip_seeds = lambda c: {k: v for k, v in c.items() if k != 'seeds'}
    if strip_seeds(previous['config']) != strip_seeds(identity['config']):
        return False
    if set(previous['source_sha256']) != set(identity['source_sha256']):
        return False

    def canonical(source):
        for state in ('reuse', 'training'):
            source = source.replace(
                "f'Day {window.days} train/validation seed {seed} {name}: " + state + "'",
                "f'Day {window.days} fold {fold} seed {seed} {name}: " + state + "'")
        tree = ast.parse(source)
        tree.body = [n for n in tree.body if not (isinstance(n, ast.FunctionDef) and n.name == 'logging_only_resume_compatible')]
        class MetadataHistory(ast.NodeTransformer):
            def visit_If(self, node):
                if ast.unparse(node.test) == 'destination.exists() and sha256(destination) != sources[relative]':
                    return None
                return self.generic_visit(node)
        tree = MetadataHistory().visit(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.If) and ast.unparse(node.test) == "previous['signature'] != signature":
                node.body = [ast.Pass()]
        return ast.dump(tree, include_attributes=False)

    for relative, digest in identity['source_sha256'].items():
        if digest == previous['source_sha256'][relative]:
            continue
        if relative.replace('\\', '/') not in ('run_risk_article.py', 'risk_experiments/pipeline.py'):
            return False
        saved = output / 'source' / relative
        current = Path(__file__).resolve().parent / relative
        if not saved.is_file() or not current.is_file():
            return False
        if canonical(saved.read_text(encoding='utf-8')) != canonical(current.read_text(encoding='utf-8')):
            return False
    return True


def main():
    configure_utf8_output()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--windows', type=int, nargs='+', default=[7, 14, 28, 56, 84, 112, 140, 168])
    parser.add_argument('--primary-day', type=int, default=56)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42])
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
    parser.add_argument('--improved', action='store_true', help='Rich features, inner-only tuning and four ensemble variants')
    parser.add_argument('--basic-features', action='store_true', help='Feature ablation: tune the original features with the same inner protocol')
    parser.add_argument('--tuning-trials', type=int, default=4)
    parser.add_argument('--screen-epochs', type=int, default=15)
    parser.add_argument('--smoke', action='store_true', help='Synthetic data, one epoch, selected evaluation scheme; not for publication')
    args = parser.parse_args()
    if not 1 <= args.tuning_trials <= 6 or args.screen_epochs < 1:
        parser.error('Use 1..6 tuning trials and positive screening epochs')
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
    from risk_experiments.tuning import ENSEMBLES
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
              'split_seed': 42, 'folds': 1 if args.improved else 5,
              'split_scheme': 'holdout' if args.improved else 'grouped_cv',
              'train_pool_fraction': .7 if args.improved else None,
              'validation_fraction': .3 if args.improved else None,
              'min_delta': 1e-4, 'precision': 'FP32', 'deterministic': True,
              'inference_batch_size': 512, 'train_probe_max_rows': 512,
              'optimizer': 'Adam', 'optimizer_fused': device == 'cuda',
              'evaluate_test': False,
              'mlp_iterations': 2 if args.smoke else 200, 'synthetic': args.smoke,
              'improved': args.improved, 'feature_profile': 'rich' if args.improved and not args.basic_features else 'basic',
              'tuning_trials': min(2, args.tuning_trials) if args.smoke else args.tuning_trials,
              'screen_epochs': 1 if args.smoke else args.screen_epochs,
              'models': list(MODELS) + list(ENSEMBLES) if args.improved else list(MODELS),
              'protocol': 'oulad_risk_nested_tuning_v2' if args.improved else 'oulad_landmark_composite_risk_bce_cox_v1'}
    print(f"Device: {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}", flush=True)
    print(f"{len(config['models'])} methods; windows {config['windows']}; {config['split_scheme']}; seeds {config['seeds']}; FP32.", flush=True)
    if args.check:
        print('Dependencies and input paths OK. No training started.', flush=True)
        return
    hashes = {'synthetic': 'seed3_n400'} if args.smoke else fingerprint(raw)
    paths = [Path(__file__), *root.glob('risk_experiments/*.py'), root / 'models/pred_oulad.py',
             root / 'models/oulad_hybrid_attention.py', root / 'data/oulad_loader.py', root / 'evaluation_metrics.py',
             root / 'run_improved_article.py']
    sources = {str(p.relative_to(root)): sha256(p) for p in paths}
    identity = {'config': config, 'data_sha256': hashes, 'source_sha256': sources, 'versions': versions}
    # Seeds identify independent job directories, not common feature/model settings.
    # Excluding the list lets a later invocation append seeds without refitting seed 42.
    signature = experiment_signature(identity)
    output = resume or root / 'runs' / ('risk_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    if resume:
        previous = json.loads((output / 'run_manifest.json').read_text())
        if previous['signature'] != signature:
            if logging_only_resume_compatible(previous, identity, output):
                signature = previous['signature']
            else:
                raise ValueError('Cannot resume: code, data, packages, device or experiment settings differ. Use the original flags.')
        if not set(previous['config']['seeds']).issubset(config['seeds']):
            raise ValueError('Keep completed seeds in --seeds when appending new seeds')
    else:
        output.mkdir(parents=True, exist_ok=False)
    for relative in sources:
        destination = output / 'source' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and sha256(destination) != sources[relative]:
            history = output / 'source_history' / sha256(destination) / relative
            history.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(destination, history)
        shutil.copy2(root / relative, destination)
    shutil.copy2(root / ('docs/IMPROVED_PROTOCOL_RU.md' if args.improved else 'docs/RISK_ARTICLE_PROTOCOL_RU.md'), output / 'protocol.md')
    environment = {'python': sys.version, 'platform': platform.platform(), 'processor': platform.processor(),
                   'logical_cpus': os.cpu_count(), 'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
                   'gpu': str(torch.cuda.get_device_properties(0)) if device == 'cuda' else None,
                   'packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions() if d.metadata['Name']}}
    write_json(output / 'environment.json', environment)
    manifest = {**identity, 'signature': signature, 'status': 'running', 'command': sys.argv,
                'output': str(output), 'started_utc': datetime.now(timezone.utc).isoformat(),
                'primary_contrast': 'proposed vs no_cox on 2013 validation; external 2014 test disabled',
                'score_availability_assumption': 'Scores omitted in primary analysis. Optional proxy=max(submission_date, due_date)+score_lag; not an observed release timestamp.',
                'cox_objective': 'minibatch partial likelihood, Breslow ties, lambda=cox_weight; event=withdrawal only',
                'selection': ('inner-only screening/refinement, matched ablations, separate calibration; outer held never selects settings'
                              if args.improved else 'fixed hyperparameters; neural early stop on inner validation raw BCE; no test-driven model selection')}
    write_json(output / 'run_manifest.json', manifest)
    (output / 'experiment_card.md').write_text(
        '# OULAD BCE + Cox experiment\n\n'
        'Primary endpoint: Fail + Withdrawn vs Pass + Distinction. Survival event: withdrawal.\n\n'
        'Learner-disjoint 2013 training/validation only. External 2014 test disabled. '
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
