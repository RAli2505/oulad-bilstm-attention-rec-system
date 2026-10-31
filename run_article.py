"""Run all comparison experiments into a new, reproducible output directory."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--latent-viz', action='store_true', help='Also compute t-SNE/PCA/UMAP')
    parser.add_argument('--check', action='store_true', help='Validate setup without training')
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    root = Path(__file__).resolve().parent
    experiment = root / 'experiments' / 'hybrid-attention'
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[name] = str(args.threads)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    sys.path.insert(0, str(experiment))
    import numpy as np
    import torch
    from importlib.metadata import version
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    required = ('torch', 'numpy', 'pandas', 'scikit-learn', 'pyarrow', 'scipy',
                'matplotlib', 'seaborn', 'tqdm', 'xgboost', 'lightgbm')
    packages = {name: version(name) for name in required}
    if args.latent_viz:
        packages['umap-learn'] = version('umap-learn')
    raw = root / 'data' / 'oulad' / 'raw'
    inputs = {}
    for name in ('studentInfo.csv', 'studentVle.csv', 'studentAssessment.csv', 'assessments.csv'):
        path = raw / name
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        inputs[name] = digest.hexdigest()
    print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU', flush=True)
    print('5 grouped folds; FP32; deterministic; batch 64; max 100 epochs; patience 7.', flush=True)
    if args.check:
        print('Dependencies and input data: OK', flush=True)
        return
    output = root / 'runs' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True, exist_ok=False)
    os.environ['OULAD_RUN_DIR'] = str(output)
    import config
    config.RUN_BASELINE_ML = config.RUN_KFOLD = config.RUN_ABLATION = True
    config.RUN_LATENT_VIZ = args.latent_viz
    config.KFOLD_EPOCHS = 100
    metadata = {
        'status': 'running', 'protocol': 'learner_disjoint_retrospective_success_v2',
        'seed': 42, 'folds': 5, 'max_epochs': 100, 'patience': 7,
        'batch_train': 64, 'precision': 'FP32', 'deterministic': True,
        'threads': args.threads, 'python': platform.python_version(), 'packages': packages,
        'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
        'inputs_sha256': inputs,
        'source_sha256': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [Path(__file__).resolve(), *experiment.glob('*.py')]},
    }
    manifest = output / 'run_manifest.json'
    manifest.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print('Results:', output, flush=True)
    os.chdir(experiment)
    try:
        with threadpool_limits(limits=args.threads):
            runpy.run_path(str(experiment / 'main.py'), run_name='__main__')
    except BaseException as exc:
        metadata.update(status='failed', error=repr(exc))
        raise
    else:
        metadata['status'] = 'completed'
    finally:
        manifest.write_text(json.dumps(metadata, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
