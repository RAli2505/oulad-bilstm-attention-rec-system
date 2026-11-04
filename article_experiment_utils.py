"""Logging and verified recovery for standalone article experiments."""
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

import pandas as pd
from run_oulad_agent_experiment import _Tee


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic_json(path, value):
    path = Path(path); temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_csv(path, frame):
    temporary = Path(path).with_suffix('.tmp')
    frame.to_csv(temporary, index=False); os.replace(temporary, path)


def ready(directory, identity):
    directory = Path(directory); marker = directory / 'checkpoint.json'
    if not marker.exists():
        return False
    saved = json.loads(marker.read_text(encoding='utf-8'))
    if saved['identity'] != identity:
        raise ValueError(f'Checkpoint configuration mismatch: {directory}')
    for name, digest in saved['files'].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or not path.is_file() or sha(path) != digest:
            raise ValueError(f'Checkpoint checksum mismatch: {path}')
    return True


def checkpoint(directory, identity):
    directory = Path(directory)
    files = {str(p.relative_to(directory)): sha(p) for p in directory.rglob('*')
             if p.is_file() and p.name not in ('checkpoint.json', 'checkpoint.json.tmp')}
    atomic_json(directory / 'checkpoint.json', {'identity': identity, 'files': files})
    print(f'Checkpoint saved: {directory}', flush=True)


@contextmanager
def session(prefix, config, script, output=None, resume=None):
    root = Path(__file__).parent
    out = Path(resume or output or root / 'runs' / (prefix + '_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))).resolve()
    config = {**config, 'script_sha256': sha(script), 'utility_sha256': sha(Path(__file__)),
              'versions': {n: importlib.metadata.version(n) for n in ['numpy', 'pandas', 'scikit-learn', 'torch', 'lightgbm']}}
    if resume and json.loads((out / 'config.json').read_text(encoding='utf-8')) != config:
        raise ValueError('Resume configuration, code or environment mismatch')
    out.mkdir(parents=True, exist_ok=bool(resume))
    lock = out / 'writer.lock'
    if lock.exists():
        import psutil
        saved = json.loads(lock.read_text())
        try:
            if abs(psutil.Process(saved['pid']).create_time() - saved['created']) < .01:
                raise ValueError(f'Another writer is active: {out}')
        except psutil.NoSuchProcess:
            pass
        lock.unlink()
    import psutil
    with lock.open('x', encoding='utf-8') as stream:
        json.dump({'pid': os.getpid(), 'created': psutil.Process().create_time()}, stream)
    atomic_json(out / 'config.json', config)
    started = time.perf_counter()
    try:
        with (out / f'log_{time.time_ns()}.txt').open('x', encoding='utf-8') as log, redirect_stdout(_Tee(sys.stdout, log)), redirect_stderr(_Tee(sys.stderr, log)):
            print(f'Output: {out}', flush=True)
            atomic_json(out / 'status.json', {'status': 'running'})
            try:
                yield out, config
            except BaseException as exc:
                import traceback
                traceback.print_exc()
                atomic_json(out / 'status.json', {'status': 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', 'error': repr(exc), 'wall_seconds': time.perf_counter() - started})
                raise
            atomic_json(out / 'status.json', {'status': 'completed', 'wall_seconds': time.perf_counter() - started})
            print(f'Finished: {out}', flush=True)
    finally:
        lock.unlink(missing_ok=True)
