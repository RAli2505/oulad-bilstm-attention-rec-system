"""Atomic, checksum-verified checkpoints for complete experiment jobs."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def complete_job(directory, day, seed):
    directory = Path(directory)
    required = ['runtime_memory.json', 'ranking_summary.csv', 'held_candidate_predictions.npz',
                'agent_bundle/bundle.json', 'contextual_probability_metrics.csv', 'generalization_gaps.csv']
    if not all((directory / name).is_file() for name in required):
        return False
    marker = directory / 'checkpoint.json'
    if marker.exists():
        saved = json.loads(marker.read_text(encoding='utf-8'))
        if saved['day'] != day or saved['seed'] != seed or saved['status'] != 'completed':
            raise ValueError(f'Checkpoint identity mismatch: {directory}')
        for name, expected in saved['files_sha256'].items():
            path = (directory / name).resolve()
            if not path.is_relative_to(directory.resolve()) or not path.is_file() or digest(path) != expected:
                raise ValueError(f'Checkpoint checksum mismatch: {path}')
    # Also validate legacy complete jobs written before checkpoints existed.
    bundle = json.loads((directory / 'agent_bundle/bundle.json').read_text(encoding='utf-8'))
    for name, expected in bundle['artifacts_sha256'].items():
        path = (directory / 'agent_bundle' / name).resolve()
        if not path.is_relative_to((directory / 'agent_bundle').resolve()) or digest(path) != expected:
            raise ValueError(f'Bundle checksum mismatch: {path}')
    summary = pd.read_csv(directory / 'ranking_summary.csv')
    if summary.empty or not {'role', 'method', 'k'}.issubset(summary):
        raise ValueError(f'Invalid completed summary: {directory}')
    with np.load(directory / 'held_candidate_predictions.npz', allow_pickle=False) as saved:
        arrays = [saved[name] for name in ('labels', 'score', 'probability')]
        if not arrays[0].size or any(a.shape != arrays[0].shape or not np.isfinite(a).all() for a in arrays):
            raise ValueError(f'Invalid completed predictions: {directory}')
    return True


def save_job_checkpoint(directory, day, seed):
    directory = Path(directory)
    if not complete_job(directory, day, seed):
        raise ValueError(f'Cannot checkpoint incomplete job: {directory}')
    files = {str(p.relative_to(directory)): digest(p) for p in directory.rglob('*')
             if p.is_file() and p.name not in ('checkpoint.json', 'checkpoint.json.tmp')}
    temporary = directory / 'checkpoint.json.tmp'
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump({'status': 'completed', 'day': day, 'seed': seed, 'files_sha256': files}, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, directory / 'checkpoint.json')
