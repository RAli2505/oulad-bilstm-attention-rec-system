"""Regression checks for fold isolation, checkpoints and faster batching."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

EXPERIMENT = Path(__file__).resolve().parents[1] / 'experiments' / 'hybrid-attention'
sys.path.insert(0, str(EXPERIMENT))
import training
from protocol import folds, outer_split
sys.path.remove(str(EXPERIMENT))


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 2)

    def forward(self, static, dynamic):
        return self.fc(static)


def test_student_isolation_and_fold_coverage():
    groups = np.repeat(np.arange(100), 2)
    y = np.tile([0, 1], 100)
    dev, test, train, val = outer_split(y, groups)
    assert not set(groups[dev]) & set(groups[test])
    assert not set(groups[train]) & set(groups[val])
    coverage = np.zeros(len(dev))
    for fit, held in folds(y[dev], groups[dev]):
        assert not set(groups[dev][fit]) & set(groups[dev][held])
        coverage[held] += 1
    assert np.all(coverage == 1)


def test_resident_batches_match_original_order():
    tensors = (torch.arange(24).reshape(8, 3).float(), torch.ones(8, 2), torch.arange(8) % 2)
    loader = DataLoader(TensorDataset(*tensors), batch_size=3)
    cached = training.ResidentLoader(loader)
    for original, resident in zip(loader, cached):
        for a, b in zip(original, resident):
            assert torch.equal(a, b.cpu())


def test_fold_checkpoint_cannot_replace_final_model(tmp_path, monkeypatch):
    monkeypatch.setattr(training, 'MODEL_PATH', str(tmp_path / 'best_model.pth'))
    generator = torch.Generator().manual_seed(3)
    x = torch.randn(16, 3, generator=generator)
    y = torch.arange(16) % 2
    loader = DataLoader(TensorDataset(x, torch.ones(16, 2), y), batch_size=8)
    training.train_and_evaluate('Hybrid (Balanced)', TinyNet(), loader, loader, loader,
                               y.numpy(), epochs=1)
    final = (tmp_path / 'best_model.pth').read_bytes()
    training.train_and_evaluate('Hybrid (Balanced)-fold1', TinyNet(), loader, loader, loader,
                               y.numpy(), epochs=1)
    assert (tmp_path / 'best_model.pth').read_bytes() == final
    assert len(list((tmp_path / 'checkpoints').glob('*.pth'))) == 2
    assert len(list((tmp_path / 'checkpoints').glob('*.predictions.npz'))) == 2
