from copy import deepcopy
import hashlib

import pytest
import run_risk_article as runner


def fixture(tmp_path, monkeypatch):
    root, output = tmp_path / 'root', tmp_path / 'run'
    root.mkdir()
    (output / 'source').mkdir(parents=True)
    monkeypatch.setattr(runner, '__file__', str(root / 'run_risk_article.py'))
    old = "def train():\n    config = {'epochs': 100}\n    return config\n"
    new = "def train():\n    config = {'epochs': 100, 'system_only': args.system_only}\n    return config\n"
    (root / 'run_risk_article.py').write_text(new)
    (output / 'source/run_risk_article.py').write_text(old)
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    previous = {'config': {'epochs': 100, 'seeds': [42]}, 'data_sha256': {'data': 'original'},
                'versions': {'torch': 'original'},
                'source_sha256': {'run_risk_article.py': digest(output / 'source/run_risk_article.py')}}
    identity = deepcopy(previous)
    identity['config'].update(system_only=False, seeds=[42, 13])
    identity['source_sha256']['run_risk_article.py'] = digest(root / 'run_risk_article.py')
    return root, output, previous, identity


def test_accepts_verified_inactive_system_only_addition(tmp_path, monkeypatch):
    _, output, previous, identity = fixture(tmp_path, monkeypatch)
    assert runner.legacy_full_resume_compatible(previous, identity, output)


@pytest.mark.parametrize('change', ['system_only', 'epochs', 'data', 'packages', 'training_code', 'snapshot'])
def test_rejects_scientific_or_provenance_changes(tmp_path, monkeypatch, change):
    root, output, previous, identity = fixture(tmp_path, monkeypatch)
    if change == 'system_only':
        identity['config']['system_only'] = True
    elif change == 'epochs':
        identity['config']['epochs'] = 200
    elif change == 'data':
        identity['data_sha256']['data'] = 'changed'
    elif change == 'packages':
        identity['versions']['torch'] = 'changed'
    elif change == 'training_code':
        path = root / 'run_risk_article.py'
        path.write_text(path.read_text().replace('100', '200'))
        identity['source_sha256']['run_risk_article.py'] = hashlib.sha256(path.read_bytes()).hexdigest()
    elif change == 'snapshot':
        path = output / 'source/run_risk_article.py'
        path.write_text(path.read_text().replace('100', '200'))
    assert not runner.legacy_full_resume_compatible(previous, identity, output)
