import sys

import run_oulad_agent_experiment as runner


def test_cli_duplicates_progress_to_log_without_launching_training(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(runner, '__file__', str(tmp_path / 'runner.py'))
    monkeypatch.setattr(sys, 'argv', ['runner.py', '--risk-run', str(tmp_path / 'risk'),
                                    '--output', str(tmp_path / 'output'), '--inference-batch-size', '128'])
    def fake_run(**kwargs):
        assert kwargs['inference_batch_size'] == 128
        print('Personal signals: 128/256 queries; ETA=2s', flush=True)
        print('example stderr', file=sys.stderr, flush=True)
        return kwargs['output']
    monkeypatch.setattr(runner, 'run_full_agent', fake_run)
    runner.main()
    logs = list((tmp_path / 'runs').glob('*.log'))
    assert len(logs) == 1
    contents = logs[0].read_text(encoding='utf-8')
    captured = capsys.readouterr()
    assert '128/256' in contents and '128/256' in captured.out
    assert 'example stderr' in contents and 'example stderr' in captured.err
