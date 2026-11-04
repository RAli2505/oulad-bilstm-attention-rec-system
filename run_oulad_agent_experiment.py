"""Train/evaluate the standalone full agent against an immutable risk run."""
import argparse
import os
import sys
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone, timedelta
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

from recommendations.full_agent_experiment import run_full_agent


class _Tee:
    def __init__(self, console, logfile):
        self.console, self.logfile = console, logfile
        self.line_start = True

    def write(self, value):
        for part in value.splitlines(keepends=True):
            prefix = (datetime.now(timezone(timedelta(hours=5))).strftime('%H:%M') + ' ') if self.line_start and part.strip() else ''
            self.console.write(prefix + part)
            self.logfile.write(prefix + part)
            self.line_start = part.endswith(('\n', '\r'))
        self.logfile.flush()
        return len(value)

    def flush(self):
        self.console.flush()
        self.logfile.flush()

    def __getattr__(self, name):
        return getattr(self.console, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--risk-run', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--windows', nargs='+', type=int, default=list(range(7, 169, 7)))
    parser.add_argument('--seeds', nargs='+', type=int, default=[42])
    parser.add_argument('--horizon', type=int, default=7)
    parser.add_argument('--ks', nargs='+', type=int, default=[5, 10, 20])
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--inference-batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--patience', type=int, default=4)
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--max-learners', type=int, help='Pilot learners per split; not publication results')
    parser.add_argument('--raw', type=Path)
    args = vars(parser.parse_args())
    resume_path = args.pop('resume')
    args['resume'] = resume_path is not None
    if resume_path is not None:
        args['output'] = resume_path
    if args['output'] is None:
        args['output'] = Path('runs') / ('full_agent_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    logs = Path(__file__).resolve().parent / 'runs'
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / (args['output'].name + '_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '.log')
    print(f'Console log: {log_path}', flush=True)
    with log_path.open('x', encoding='utf-8', buffering=1) as logfile:
        with redirect_stdout(_Tee(sys.stdout, logfile)), redirect_stderr(_Tee(sys.stderr, logfile)):
            try:
                print(run_full_agent(**args), flush=True)
            except BaseException:
                import traceback
                traceback.print_exc()
                raise


if __name__ == '__main__':
    main()
