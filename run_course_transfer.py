"""Leave-one-course-out risk training on 2013 only; held students excluded from fit."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from article_experiment_utils import session, sha, ready, checkpoint, atomic_csv, atomic_json


def logged_fit_function(training):
    """Add display-only epoch progress in this process, preserving archived code."""
    import ast
    import inspect
    tree = ast.parse(inspect.getsource(training.fit_predict))
    class Progress(ast.NodeTransformer):
        def visit_If(self, node):
            node = self.generic_visit(node)
            if ast.unparse(node.test) == "monitor < best_loss - config['min_delta']":
                message = '''print(f"{name} seed {seed} epoch {epoch + 1}/{config['epochs']}: val_loss={monitor:.6f}; epoch_seconds={trace[-1]['seconds']:.1f}; remaining_epoch_budget={config['epochs'] - epoch - 1}", flush=True)'''
                return [ast.parse(message).body[0], node]
            return node
    tree = ast.fix_missing_locations(Progress().visit(tree))
    namespace = dict(vars(training))
    exec(compile(tree, '<course-transfer-epoch-progress>', 'exec'), namespace)
    return namespace['fit_predict'], ast.unparse(tree)


def course_cohort(target, course):
    held_students = set(target.loc[target.code_module == course, 'id_student'])
    # Returning students must not leak via enrollment on a different course.
    selected = target.loc[(target.code_module == course) | ~target.id_student.isin(held_students)].copy()
    if not held_students or not (selected.code_module != course).any():
        raise ValueError('Both held course and training courses required')
    return selected, np.array(sorted(held_students))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--risk-run', type=Path, required=True, help='Protocol template only, not fitted models')
    parser.add_argument('--windows', type=int, nargs='+', default=[28, 56, 112])
    parser.add_argument('--courses', nargs='+', help='Default: every 2013 course')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42])
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    parser.add_argument('--output', type=Path); parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    if min(args.windows + [args.threads]) < 1 or not args.seeds:
        parser.error('Positive windows, threads and nonempty seeds required')
    for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        os.environ[name] = str(args.threads)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True); torch.use_deterministic_algorithms(True)
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    if device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable')
    from risk_experiments.features import fingerprint, load_data
    import risk_experiments.pipeline as pipeline
    import risk_experiments.training as training
    import risk_experiments.tuning as tuning
    root = Path(__file__).parent
    source = args.risk_run.resolve(); template = json.loads((source / 'run_manifest.json').read_text())
    if template['status'] != 'completed' or template['config']['evaluate_test']:
        raise ValueError('Completed validation-only protocol source required')
    raw = root / 'data/oulad/raw'
    print('Verifying raw data and training sources', flush=True)
    hashes = fingerprint(raw)
    if hashes != template['data_sha256']:
        raise ValueError('Raw data differs from protocol template')
    files = [*root.glob('risk_experiments/*.py'), *root.glob('models/*oulad*.py'),
             root / 'data/oulad_loader.py', root / 'evaluation_metrics.py']
    config = {**template['config'], 'windows': args.windows, 'seeds': args.seeds,
              'device': device, 'threads': args.threads, 'optimizer_fused': device == 'cuda',
              'models': ['proposed', 'lstm', 'lightgbm', 'ensemble_lstm_attention_lgbm_weighted'],
              'system_only': True, 'evaluate_test': False,
              'protocol': '2013_leave_one_course_out_learner_disjoint', 'requested_courses': args.courses,
              'data_sha256': hashes, 'training_source_sha256': {str(p.relative_to(root)): sha(p) for p in files},
              'template_manifest_sha256': sha(source / 'run_manifest.json'),
              'inner_policy': 'fit/selection/calibration only on remaining courses; no held-course tuning',
              'output_label_note': 'validation_2013 files are held-course results; nominal 70/30 metadata does not apply',
              'primary_day': 56 if 56 in args.windows else args.windows[0]}
    with session('course_transfer', config, __file__, args.output, args.resume) as (output, identity), threadpool_limits(limits=args.threads):
        logged_fit, logged_source = logged_fit_function(training)
        (output / 'epoch_logged_training.py').write_text(logged_source, encoding='utf-8')
        pipeline.fit_predict = tuning.fit_predict = logged_fit
        print(f'Loading data; device={device}; epochs={config["epochs"]}; patience={config["patience"]}', flush=True)
        target, daily, assess = load_data(raw, root / '.risk-cache', hashes)
        target = target.loc[target.code_presentation.isin(['2013B', '2013J'])].copy()
        daily = daily.loc[daily.code_presentation.isin(['2013B', '2013J'])]
        assess = assess.loc[assess.code_presentation.isin(['2013B', '2013J'])]
        courses = args.courses or sorted(target.code_module.unique())
        if not set(courses) <= set(target.code_module):
            raise ValueError('Requested course missing from 2013')
        summaries = []; original_split = pipeline.holdout_student_groups
        try:
            for course in courses:
                cohort, held_students = course_cohort(target, course)
                if set(cohort.loc[cohort.code_module != course, 'id_student']) & set(held_students):
                    raise AssertionError('Held students leaked into training courses')
                pipeline.holdout_student_groups = lambda frame, seed, ids=held_students: ids
                for day in args.windows:
                    directory = output / f'held_{course}' / f'day_{day}'; directory.mkdir(parents=True, exist_ok=True)
                    job_identity = {**identity, 'held_course': course, 'day': day}
                    if not ready(directory, job_identity):
                        signature = hashlib.sha256(json.dumps(job_identity, sort_keys=True).encode()).hexdigest()
                        print(f'Held course={course}, day={day}: training only other courses; seeds={args.seeds}', flush=True)
                        pipeline.run_window(cohort, daily, assess, {**config, 'days': day}, directory, signature)
                        audit = pd.read_csv(directory / 'cohort_and_folds.csv')
                        held = audit.loc[audit.role == 'validation_2013']
                        train = audit.loc[audit.role == 'train_pool_2013']
                        if not held.code_module.eq(course).all() or train.code_module.eq(course).any() or set(train.id_student) & set(held.id_student):
                            raise AssertionError('Course/learner isolation audit failed')
                        atomic_json(directory / 'course_transfer_audit.json', {
                            'held_course': course, 'held_students': held.id_student.nunique(),
                            'training_courses': sorted(train.code_module.unique().tolist()),
                            'student_overlap': 0, 'held_course_rows_in_train': 0, 'test_2014_used': False})
                        checkpoint(directory, job_identity)
                    summaries.append(pd.read_csv(directory / 'summary.csv').assign(held_course=course, day=day))
                    atomic_csv(output / 'summary.csv', pd.concat(summaries, ignore_index=True))
        finally:
            pipeline.holdout_student_groups = original_split


if __name__ == '__main__':
    main()
