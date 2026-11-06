"""Verified execution-only acceleration; original full-agent code stays intact."""
import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import pandas as pd

from recommendations.sequential import DailySASRecService, ENROLLMENT


def fast_baskets(self, interactions, keys, day):
    """Same grouping, sums, scalar log1p and sorting; one row iterator per user.

    Avoid a separate pandas subframe/itertuples factory for each course day.
    Neither training prefixes, RNG, precision nor GPU batching change.
    """
    if interactions.empty:
        return {}
    required=ENROLLMENT+['id_site','date','sum_click']
    missing=set(required)-set(interactions.columns)
    if missing:
        raise ValueError(f'Missing interaction columns: {sorted(missing)}')
    frame=interactions.loc[pd.to_numeric(interactions.date,errors='coerce')<day,required].copy()
    frame=frame.loc[[tuple((int(r[0]),str(r[1]),str(r[2]))) in keys
                     for r in frame[ENROLLMENT].itertuples(index=False,name=None)]]
    frame['sum_click']=pd.to_numeric(frame.sum_click,errors='coerce').fillna(0).clip(lower=0)
    frame=frame.groupby(ENROLLMENT+['date','id_site'],sort=True,as_index=False).sum(numeric_only=True)
    result={}
    for position,(enrollment,events) in enumerate(frame.groupby(ENROLLMENT,sort=True),1):
        days=[]; basket=[]; current=None
        for row in events.itertuples(index=False):
            if current is not None and row.date!=current:
                if basket: days.append(sorted(basket))
                basket=[]
            current=row.date
            idx=self.item_to_index.get((str(row.code_module),str(row.code_presentation),int(row.id_site)))
            if idx is not None and row.sum_click>0:
                strength=float(np.log1p(min(row.sum_click,self.config['click_cap'])))
                basket.append((idx,strength))
        if basket: days.append(sorted(basket))
        result[tuple(enrollment)]=days
        if len(keys)>1 and (position%512==0 or position==len(keys)):
            print(f'  SASRec baskets: {position}/{len(keys)} enrollments',flush=True)
    return result


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--validation-report',type=Path,required=True)
    options,remaining=parser.parse_known_args()
    report=json.loads(options.validation_report.read_text(encoding='utf-8'))
    root=Path(__file__).resolve().parent
    if not report.get('exact_equal') or report['fast_code_sha256']!=digest(__file__):
        raise ValueError('Acceleration validation missing or stale')
    if report['original_sequential_sha256']!=digest(root/'recommendations/sequential.py'):
        raise ValueError('Sequential source differs from validated reference')
    if report['original_full_agent_sha256']!=digest(root/'recommendations/full_agent_experiment.py'):
        raise ValueError('Full-agent source differs from validated reference')
    if '--resume' not in remaining:
        raise ValueError('This entry point is for verified continuation only')
    output=Path(remaining[remaining.index('--resume')+1]).resolve()
    import psutil
    for process in psutil.process_iter(['pid','name','cmdline']):
        if process.pid==os.getpid(): continue
        # RTK and py.exe parents carry the same script argument but are not
        # additional experiment writers.
        if not (process.info['name'] or '').lower().startswith('python'): continue
        args=process.info['cmdline'] or []
        if any(Path(arg).name in ['run_oulad_agent_experiment.py','fast_full_agent_resume.py'] for arg in args):
            raise ValueError('Another full-agent process exists; do not run two writers')
    metadata={'local_time':datetime.now(timezone(timedelta(hours=5))).isoformat(),
              'execution_change':'replace per-day pandas basket iterators with per-user iterator',
              'scientific_protocol_changed':False,'test_2014_used':False,
              'fast_code_sha256':digest(__file__),'validation':report}
    with (output/('execution_acceleration_'+str(os.getpid())+'.json')).open('x',encoding='utf-8') as stream:
        json.dump(metadata,stream,ensure_ascii=False,indent=2)
    DailySASRecService._baskets=fast_baskets
    print(datetime.now(timezone(timedelta(hours=5))).strftime('%H:%M')+' Verified fast basket construction; Python 3.12; original checkpoints retained',flush=True)
    import run_oulad_agent_experiment
    sys.argv=[str(root/'run_oulad_agent_experiment.py'),*remaining]
    run_oulad_agent_experiment.main()


if __name__=='__main__':
    main()
