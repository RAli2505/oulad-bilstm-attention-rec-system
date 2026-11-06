"""Bounded parity/median benchmark; no fitting and no test outcomes."""
from contextlib import redirect_stdout
from datetime import datetime,timezone,timedelta
import io
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from fast_full_agent_resume import fast_baskets,digest
from recommendations.sequential import DailySASRecService
from recommendations.experiment import load_events


def main():
    root=Path(__file__).resolve().parent
    output=root.parent/'tmp/fast_full_agent_validation'; output.mkdir(exist_ok=True)
    from run_oulad_agent_experiment import _Tee
    import sys
    with (output/'verification.log').open('w',encoding='utf-8') as logfile, redirect_stdout(_Tee(sys.stdout,logfile)):
        source=root/'runs/full_agent_20261102T190210371601Z'
        print('Bounded 2013-only benchmark: no model fitting, no test evaluation',flush=True)
        queries=pd.read_csv(source/'day_98/queries.csv')
        # Different course contexts and roles, capped only for this benchmark.
        sampled=queries.groupby(['code_module','role'],sort=True).head(2)
        keys=sampled[['id_student','code_module','code_presentation']].drop_duplicates()
        manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
        print(f'Loading pre-cutoff 2013 history for {len(keys)} benchmark enrollments',flush=True)
        events=load_events(Path(manifest['raw_source']),keys,98)
        if not events.code_presentation.isin(['2013B','2013J']).all() or not events.date.lt(98).all():
            raise ValueError('Benchmark must remain in pre-cutoff 2013 histories')
        service=DailySASRecService()
        resource_keys=sorted(set(events[['code_module','code_presentation','id_site']].itertuples(index=False,name=None)),key=repr)
        service.item_to_index={tuple(k):i+1 for i,k in enumerate(resource_keys)}
        memberships=set(keys.itertuples(index=False,name=None))
        requests=[(key,part.copy()) for key,part in events.groupby(['id_student','code_module','code_presentation'],sort=True)]
        # Warmup both implementations; compare complete token IDs, exact FP64
        # strengths, ordering and prefixes, not only aggregate counts.
        for key,history in requests:
            expected=DailySASRecService._baskets(service,history,{key},98)
            actual=fast_baskets(service,history,{key},98)
            assert actual==expected
            assert service._prefixes(actual,limit=8)==service._prefixes(expected,limit=8)
        # Multi-learner training path plus cold/unknown/cutoff/duplicate cases.
        assert fast_baskets(service,events,memberships,98)==DailySASRecService._baskets(service,events,memberships,98)
        extra=events.iloc[:12].copy()
        if len(extra):
            extra=pd.concat([extra,extra,extra.assign(date=98),extra.assign(id_site=-999)],ignore_index=True)
            assert fast_baskets(service,extra,memberships,98)==DailySASRecService._baskets(service,extra,memberships,98)
        assert fast_baskets(service,events.iloc[:0],memberships,98)=={}
        timings={'reference':[],'fast':[]}
        for repeat in range(3):
            for name,fn in [('reference',DailySASRecService._baskets),('fast',fast_baskets)]:
                started=time.perf_counter()
                for key,history in requests: fn(service,history,{key},98)
                elapsed=time.perf_counter()-started; timings[name].append(elapsed)
                print(f'benchmark repeat={repeat+1}/3 method={name}; queries={len(requests)}; elapsed={elapsed:.3f}s',flush=True)
        medians={k:float(np.median(v)) for k,v in timings.items()}
        report={'local_time':datetime.now(timezone(timedelta(hours=5))).isoformat(),
                'exact_equal':True,'queries':len(requests),'history_rows':len(events),
                'checks':'all baskets exact; prefix pairs exact; multi-user, duplicates, cold, unknown and cutoff cases',
                'median_seconds':medians,'basket_stage_speedup':medians['reference']/medians['fast'],
                'whole_experiment_speedup_measured':False,'test_2014_used':False,
                'fast_code_sha256':digest(root/'fast_full_agent_resume.py'),
                'original_sequential_sha256':digest(root/'recommendations/sequential.py'),
                'original_full_agent_sha256':digest(root/'recommendations/full_agent_experiment.py')}
        (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
