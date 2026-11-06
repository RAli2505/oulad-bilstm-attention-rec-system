"""Frozen OULAD ranking: repeat/discovery quotas, learned policy and baselines.

Only exploratory 2013 fit/selection/held are allowed. No 2014 outcome/test files
are opened. Original models and source runs are read-only.
"""
from __future__ import annotations

import argparse
import csv
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
import io
import json
from pathlib import Path
import threading
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from article_experiment_utils import session, sha, ready, checkpoint, atomic_csv, atomic_json
from recommendations.full_agent_experiment import compact_rank
from recommendations.experiment_metrics import ranking_metrics, cluster_interval, ResourceMonitor

KEY = ['id_student', 'code_module', 'code_presentation']
FEATURES = ['log_seen_items', 'log_resource_days', 'repeat_concentration',
            'mean_preference_confidence', 'preference_dispersion', 'recent_repeat_share', 'course_day']
BASELINES = ['last_visited', 'most_frequent', 'popularity']
MIX_WEIGHTS = (0., .25, .5, .75, 1.)


def mixed_scores(items, scores, seen, age, recent_visits, popularity, recency, discovery):
    """Reorder the score slots of each group, then blend with their original order."""
    order = np.lexsort((items, -scores))
    base = np.empty(len(items), dtype=float)
    base[order] = np.arange(len(items), 0, -1)
    blended = base.copy()
    for mask, weight, primary, secondary in (
        (seen, recency, age, -recent_visits),
        (~seen, discovery, -popularity, -scores),
    ):
        if weight == 0:
            continue
        group = np.flatnonzero(mask)
        ranked = group[np.lexsort((items[group], -scores[group], secondary[group], primary[group]))]
        blended[ranked] = (1 - weight) * base[ranked] + weight * np.sort(base[group])[::-1]
    return blended
TZ = timezone(timedelta(hours=5))


@contextmanager
def stage(name):
    """Keep IO, hashing, prediction and fitting visible even without counters."""
    started = time.perf_counter()
    stop = threading.Event()
    print(f'{name}: start; stage ETA=unknown', flush=True)
    def pulse():
        while not stop.wait(20):
            print(f'{name}: active; elapsed={time.perf_counter()-started:.0f}s; stage ETA=unknown', flush=True)
    worker = threading.Thread(target=pulse, daemon=True); worker.start()
    try:
        yield
    finally:
        stop.set(); worker.join()
        print(f'{name}: end; elapsed={time.perf_counter()-started:.1f}s', flush=True)


def quota_rank(repeat_order, new_order, fraction, k, score_by_item):
    """Allocate round-half-up repeat slots, backfill shortages, preserve score order.

    Fractions 0 and 1 are quotas with backfill, NOT strict candidate exclusions.
    Ranking is recomputed separately for each K.
    """
    if not 0 <= fraction <= 1 or k < 1:
        raise ValueError('Invalid quota/K')
    nr = int(np.floor(k * fraction + .5))
    selected = list(repeat_order[:nr]) + list(new_order[:k-nr])
    used = set(selected)
    rest = [i for i in [*repeat_order, *new_order] if i not in used]
    rest.sort(key=lambda i: (-score_by_item[i], i))
    selected.extend(rest[:max(0, k-len(selected))])
    selected.sort(key=lambda i: (-score_by_item[i], i))
    return selected[:k]


def split_metrics(ranked, truth, seen, k):
    values = ranking_metrics(ranked, truth, k)
    for group, group_truth in [('repeat', truth & seen), ('new', truth - seen)]:
        group_values = ranking_metrics(ranked, group_truth, k)
        for metric in ['recall', 'ndcg', 'map', 'hits']:
            values[f'{group}_{metric}'] = group_values[metric] if group_truth else np.nan
        values[f'{group}_support'] = int(bool(group_truth))
        values[f'{group}_relevant'] = len(group_truth)
    top = ranked[:k]
    values['repeat_share'] = sum(i in seen for i in top) / len(top) if top else 0.
    values['new_share'] = sum(i not in seen for i in top) / len(top) if top else 0.
    return values


def history_features(days, preference, confidence, recent, day):
    seen = days > 0; total = float(days.sum())
    return np.array([np.log1p(seen.sum()), np.log1p(total),
                     float(days.max()/total) if total else 0.,
                     float(np.mean(confidence)) if len(confidence) else 0.,
                     float(np.std(preference)) if len(preference) else 0.,
                     float(np.mean(recent[seen])) if seen.any() else 0., day/168.], dtype=float)


def validate_queries(queries, data):
    if not set(queries.role).issubset({'fit', 'selection', 'calibration', 'held'}):
        raise ValueError('Unknown/test role rejected')
    if not queries.code_presentation.isin(['2013B', '2013J']).all():
        raise ValueError('Only 2013 presentations permitted; test remains closed')
    if queries['query'].duplicated().any():
        raise ValueError('Duplicate query')
    roles = queries.groupby('id_student').role.nunique()
    if (roles > 1).any():
        raise ValueError('Learner crosses split roles')
    starts, stops = queries.start.to_numpy(), queries.stop.to_numpy()
    if len(queries) == 0 or starts[0] != 0 or stops[-1] != len(data) or not np.array_equal(starts[1:], stops[:-1]) or (stops <= starts).any():
        raise ValueError('Candidate boundaries not exhaustive/contiguous')
    expected = np.repeat(queries['query'].to_numpy(), stops-starts)
    if not np.array_equal(expected, data['query'].to_numpy()):
        raise ValueError('Candidate/query alignment mismatch')
    if data.duplicated(['query', 'item']).any() or not data.label.isin([0, 1]).all():
        raise ValueError('Nonunique candidates or nonbinary labels')


def source_paths(source, day, seed):
    base=source/f'day_{day}'; job=base/f'seed_{seed}'
    if not (job/'runtime_memory.json').is_file():
        raise ValueError(f'Frozen source job incomplete: {job}')
    return [base/'queries.csv', base/'candidate_features_and_labels.parquet',
            job/'full_candidate_features.npy', job/'candidate_utilities.npz',
            job/'agent_bundle/ranking.joblib', job/'agent_bundle/bundle.json', job/'runtime_memory.json']


def build_history(raw_file, queries, day):
    """Parse only 2013 pre-cutoff lines; never load final outcomes or 2014 tests.

    The source studentVle.csv is a mixed-year interaction journal. Presentation
    and date are filtered as text BEFORE numeric parsing/aggregation.
    """
    wanted=set(map(tuple, queries[KEY].itertuples(index=False, name=None)))
    columns=KEY+['id_site', 'date', 'sum_click']
    parts=[]; batch=[]; scanned=kept=0; start=time.perf_counter(); last=start
    def consume(lines):
        if not lines: return
        frame=pd.DataFrame(lines, columns=header)[columns]
        for column in ['id_student', 'id_site', 'date', 'sum_click']:
            frame[column]=pd.to_numeric(frame[column],errors='raise')
        keys=list(frame[KEY].itertuples(index=False, name=None))
        frame=frame.loc[[k in wanted for k in keys]]
        if len(frame): parts.append(frame)
    with raw_file.open(encoding='utf-8-sig') as stream:
        reader=csv.reader(stream)
        header=next(reader)
        pres=header.index('code_presentation'); date=header.index('date'); clicks=header.index('sum_click')
        for fields in reader:
            scanned+=1
            if fields[pres] in ('2013B','2013J') and int(fields[date]) < day and int(fields[clicks]) > 0:
                batch.append(fields); kept+=1
            if len(batch)>=100000: consume(batch); batch=[]
            if time.perf_counter()-last>=20:
                elapsed=time.perf_counter()-start
                print(f'history day={day}: scanned={scanned}; eligible2013={kept}; {scanned/max(elapsed,1e-9):.0f} lines/s; stage ETA=unknown',flush=True)
                last=time.perf_counter()
        consume(batch)
    if not parts: return pd.DataFrame(columns=KEY+['item','last_date','clicks','resource_days','recent_visits'])
    events=pd.concat(parts,ignore_index=True)
    # Count distinct positive resource-days exactly as the frozen facade does.
    daily=events.groupby(KEY+['id_site','date'],as_index=False).sum_click.sum()
    daily['recent_visit'] = daily.date.ge(day - 7).astype(int)
    result=daily.groupby(KEY+['id_site'],as_index=False).agg(last_date=('date','max'),clicks=('sum_click','sum'),resource_days=('date','size'),recent_visits=('recent_visit','sum'))
    return result.rename(columns={'id_site':'item'})


def prepare_job(source, day, seed, history, fractions, ks, directory, identity, batch_size, improve=False):
    """Cache policy training targets and held rankings, checkpoint every 256 queries."""
    directory.mkdir(exist_ok=True, parents=True)
    complete=directory/'complete'
    if ready(complete, identity):
        print(f'day={day} seed={seed}: reuse prepared stage',flush=True)
        return
    paths=source_paths(source,day,seed)
    with stage(f'day={day} seed={seed} loading frozen inputs'):
        queries=pd.read_csv(paths[0]); data=pd.read_parquet(paths[1]); validate_queries(queries,data)
        settings=json.loads(paths[5].read_text(encoding='utf-8'))
        if sha(paths[4]) != settings['artifacts_sha256']['ranking.joblib']: raise ValueError('Bundle checksum mismatch')
        model=joblib.load(paths[4]); names=model['feature_names']
        if names!=settings['feature_names']: raise ValueError('Feature schema mismatch')
        x=np.load(paths[2],mmap_mode='r',allow_pickle=False)
        with np.load(paths[3],allow_pickle=False) as z:
            days=z['history_resource_days'].copy(); affinity=z['sasrec_affinity'].copy(); fallback=z['fallback'].copy()
            if not np.array_equal(z['labels'],data.label.to_numpy()): raise ValueError('Utility labels mismatch')
        if x.shape!=(len(data),len(names)) or len(days)!=len(data): raise ValueError('Feature length mismatch')
        scores=np.empty(len(data),dtype=float)
        if model['candidate_model'] is None: scores[:]=fallback
        else:
            started=time.perf_counter()
            for pos in range(0,len(data),batch_size):
                end=min(pos+batch_size,len(data)); scores[pos:end]=model['candidate_model'].predict_proba(x[pos:end])[:,1]
                elapsed=time.perf_counter()-started
                print(f'day={day} seed={seed} frozen inference: {end}/{len(data)} pairs; {end/max(elapsed,1e-9):.0f}/s; stage ETA={elapsed/end*(len(data)-end):.0f}s',flush=True)
        if not np.isfinite(scores).all(): raise ValueError('Nonfinite scores')
    groups={k:g.set_index('item') for k,g in history.groupby(KEY,sort=False)}
    fitkeys=queries.loc[queries.role=='fit',KEY]
    fit_history=history.merge(fitkeys,on=KEY,how='inner',validate='many_to_one')
    pops=fit_history.groupby(['code_module','code_presentation','item']).clicks.sum()
    # Only fit learners contribute to popularity, never held/selection learners.
    items=data.item.to_numpy(); labels=data.label.to_numpy(); kinds=pd.Categorical(data.kind).codes
    todo=queries.loc[queries.role.isin(['fit','selection','held'])]
    rows=list(todo.itertuples(index=False)); start=time.perf_counter(); processed=0
    preference=data.preference.to_numpy()
    def history_signals(q, ids):
        h = groups.get((q.id_student, q.code_module, q.code_presentation))
        last = np.full(len(ids), -1e9)
        counts = np.zeros(len(ids))
        observed_days = np.zeros(len(ids), dtype=int)
        recent = np.zeros(len(ids))
        if h is not None:
            joined = h.reindex(ids)
            last = joined.last_date.fillna(-1e9).to_numpy()
            counts = joined.clicks.fillna(0).to_numpy()
            observed_days = joined.resource_days.fillna(0).to_numpy()
            if improve:
                recent = joined.recent_visits.fillna(0).to_numpy()
        pop = np.array([pops.get((q.code_module, q.code_presentation, int(i)), 0.) for i in ids])
        return last, counts, observed_days, recent, pop

    def improved_rank(ids, sl, seen, last, recent, pop, recency, discovery, k):
        mixed = mixed_scores(ids, scores[sl], seen, day-last, recent, pop, recency, discovery)
        return mixed, ids[np.lexsort((ids, -mixed))][:k].astype(int).tolist()

    if improve:
        tuning = directory/'ranking_tuning'
        tuning.mkdir(exist_ok=True)
        if not ready(tuning, identity):
            totals = {(r, n): np.zeros(2) for r in MIX_WEIGHTS for n in MIX_WEIGHTS}
            selected = list(queries.loc[queries.role=='selection'].itertuples(index=False))
            if not selected:
                raise ValueError('Empty selection split')
            started = time.perf_counter()
            with stage(f'day={day} seed={seed} selection-only ranking weights'):
                for i, q in enumerate(selected, 1):
                    sl = slice(q.start, q.stop); ids = items[sl]; seen = days[sl]>0
                    last, _, observed, recent, pop = history_signals(q, ids)
                    if not np.array_equal(observed, days[sl]):
                        raise ValueError(f'Historical alignment mismatch query={q.query}')
                    truth = set(ids[labels[sl]==1])
                    new_truth = truth - set(ids[seen])
                    for (r,n), total in totals.items():
                        _, ranked = improved_rank(ids,sl,seen,last,recent,pop,r,n,10)
                        total[0] += ranking_metrics(ranked,truth,10)['ndcg']
                        if new_truth:
                            total[1] += len(new_truth.intersection(ranked))/len(new_truth)
                    if i % 256 == 0 or i == len(selected):
                        elapsed = time.perf_counter()-started
                        print(f'day={day} seed={seed} tuning {i}/{len(selected)}; {i/max(elapsed,1e-9):.1f} queries/s; stage ETA={elapsed/i*(len(selected)-i):.0f}s',flush=True)
            recency = max(MIX_WEIGHTS, key=lambda r: totals[r,0.][0])
            # Allow at most 0.002 selection NDCG loss for the discovery gain.
            eligible = [n for n in MIX_WEIGHTS if totals[recency,n][0] >= totals[recency,0.][0]-.002*len(selected)]
            discovery = max(eligible, key=lambda n: (totals[recency,n][1], totals[recency,n][0]))
            atomic_json(tuning/'weights.json', {'recency':recency,'popularity':discovery,'selection_queries':len(selected),
                'grid':[{'recency':r,'popularity':n,'ndcg10':float(v[0]/len(selected)),
                         'new_recall_sum':float(v[1])} for (r,n),v in totals.items()]})
            checkpoint(tuning,identity)
        weights=json.loads((tuning/'weights.json').read_text(encoding='utf-8'))
        recency, discovery=weights['recency'],weights['popularity']
        print(f'day={day} seed={seed} selected recency={recency:g} popularity={discovery:g}',flush=True)
    for offset in range(0,len(rows),256):
        chunk=directory/f'chunk_{offset:06d}'; chunk.mkdir(exist_ok=True)
        if ready(chunk,identity): continue
        records=[]
        for q in rows[offset:offset+256]:
            sl=slice(q.start,q.stop); ids=items[sl]; seen_mask=days[sl]>0
            last, counts, observed_days, recent, pop=history_signals(q,ids)
            if not np.array_equal(observed_days,days[sl]): raise ValueError(f'Historical alignment mismatch query={q.query}')
            scoremap=dict(zip(ids.astype(int),scores[sl])); seen=set(ids[seen_mask].astype(int)); truth=set(ids[labels[sl]==1].astype(int))
            # Same within-group ordering for ALL quota variants; MMR disabled.
            if improve:
                mixed, improved_free=improved_rank(ids,sl,seen_mask,last,recent,pop,recency,discovery,max(ks))
                scoremap=dict(zip(ids.astype(int),mixed))
                ranked_ids=ids[np.lexsort((ids,-mixed))]
                repeat=[int(item) for item in ranked_ids if item in seen][:max(ks)]
                new=[int(item) for item in ranked_ids if item not in seen][:max(ks)]
            else:
                repeat=compact_rank(ids[seen_mask],kinds[sl][seen_mask],scores[sl][seen_mask],affinity[sl][seen_mask],days[sl][seen_mask],max(ks),1.,settings['sasrec_weight'])
                new=compact_rank(ids[~seen_mask],kinds[sl][~seen_mask],scores[sl][~seen_mask],affinity[sl][~seen_mask],days[sl][~seen_mask],max(ks),1.,settings['sasrec_weight'])
            feat=history_features(days[sl],preference[sl],x[sl,names.index('preference_confidence')],last>=day-7,day)
            record={'query':int(q.query),'id_student':int(q.id_student),'role':q.role,
                    'code_module':q.code_module,'code_presentation':q.code_presentation,
                    'features':feat.tolist(),'targets':[],'evaluations':{}}
            for fraction in fractions:
                key=f'fixed_{fraction:g}'; evaluation={}
                for k in ks:
                    ranked=quota_rank(repeat,new,fraction,k,scoremap)
                    if k==10: record['targets'].append(ranking_metrics(ranked,truth,k)['ndcg'])
                    if q.role=='held': evaluation[str(k)]={**split_metrics(ranked,truth,seen,k),'items':ranked,'new_items':[i for i in ranked if i not in seen]}
                if q.role=='held': record['evaluations'][key]=evaluation
            if q.role=='held':
                orders={'free_hybrid':compact_rank(ids,kinds[sl],scores[sl],affinity[sl],days[sl],max(ks),1.,settings['sasrec_weight']),
                        'original_mmr_hybrid':compact_rank(ids,kinds[sl],scores[sl],affinity[sl],days[sl],max(ks),settings['mmr_lambda'],settings['sasrec_weight']),
                        'last_visited':ids[np.lexsort((ids,-pop,-last))][:max(ks)].astype(int).tolist(),
                        'most_frequent':ids[np.lexsort((ids,-pop,-counts))][:max(ks)].astype(int).tolist(),
                        'popularity':ids[np.lexsort((ids,-pop))][:max(ks)].astype(int).tolist()}
                if improve:
                    orders['improved_free']=improved_free
                    for name,r in [('score_only',0.),('recency_low',.25),('recency_high',.75),('recency_selected',recency)]:
                        orders[name]=improved_rank(ids,sl,seen_mask,last,recent,pop,r,0.,max(ks))[1]
                for method,ranked in orders.items():
                    record['evaluations'][method]={str(k):{**split_metrics(ranked,truth,seen,k),'items':ranked[:k],'new_items':[i for i in ranked[:k] if i not in seen]} for k in ks}
                record['catalog']=ids.astype(int).tolist(); record['new_catalog']=ids[~seen_mask].astype(int).tolist()
            records.append(record)
        # JSON uses null for undefined group metrics; no fabricated zero-support scores.
        clean=json.loads(json.dumps(records,allow_nan=True).replace('NaN','null'))
        atomic_json(chunk/'records.json',clean); checkpoint(chunk,identity)
        processed+=len(records); done=min(offset+256,len(rows)); elapsed=time.perf_counter()-start
        rate=processed/max(elapsed,1e-9)
        print(f'day={day} seed={seed} prepare {done}/{len(rows)} queries; {rate:.1f} new queries/s; stage ETA={(len(rows)-done)/max(rate,1e-9):.0f}s',flush=True)
    complete.mkdir(exist_ok=True); atomic_json(complete/'done.json',{'queries':len(rows)})
    checkpoint(complete,identity)


def evaluate_job(directory, fractions, ks, bootstrap, seed, day, identity):
    out=directory/'evaluation'; out.mkdir(exist_ok=True)
    if ready(out,identity): print(f'day={day} seed={seed}: reuse evaluation',flush=True); return
    with stage(f'day={day} seed={seed} load prepared query records'):
        records=[]
        for p in sorted(directory.glob('chunk_*/records.json')):
            if not ready(p.parent,identity): raise ValueError('Unverified query chunk')
            records.extend(json.loads(p.read_text(encoding='utf-8')))
    fit=[r for r in records if r['role']=='fit']; selection=[r for r in records if r['role']=='selection']; held=[r for r in records if r['role']=='held']
    if not all([fit,selection,held]): raise ValueError('Empty fit/selection/held role')
    xf=np.asarray([r['features'] for r in fit]); yf=np.asarray([r['targets'] for r in fit])
    xs=np.asarray([r['features'] for r in selection]); ys=np.asarray([r['targets'] for r in selection])
    fixed_index=int(np.argmax(ys.mean(axis=0))); search=[]; best=None
    with stage(f'day={day} seed={seed} fit quota policy CPU; fit only / select on selection'):
        for alpha in [.1,1.,10.,100.]:
            model=make_pipeline(StandardScaler(),Ridge(alpha=alpha))
            model.fit(xf,yf); choices=np.argmax(model.predict(xs),axis=1)
            value=float(ys[np.arange(len(ys)),choices].mean())
            search.append({'alpha':alpha,'selection_ndcg10':value})
            print(f'day={day} seed={seed} policy alpha={alpha:g}; selection NDCG10={value:.6f}',flush=True)
            if best is None or value>best[0]: best=(value,alpha,model)
    joblib.dump(best[2],out/'quota_policy.joblib')
    atomic_json(out/'selection.json',{'features':FEATURES,'best_alpha':best[1],'search':search,
                'fixed_repeat_fraction':fractions[fixed_index],'fixed_selection_ndcg10':float(ys[:,fixed_index].mean()),
                'adaptive_selection_ndcg10':best[0], 'objective':'NDCG@10 on full future relevance',
                'tie_break':'smaller fraction / smaller alpha', 'held_used_for_selection':False})
    choices=np.argmax(best[2].predict(np.asarray([r['features'] for r in held])),axis=1)
    rows=[]; catalogs={}; newcatalogs={}
    for record,choice in zip(held,choices):
        context=(record['code_module'],record['code_presentation'])
        catalogs.setdefault(context,set()).update(record['catalog']); newcatalogs.setdefault(context,set()).update(record['new_catalog'])
        methods=dict(record['evaluations'])
        methods['best_fixed']=methods[f'fixed_{fractions[fixed_index]:g}']
        methods['adaptive']=methods[f'fixed_{fractions[int(choice)]:g}']
        for method,evaluation in methods.items():
            for k in ks:
                rows.append({'day':day,'seed':seed,'query':record['query'],'id_student':record['id_student'],
                             'code_module':context[0],'code_presentation':context[1],'method':method,'k':k,
                             'selected_repeat_fraction':fractions[int(choice)] if method=='adaptive' else fractions[fixed_index] if method=='best_fixed' else float(method[6:]) if method.startswith('fixed_') else np.nan,
                             **evaluation[str(k)]})
    frame=pd.DataFrame(rows)
    # Keep per-query recommendations for audit and coverage recomputation.
    frame.to_parquet(out/'query_metrics.parquet',index=False)
    summaries=[]; differences=[]
    metrics=['precision','recall','ndcg','map','mrr','hit_rate','fill_rate','repeat_recall','new_recall','repeat_ndcg','new_ndcg','repeat_share','new_share']
    with stage(f'day={day} seed={seed} aggregation and learner bootstrap'):
        for (method,k),part in frame.groupby(['method','k'],sort=True):
            rec=set(); newrec=set()
            for row in part.itertuples(index=False):
                prefix=(row.code_module,row.code_presentation)
                rec.update((*prefix,int(i)) for i in row.items); newrec.update((*prefix,int(i)) for i in row.new_items)
            denominator=sum(len(s) for s in catalogs.values()); newden=sum(len(s) for s in newcatalogs.values())
            ci=cluster_interval(part.ndcg.to_numpy(),part.id_student.to_numpy(),repeats=bootstrap,seed=seed)
            summaries.append({'day':day,'seed':seed,'method':method,'k':int(k),'queries':len(part),
                **{m:float(part[m].mean()) if part[m].notna().any() else np.nan for m in metrics},
                'repeat_supported_queries':int(part.repeat_support.sum()),'new_supported_queries':int(part.new_support.sum()),
                'catalog_coverage':len(rec)/denominator if denominator else 0.,
                'new_catalog_coverage':len(newrec)/newden if newden else 0.,
                'ndcg_ci_low':ci['low'],'ndcg_ci_high':ci['high']})
            print(f'day={day} seed={seed} summarize {method} K={k}; NDCG={part.ndcg.mean():.6f}',flush=True)
        pairs=[('adaptive', method) for method in ['best_fixed','free_hybrid',*BASELINES]]
        if 'improved_free' in set(frame.method):
            pairs.extend([('recency_selected','last_visited'),('recency_selected','score_only'),
                          ('recency_high','recency_low'),('improved_free','recency_selected'),
                          ('improved_free','last_visited'),('improved_free','free_hybrid'),
                          ('adaptive','improved_free')])
        for k in ks:
            for treatment,method in pairs:
                adaptive=frame.loc[(frame.method==treatment)&(frame.k==k)].set_index('query').sort_index()
                other=frame.loc[(frame.method==method)&(frame.k==k)].set_index('query').sort_index()
                if not adaptive.index.equals(other.index) or not adaptive.id_student.equals(other.id_student): raise ValueError('Paired query mismatch')
                for metric in ['ndcg','recall','new_recall','repeat_recall']:
                    valid=adaptive[metric].notna() & other[metric].notna()
                    if not valid.any(): continue
                    interval=cluster_interval((adaptive.loc[valid,metric]-other.loc[valid,metric]).to_numpy(),adaptive.loc[valid,'id_student'].to_numpy(),repeats=bootstrap,seed=seed)
                    differences.append({'day':day,'seed':seed,'k':k,'metric':metric,'contrast':f'{treatment} minus {method}',**interval})
    atomic_csv(out/'summary.csv',pd.DataFrame(summaries)); atomic_csv(out/'paired_differences.csv',pd.DataFrame(differences))
    checkpoint(out,identity)


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--agent-run',type=Path,required=True)
    p.add_argument('--windows',type=int,nargs='+',default=[7,14,28,56])
    p.add_argument('--seeds',type=int,nargs='+',default=[42,13,7,23,101])
    p.add_argument('--fractions',type=float,nargs='+',default=[0.,.25,.5,.75,1.])
    p.add_argument('--ks',type=int,nargs='+',default=[5,10,20])
    p.add_argument('--bootstrap',type=int,default=1000); p.add_argument('--threads',type=int,default=6)
    p.add_argument('--inference-batch-size',type=int,default=65536)
    p.add_argument('--output',type=Path); p.add_argument('--resume',type=Path)
    p.add_argument('--improved-ranking',action='store_true')
    p.add_argument('--preflight-only',action='store_true',help='Verify files, hashes, splits and bundle versions only; no experiment')
    a=p.parse_args(argv)
    if a.output and a.resume: p.error('Use output OR resume')
    a.windows=sorted(set(a.windows)); a.seeds=sorted(set(a.seeds)); a.fractions=sorted(set(a.fractions)); a.ks=sorted(set(a.ks))
    if not a.windows or min(a.windows)<1 or 10 not in a.ks or min(a.ks+[a.bootstrap,a.threads,a.inference_batch_size])<1 or len(a.fractions)<2 or min(a.fractions)<0 or max(a.fractions)>1:
        p.error('Positive limits, K=10 and at least two fractions in [0,1] required')
    return a


def preflight(args):
    source=args.agent_run.resolve(); manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('external_test_used') or manifest.get('pilot_not_for_publication') or manifest.get('max_learners_per_split'):
        raise ValueError('Require a full 2013-only frozen source')
    if manifest['horizon']!=7: raise ValueError('Expected frozen seven-day future horizon')
    if not set(args.windows)<=set(manifest['windows']) or not set(args.seeds)<=set(manifest['seeds']): raise ValueError('Unavailable windows/seeds')
    output=args.resume or args.output
    if output:
        output=output.resolve()
        if output==source or output.is_relative_to(source) or source.is_relative_to(output): raise ValueError('Separate output required')
    raw=Path(manifest['raw_source'])/'studentVle.csv'
    paths=[source/'manifest.json',raw]
    for day in args.windows:
        queries=pd.read_csv(source/f'day_{day}/queries.csv')
        if not queries.code_presentation.isin(['2013B','2013J']).all() or not set(queries.role)<=set(['fit','selection','calibration','held']): raise ValueError('Test records rejected')
        if (queries.groupby('id_student').role.nunique()>1).any(): raise ValueError('Learner split leakage')
        for seed in args.seeds:
            files=source_paths(source,day,seed); paths.extend(files)
            settings=json.loads(files[5].read_text(encoding='utf-8'))
            if sha(files[4])!=settings['artifacts_sha256']['ranking.joblib']: raise ValueError('Ranking checksum mismatch')
            bundle=joblib.load(files[4]); x=np.load(files[2],mmap_mode='r',allow_pickle=False)
            if bundle['feature_names']!=settings['feature_names'] or x.shape[1]!=len(bundle['feature_names']): raise ValueError('Frozen model schema mismatch')
            if bundle['candidate_model'] is not None:
                prediction=bundle['candidate_model'].predict_proba(x[:16])
                if not np.isfinite(prediction).all(): raise ValueError('Invalid frozen inference')
    # Mutable top-level manifest is NOT an identity dependency: requested complete
    # source jobs may coexist with an unrelated continuing source run.
    paths=list(dict.fromkeys(paths[1:]))
    hashes={}
    for i,path in enumerate(paths,1):
        hashes[str(path)]=sha(path)
        if i%10==0 or i==len(paths): print(f'preflight hashes {i}/{len(paths)} files',flush=True)
    if hashes[str(raw)]!=manifest.get('raw_sha256',{}).get('studentVle.csv'): raise ValueError('Raw journal checksum mismatch')
    print(f'preflight passed: {len(args.windows)} windows x {len(args.seeds)} seeds; test_2014_used=False',flush=True)
    return source,raw,hashes


def main(argv=None):
    args=parse_args(argv)
    if args.improved_ranking and not (args.output or args.resume or args.preflight_only):
        raise ValueError('Specify a separate output directory for improved ranking')
    # Before the session, only metadata/hash validation; timestamp explicitly.
    print(datetime.now(TZ).strftime('%H:%M')+' Preflight: validating source; no test evaluation',flush=True)
    if args.preflight_only:
        # Timestamp every line even during read-only preflight.
        from run_oulad_agent_experiment import _Tee
        from contextlib import redirect_stdout
        import sys
        with io.StringIO() as sink, redirect_stdout(_Tee(sys.stdout,sink)),stage('preflight'):
            preflight(args)
        return
    source=args.agent_run.resolve()
    config={'source':str(source),'windows':args.windows,'seeds':args.seeds,'fractions':args.fractions,'ks':args.ks,
            'improved_ranking':args.improved_ranking,
            'bootstrap':args.bootstrap,'threads':args.threads,'inference_batch_size':args.inference_batch_size,
            'test_2014_used':False,'exploratory_validation':True,'device':'CPU; frozen GPU model outputs reused',
            'protocol':'fit quota utility Ridge on fit; alpha and best fixed fraction on selection; held diagnostics only',
            'quota_backfill':True,'mmr_for_quota':False,'policy_features':FEATURES,
            'intervals':'unadjusted learner-cluster bootstrap; five seeds use the same split',
            'dependency_hashes':{name:sha(Path(__file__).parent/name) for name in ['recommendations/full_agent_experiment.py','recommendations/experiment_metrics.py','run_oulad_agent_experiment.py']}}
    with session('repeat_discovery',config,__file__,args.output,args.resume) as (output,identity),threadpool_limits(limits=args.threads):
        with stage('preflight frozen input hashing'):
            source,raw,hashes=preflight(args)
        manifest_path=output/'inputs.json'
        if manifest_path.exists() and json.loads(manifest_path.read_text(encoding='utf-8'))!=hashes: raise ValueError('Frozen inputs changed since checkpoint')
        atomic_json(manifest_path,hashes)
        identity={**identity,'inputs':hashes}
        atomic_json(output/'protocol.json',{'started_or_resumed_local':datetime.now(TZ).isoformat(),
          'target':'observed future resource interactions; not learning gain','cutoff':'date < day; seven-day future labels reused',
          'new_definition':'no positive resource-day before cutoff','popularity':'fit learners only; pre-cutoff clicks by course-presentation',
          'last_visited':'last pre-cutoff positive interaction; ties by fit popularity then item id',
          'most_frequent':'pre-cutoff click sum; ties by fit popularity then item id',
          'quotas':'round-half-up per K; shortages backfilled, then frozen hybrid score order',
          'group_metrics':'original list positions; group recall/NDCG undefined without future group positives; support counts reported',
          'coverage':'unique (module,presentation,item) recommended / union of available held catalogs',
          'selection':'primary comparison adaptive vs selection-best fixed, NDCG@10; all other comparisons exploratory',
          'recovery':'history per day; preparation per 256 queries; evaluation per day/seed; failed evaluation reruns only evaluation',
          'raw_access':'mixed studentVle journal scanned; only 2013 pre-cutoff positive lines parsed; no final-result/test files',
          'test_2014_used':False})
        for day in args.windows:
            history_dir=output/f'history_day_{day}'; history_dir.mkdir(exist_ok=True)
            if ready(history_dir,identity): history=pd.read_parquet(history_dir/'history.parquet')
            else:
                with stage(f'day={day} reconstruct historical 2013 interactions'):
                    queries=pd.read_csv(source/f'day_{day}/queries.csv'); history=build_history(raw,queries,day)
                    history.to_parquet(history_dir/'history.parquet',index=False)
                    checkpoint(history_dir,identity)
            for seed in args.seeds:
                job=output/f'day_{day}_seed_{seed}'; job.mkdir(exist_ok=True)
                with ResourceMonitor() as monitor:
                    prepare_job(source,day,seed,history,args.fractions,args.ks,job,identity,args.inference_batch_size,args.improved_ranking)
                    evaluate_job(job,args.fractions,args.ks,args.bootstrap,seed,day,identity)
                atomic_json(job/'runtime_memory.json',monitor.result)
                summaries=[]; contrasts=[]
                for p in sorted(output.glob('day_*_seed_*/evaluation')):
                    if ready(p,identity):
                        summaries.append(pd.read_csv(p/'summary.csv')); contrasts.append(pd.read_csv(p/'paired_differences.csv'))
                merged=pd.concat(summaries,ignore_index=True); atomic_csv(output/'summary.csv',merged)
                atomic_csv(output/'paired_differences.csv',pd.concat(contrasts,ignore_index=True))
                aggregate=merged.groupby(['day','method','k']).agg(seeds=('seed','nunique'),ndcg_mean=('ndcg','mean'),ndcg_sd=('ndcg','std'),new_recall_mean=('new_recall','mean'),repeat_recall_mean=('repeat_recall','mean'),new_share_mean=('new_share','mean'),coverage_mean=('catalog_coverage','mean')).reset_index()
                atomic_csv(output/'across_seed_summary.csv',aggregate)
                atomic_json(output/'progress.json',{'local_time':datetime.now(TZ).isoformat(),'completed_jobs':len(summaries),'total_jobs':len(args.windows)*len(args.seeds),'last_day':day,'last_seed':seed})


if __name__=='__main__':
    main()
