"""Protocol tests for quota allocation, leakage barriers and verified recovery."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from article_experiment_utils import atomic_json, ready, sha
from run_repeat_discovery_experiment import (
    quota_rank, split_metrics, validate_queries, build_history,
    prepare_job, evaluate_job, history_features,
)


def test_quotas_backfill_ties_and_per_k():
    score={1:.9,2:.8,3:.7,4:.6,5:.5}
    assert quota_rank([1,2],[3,4,5],.5,4,score)==[1,2,3,4]
    assert quota_rank([1],[2,3,4,5],1.,4,score)==[1,2,3,4]
    assert quota_rank([],[],.5,10,{})==[]
    # 25% of ten is rounded up to three repeat slots, not truncated to two.
    s={i:100-i for i in range(20)}
    ranked=quota_rank(list(range(10)),list(range(10,20)),.25,10,s)
    assert sum(i<10 for i in ranked)==3
    assert len(ranked)==len(set(ranked))


def test_group_metrics_keep_positions_and_exclude_zero_support():
    result=split_metrics([10,20],{20},{10},2)
    assert result['new_recall']==1
    assert result['new_ndcg']==pytest.approx(1/np.log2(3))
    assert result['repeat_support']==0 and np.isnan(result['repeat_recall'])
    assert result['new_share']==.5


@pytest.mark.parametrize('quoted',[False,True])
def test_history_excludes_cutoff_future_and_2014_before_parsing(tmp_path,quoted):
    import csv
    path=tmp_path/'studentVle.csv'
    path.write_text('id_student,code_module,code_presentation,id_site,date,sum_click\n'
                    '1,AAA,2013J,10,6,2\n1,AAA,2013J,10,6,3\n'
                    '1,AAA,2013J,10,7,100\n1,AAA,2013J,10,30,100\n'
                    '1,AAA,2014J,10,NOT_A_DATE,NOT_A_COUNT\n',encoding='utf-8')
    if quoted:
        rows=list(csv.reader(path.read_text(encoding='utf-8').splitlines()))
        with path.open('w',encoding='utf-8',newline='') as stream:
            csv.writer(stream,quoting=csv.QUOTE_ALL).writerows(rows)
    q=pd.DataFrame([{'id_student':1,'code_module':'AAA','code_presentation':'2013J'}])
    h=build_history(path,q,7)
    assert len(h)==1 and h.iloc[0].clicks==5 and h.iloc[0].resource_days==1
    assert h.iloc[0].last_date==6


def fixture_source(tmp_path):
    source=tmp_path/'source'; base=source/'day_7'; job=base/'seed_42'; bundle=job/'agent_bundle'
    bundle.mkdir(parents=True)
    queries=[]; data=[]; history=[]
    roles=['fit']*4+['selection']*2+['held']*2
    for query,role in enumerate(roles):
        student=query+1
        queries.append(dict(query=query,id_student=student,code_module='AAA',code_presentation='2013J',role=role,start=query*4,stop=query*4+4))
        for item in range(4):
            data.append(dict(query=query,item=item,kind='forum' if item%2 else 'resource',label=int(item==(query%4)),preference=.2+item/10))
        history.append(dict(id_student=student,code_module='AAA',code_presentation='2013J',item=0,last_date=6,clicks=2,resource_days=1))
    q=pd.DataFrame(queries); d=pd.DataFrame(data)
    q.to_csv(base/'queries.csv',index=False); d.to_parquet(base/'candidate_features_and_labels.parquet',index=False)
    x=np.array([[r['preference'],.5] for r in data]); np.save(job/'full_candidate_features.npy',x)
    model=LogisticRegression().fit(x,np.tile([1,0,0,0],8))
    joblib.dump({'feature_names':['preference','preference_confidence'],'candidate_model':model},bundle/'ranking.joblib')
    atomic_json(bundle/'bundle.json',{'feature_names':['preference','preference_confidence'],'sasrec_weight':.05,'mmr_lambda':.8,'artifacts_sha256':{'ranking.joblib':sha(bundle/'ranking.joblib')}})
    np.savez(job/'candidate_utilities.npz',history_resource_days=np.tile([1,0,0,0],8),sasrec_affinity=np.tile([.3,.2,.1,0],8),fallback=np.ones(32),labels=d.label.to_numpy())
    atomic_json(job/'runtime_memory.json',{})
    return source,pd.DataFrame(history),q,d


def test_test_barrier_and_learner_independence(tmp_path):
    _,_,q,d=fixture_source(tmp_path)
    validate_queries(q,d)
    changed=q.copy(); changed.loc[0,'code_presentation']='2014J'
    with pytest.raises(ValueError,match='2013'):validate_queries(changed,d)
    changed=q.copy(); changed.loc[7,'id_student']=1
    with pytest.raises(ValueError,match='Learner'):validate_queries(changed,d)


@pytest.mark.parametrize('improve',[False,True])
def test_end_to_end_selection_recovery_and_checksums(tmp_path,improve):
    source,history,_,_=fixture_source(tmp_path)
    history['recent_visits']=1
    job=tmp_path/'output'; identity={'synthetic':True}; fractions=[0.,.5,1.]; ks=[5,10,20]
    prepare_job(source,7,42,history,fractions,ks,job,identity,8,improve)
    evaluate_job(job,fractions,ks,20,42,7,identity)
    summary=pd.read_csv(job/'evaluation/summary.csv')
    assert {'adaptive','best_fixed','last_visited','most_frequent','popularity'}<=set(summary.method)
    assert summary.queries.eq(2).all()
    if improve:
        assert {'improved_free','recency_low','recency_high','score_only'}<=set(summary.method)
        weights=json.loads((job/'ranking_tuning/weights.json').read_text())
        assert len(weights['grid'])==25
        assert weights['selection_queries']==2
    selection=json.loads((job/'evaluation/selection.json').read_text())
    assert not selection['held_used_for_selection']
    assert ready(job/'evaluation',identity)
    before=sha(job/'evaluation/summary.csv')
    prepare_job(source,7,42,history,fractions,ks,job,identity,8,improve)
    evaluate_job(job,fractions,ks,20,42,7,identity)
    assert sha(job/'evaluation/summary.csv')==before
    # Corruption is rejected, never silently reused.
    (job/'evaluation/summary.csv').write_text('corrupt',encoding='utf-8')
    with pytest.raises(ValueError,match='checksum'):ready(job/'evaluation',identity)


def test_policy_features_do_not_take_future_labels():
    a=history_features(np.array([1,0]),np.array([.2,.3]),np.array([.5,.5]),np.array([True,False]),7)
    assert np.isfinite(a).all() and len(a)==7
