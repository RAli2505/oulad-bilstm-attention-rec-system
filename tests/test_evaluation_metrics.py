from evaluation_metrics import risk_metrics, early_warning_recall, bootstrap_auc_delta, ranking_extras, explanation_metrics
def test_extra_metrics():
    y,p=[0,0,1,1],[.1,.2,.8,.9]
    assert risk_metrics(y,p)["pr_auc"] == 1
    assert early_warning_recall([1,14,28,7],p)["early_warning_recall_7d"] >= 0
    assert bootstrap_auc_delta(y,[.7,.6,.4,.3],p,20)["auc_delta"] > 0
    assert ranking_extras([1,0,1],20,[.5,.5,.5])["precision_at_10"] == 2/3
    assert explanation_metrics("fact",["fact"])["grounded_fact_recall"] == 1
