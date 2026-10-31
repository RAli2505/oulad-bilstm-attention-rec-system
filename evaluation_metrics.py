"""Additional held-out metrics for risk, recommendations and explanations."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

def risk_metrics(y_true, probability, bins=10):
    y, p = np.asarray(y_true, int), np.asarray(probability, float)
    out = {"pr_auc": float(average_precision_score(y, p))}
    edges = np.linspace(0, 1, bins + 1); ece = mce = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if mask.any():
            gap = abs(float(y[mask].mean()) - float(p[mask].mean()))
            ece += mask.mean() * gap; mce = max(mce, gap)
    return out | {"ece": float(ece), "mce": float(mce)}

def early_warning_recall(event_days, probability, horizons=(7, 14, 28), threshold=.5):
    d, p = np.asarray(event_days), np.asarray(probability)
    return {f"early_warning_recall_{h}d": float((p[d >= h] >= threshold).mean()) if (d >= h).any() else float("nan") for h in horizons}

def intervention_metrics(y_true, probability, budgets=(.05, .10, .20)):
    """Precision/recall when staff can intervene only for the top-risk fraction."""
    y, p = np.asarray(y_true, int), np.asarray(probability, float)
    positives = max(1, int(y.sum())); out = {}
    for budget in budgets:
        n = max(1, int(np.ceil(len(y) * budget)))
        picked = np.argsort(-p)[:n]
        hits = int(y[picked].sum())
        label = f"top_{int(budget * 100)}pct"
        out[f"precision_{label}"] = hits / n
        out[f"recall_{label}"] = hits / positives
    return out

def deployable_intervention_metrics(y_true, probability, eligible, budgets=(.05, .10, .20)):
    """Top-k intervention metrics after excluding students already withdrawn at scoring time."""
    mask = np.asarray(eligible, bool)
    out = intervention_metrics(np.asarray(y_true)[mask], np.asarray(probability)[mask], budgets)
    out["n_intervention_eligible"] = int(mask.sum())
    out["n_excluded_pre_cutoff"] = int((~mask).sum())
    return out

def bootstrap_auc_delta(y, baseline, candidate, n_boot=1000, seed=42):
    y, a, b = np.asarray(y), np.asarray(baseline), np.asarray(candidate); rng = np.random.default_rng(seed); ds=[]
    for _ in range(n_boot):
        ix = rng.integers(0, len(y), len(y))
        if len(np.unique(y[ix])) > 1: ds.append(roc_auc_score(y[ix], b[ix])-roc_auc_score(y[ix], a[ix]))
    return {"auc_delta": float(roc_auc_score(y,b)-roc_auc_score(y,a)), "auc_delta_ci_low": float(np.quantile(ds,.025)), "auc_delta_ci_high": float(np.quantile(ds,.975))}

def group_auc(y, probability, groups):
    y,p,g=np.asarray(y),np.asarray(probability),np.asarray(groups)
    return {f"group_auc_{x}": float(roc_auc_score(y[g==x],p[g==x])) for x in np.unique(g) if len(np.unique(y[g==x]))>1}

def ranking_extras(ranked_relevance, catalog_size, item_popularity=None):
    rel=np.asarray(ranked_relevance); top=rel[:10]; out={"precision_at_10":float(top.mean()) if len(top) else 0.,"coverage":float(min(len(rel),10)/catalog_size) if catalog_size else 0.}
    if item_popularity is not None:
        out["novelty_at_10"]=float((-np.log2(np.clip(np.asarray(item_popularity)[:len(top)],1e-12,1))).mean()) if len(top) else 0.
    return out

def explanation_metrics(text, facts, latency_ms=0., cost_usd=0.):
    facts=list(facts); lower=text.lower()
    return {"grounded_fact_recall":sum(f.lower() in lower for f in facts)/len(facts) if facts else 1.,"explanation_words":float(len(text.split())),"latency_ms":float(latency_ms),"cost_usd":float(cost_usd)}
