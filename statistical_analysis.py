from __future__ import annotations
import argparse, json
from itertools import combinations
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare

def bootstrap_ci(x, confidence=.95, n_boot=2000, seed=42):
    x=np.asarray(x,float); rng=np.random.default_rng(seed)
    if len(x)==0: return [None,None]
    means=[rng.choice(x,len(x),replace=True).mean() for _ in range(n_boot)]
    a=(1-confidence)/2
    return [float(np.quantile(means,a)),float(np.quantile(means,1-a))]

def holm(pairs, alpha=.05):
    ordered=sorted(pairs,key=lambda z:z["p_value"])
    m=len(ordered)
    for i,r in enumerate(ordered):
        r["holm_threshold"]=alpha/(m-i)
        r["reject_h0"]=bool(r["p_value"] <= r["holm_threshold"])
    return ordered

def analyze(pred="predictive_results.csv", base="baseline_results.csv", confidence=.95, boot=2000, alpha=.05):
    p=pd.read_csv(pred); b=pd.read_csv(base)
    allr=pd.concat([p,b],ignore_index=True,sort=False)
    out={"summary":{}}
    for model,g in allr.groupby("model"):
        out["summary"][model]={}
        for metric in ["accuracy","macro_precision","macro_recall","macro_f1","pr_auc","inference_ms_per_sample"]:
            if metric in g and g[metric].notna().any():
                vals=g[metric].dropna().to_numpy(float)
                out["summary"][model][metric]={
                    "mean":float(vals.mean()),"std":float(vals.std(ddof=1) if len(vals)>1 else 0),
                    "ci95":bootstrap_ci(vals,confidence,boot,42)
                }

    piv=allr.pivot_table(index="fold",columns="model",values="macro_f1",aggfunc="mean").dropna(axis=1,how="any")
    if piv.shape[1]>=3 and piv.shape[0]>=2:
        stat,pv=friedmanchisquare(*[piv[c].values for c in piv.columns])
        out["friedman_macro_f1"]={"statistic":float(stat),"p_value":float(pv),"models":list(piv.columns)}

    pair=[]
    models=list(piv.columns)
    if "RAMT-CGPA" in models:
        for other in [m for m in models if m!="RAMT-CGPA"]:
            x=piv["RAMT-CGPA"].values; y=piv[other].values
            try: st,pv=wilcoxon(x,y,zero_method="wilcox",alternative="two-sided")
            except Exception: st,pv=np.nan,1.0
            diff=x-y; effect=float(np.mean(diff)/(np.std(diff,ddof=1)+1e-12))
            pair.append({"comparison":f"RAMT-CGPA vs {other}","statistic":float(st) if np.isfinite(st) else None,
                         "p_value":float(pv),"paired_effect_d":effect})
    out["holm_pairwise_macro_f1"]=holm(pair,alpha) if pair else []
    open("statistical_results.json","w").write(json.dumps(out,indent=2))
    return out

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--input",default="predictive_results.csv")
    ap.add_argument("--baseline",default="baseline_results.csv")
    a=ap.parse_args(); print(json.dumps(analyze(a.input,a.baseline),indent=2))
