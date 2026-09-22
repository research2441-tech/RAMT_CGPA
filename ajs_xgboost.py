from __future__ import annotations
import math, random
import numpy as np
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

def _sample(bounds, rng):
    p={}
    for k,v in bounds.items():
        if isinstance(v,list) and len(v)==2:
            lo,hi=v
            val=rng.uniform(float(lo),float(hi))
            if k in {"n_estimators","max_depth","min_child_weight"}: val=int(round(val))
            p[k]=val
    return p

def _clip(p,bounds):
    q={}
    for k,v in p.items():
        lo,hi=bounds[k]
        v=max(float(lo),min(float(hi),float(v)))
        if k in {"n_estimators","max_depth","min_child_weight"}: v=int(round(v))
        q[k]=v
    return q

def _fit_score(params, Xtr,ytr,Xv,yv,seed=42):
    model=XGBClassifier(
        objective="multi:softprob",eval_metric="mlogloss",random_state=seed,n_jobs=-1,
        tree_method="hist",**params
    )
    model.fit(Xtr,ytr)
    pred=model.predict(Xv)
    return f1_score(yv,pred,average="macro"),model

def optimize(Xtr,ytr,Xv,yv,cfg):
    seed=int(cfg.get("seed",42)); rng=random.Random(seed)
    bounds=cfg.get("xgboost",{})
    popn=int(cfg.get("ajs",{}).get("population",12))
    iters=int(cfg.get("ajs",{}).get("iterations",15))
    pop=[_sample(bounds,rng) for _ in range(popn)]
    scored=[]
    for p in pop:
        s,_=_fit_score(p,Xtr,ytr,Xv,yv,seed); scored.append(s)
    best_i=int(np.argmax(scored)); best=pop[best_i].copy(); best_s=scored[best_i]
    hist=[{"iteration":0,"best_macro_f1":best_s,**best}]
    keys=list(bounds)
    for t in range(1,iters+1):
        new=[]
        for i,p in enumerate(pop):
            q=p.copy()
            beta=1.0-t/(iters+1)
            peer=pop[rng.randrange(popn)]
            for k in keys:
                lo,hi=bounds[k]; span=float(hi)-float(lo)
                cur=float(p[k]); b=float(best[k]); pr=float(peer[k])
                ocean=(b-cur)*rng.random()
                passive=span*rng.uniform(-.15,.15)*beta
                peer_move=(pr-cur)*rng.uniform(-.5,.5)
                q[k]=cur+ocean+passive+peer_move
            new.append(_clip(q,bounds))
        pop=new; scored=[]
        for p in pop:
            s,_=_fit_score(p,Xtr,ytr,Xv,yv,seed); scored.append(s)
        bi=int(np.argmax(scored))
        if scored[bi]>best_s:
            best_s=scored[bi]; best=pop[bi].copy()
        hist.append({"iteration":t,"best_macro_f1":best_s,**best})
    final=XGBClassifier(objective="multi:softprob",eval_metric="mlogloss",random_state=seed,
                        n_jobs=-1,tree_method="hist",**best)
    final.fit(np.vstack([Xtr,Xv]),np.concatenate([ytr,yv]))
    return final,best,hist
