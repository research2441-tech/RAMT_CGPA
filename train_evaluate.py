from __future__ import annotations
import argparse, json, time
import numpy as np, pandas as pd, yaml
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,average_precision_score
from sklearn.preprocessing import label_binarize, StandardScaler
from cait_qim import assign_tiers
from sfrs import select_features
from mhkan import train_mhkan, embed
from ajs_xgboost import optimize
from baselines import evaluate_models
from data_prepare import prepare

PII_COLS={"national_identity_hash","full_name","contact_number","street_address","zip_code",
          "date_of_birth","admission_timestamp","medical_record_number","source_cohort"}

def _metrics(y,pred,prob):
    p,r,f,_=precision_recall_fscore_support(y,pred,average="macro",zero_division=0)
    try:
        Y=label_binarize(y,classes=np.arange(prob.shape[1]))
        ap=average_precision_score(Y,prob,average="macro")
    except Exception: ap=np.nan
    return {"accuracy":accuracy_score(y,pred),"macro_precision":p,"macro_recall":r,"macro_f1":f,"pr_auc":ap}

def run(cfg_path="config.yaml", demo=False):
    cfg=yaml.safe_load(open(cfg_path,encoding="utf-8"))
    seed=int(cfg.get("seed",42))
    if not Path("prepared_clinical.csv").exists() or demo:
        prepare(cfg_path, force_demo=demo)
    df=pd.read_csv("prepared_clinical.csv")
    tiers=assign_tiers(df,"target",cfg); tiers.to_csv("tier_assignments.csv",index=False)

    feature_cols=[c for c in df.columns if c not in PII_COLS|{"target"} and pd.api.types.is_numeric_dtype(df[c])]
    X=df[feature_cols].astype(float); y=df["target"].astype(int)
    skf=StratifiedKFold(n_splits=int(cfg["data"].get("cv_folds",10)),shuffle=True,random_state=seed)
    fold_rows=[]; baseline_rows=[]; selected_all={}
    for fold,(tr_idx,te_idx) in enumerate(skf.split(X,y),1):
        Xtr0,Xte0=X.iloc[tr_idx].copy(),X.iloc[te_idx].copy()
        ytr0,yte=y.iloc[tr_idx].to_numpy(),y.iloc[te_idx].to_numpy()
        Xtr,Xv,ytr,yv=train_test_split(Xtr0,ytr0,test_size=float(cfg["data"].get("validation_size",.1)),
                                       stratify=ytr0,random_state=seed+fold)
        # Fit fold-specific scaler to avoid leakage.
        sc=StandardScaler().fit(Xtr)
        Xtr=pd.DataFrame(sc.transform(Xtr),columns=feature_cols,index=Xtr.index)
        Xv=pd.DataFrame(sc.transform(Xv),columns=feature_cols,index=Xv.index)
        Xte=pd.DataFrame(sc.transform(Xte0),columns=feature_cols,index=Xte0.index)

        selected,hist=select_features(Xtr,pd.Series(ytr,index=Xtr.index),cfg,seed+fold)
        selected_all[str(fold)]={"features":selected,"history":hist.to_dict("records")}
        A,B,C=Xtr[selected].to_numpy(),Xv[selected].to_numpy(),Xte[selected].to_numpy()

        if cfg.get("mhkan",{}).get("enabled",True):
            mh=train_mhkan(A,ytr,B,yv,cfg,len(np.unique(y)))
            A2,B2,C2=embed(mh,A),embed(mh,B),embed(mh,C)
        else:
            A2,B2,C2=A,B,C

        model,best,hist2=optimize(A2,ytr,B2,yv,cfg)
        t=time.perf_counter(); prob=model.predict_proba(C2); pred=prob.argmax(1)
        infer_ms=(time.perf_counter()-t)*1000/max(len(C2),1)
        fold_rows.append({"fold":fold,"model":"RAMT-CGPA","inference_ms_per_sample":infer_ms,
                          **_metrics(yte,pred,prob),**{f"xgb_{k}":v for k,v in best.items()}})

        base=evaluate_models(A,ytr,B,yv,C,yte,seed+fold)
        base["fold"]=fold
        baseline_rows.append(base)

    res=pd.DataFrame(fold_rows); res.to_csv("predictive_results.csv",index=False)
    bres=pd.concat(baseline_rows,ignore_index=True); bres.to_csv("baseline_results.csv",index=False)
    Path("selected_features.json").write_text(json.dumps(selected_all,indent=2),encoding="utf-8")
    return res,bres

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default="config.yaml"); ap.add_argument("--demo",action="store_true")
    a=ap.parse_args(); r,b=run(a.config,a.demo)
    print(r.groupby("model")[["accuracy","macro_f1","pr_auc"]].mean())
