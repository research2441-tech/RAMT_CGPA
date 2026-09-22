from __future__ import annotations
import argparse, json, platform, sys, time
from pathlib import Path
import yaml, numpy as np, pandas as pd
from data_prepare import prepare
from train_evaluate import run as run_predictive
from security_benchmark import run as run_security
from statistical_analysis import analyze

def main(config="config.yaml", demo=False):
    started=time.time()
    cfg=yaml.safe_load(open(config,encoding="utf-8"))
    stages=[]

    t=time.time(); _,meta=prepare(config,force_demo=demo); stages.append({"stage":"data_prepare","seconds":time.time()-t})

    t=time.time(); pred,base=run_predictive(config,demo=False); stages.append({"stage":"predictive","seconds":time.time()-t})

    t=time.time(); sec,dag=run_security(config); stages.append({"stage":"security","seconds":time.time()-t})

    stcfg=cfg.get("statistics",{})
    t=time.time(); stats=analyze("predictive_results.csv","baseline_results.csv",
        float(stcfg.get("confidence",.95)),int(stcfg.get("bootstrap_iterations",2000)),float(stcfg.get("alpha",.05)))
    stages.append({"stage":"statistics","seconds":time.time()-t})

    summary={
      "framework":"RAMT-CGPA",
      "data":meta,
      "predictive_mean":pred[["accuracy","macro_precision","macro_recall","macro_f1","pr_auc","inference_ms_per_sample"]].mean().to_dict(),
      "security_operations":sec[["operation","mean_ms","p95_ms","p99_ms"]].to_dict("records"),
      "dag_scalability":dag.to_dict("records"),
      "runtime_seconds":time.time()-started,
      "stages":stages,
      "environment":{"python":sys.version,"platform":platform.platform()},
      "notes":[
        "Reported values are generated from execution; manuscript values are not hard-coded.",
        "Post-quantum measurements identify whether liboqs or the benchmark fallback was used.",
        "Research break-glass proof interface is not a production zk-STARK implementation."
      ]
    }
    Path("reproducibility_summary.json").write_text(json.dumps(summary,indent=2,default=float),encoding="utf-8")
    print(json.dumps(summary,indent=2,default=float))

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",default="config.yaml")
    ap.add_argument("--demo",action="store_true")
    a=ap.parse_args(); main(a.config,a.demo)
