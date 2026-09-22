from __future__ import annotations
import argparse, json, os, statistics, time
import numpy as np, pandas as pd, yaml, psutil
from crypto_governance import KEMWrapper, split_secret, reconstruct_secret, gaussian_ldp, merkle_root, digest
from dag_ledger import benchmark_dag
from breakglass_stark import benchmark_breakglass

def summary(values):
    a=np.asarray(values,float)
    return {
        "mean_ms":float(a.mean()),"std_ms":float(a.std(ddof=1) if len(a)>1 else 0),
        "median_ms":float(np.median(a)),"p95_ms":float(np.quantile(a,.95)),
        "p99_ms":float(np.quantile(a,.99)),"min_ms":float(a.min()),"max_ms":float(a.max())
    }

def timed(fn,*args,**kw):
    s=time.perf_counter_ns(); out=fn(*args,**kw); return out,(time.perf_counter_ns()-s)/1e6

def run(cfg_path="config.yaml"):
    cfg=yaml.safe_load(open(cfg_path,encoding="utf-8"))
    c=cfg.get("security",{}); reps=int(c.get("benchmark_repetitions",100))
    algo=c.get("kem_algorithm","Kyber1024")
    n=int(c.get("secret_shares",7)); k=int(c.get("normal_threshold",5)); ke=int(c.get("emergency_threshold",2))
    payload=os.urandom(int(c.get("payload_bytes",2048)))
    rows=[]

    kem=KEMWrapper(algo)
    vals=[]
    for _ in range(reps):
        (pk,sk),ms=timed(kem.keygen); vals.append(ms)
    rows.append({"operation":"kem_keygen","backend":kem.backend,**summary(vals)})

    pk,sk=kem.keygen()
    vals=[]; cts=[]
    for _ in range(reps):
        (ct,ss),ms=timed(kem.encapsulate,pk); vals.append(ms); cts.append(ct)
    rows.append({"operation":"kem_encapsulation","backend":kem.backend,**summary(vals)})

    vals=[]
    for ct in cts:
        _,ms=timed(kem.decapsulate,sk,ct); vals.append(ms)
    rows.append({"operation":"kem_decapsulation","backend":kem.backend,**summary(vals)})

    secret=os.urandom(32); vals=[]
    for i in range(reps):
        sh,ms=timed(split_secret,secret,n,k,42+i); vals.append(ms)
    rows.append({"operation":"avss_share_generation_proxy","backend":"finite_field_threshold",**summary(vals)})

    shares=split_secret(secret,n,k,42); vals=[]
    for _ in range(reps):
        out,ms=timed(reconstruct_secret,shares[:k],32); vals.append(ms)
        if out!=secret: raise RuntimeError("Threshold reconstruction failed")
    rows.append({"operation":"avss_reconstruction_proxy","backend":"finite_field_threshold",**summary(vals)})

    x=np.random.default_rng(42).normal(size=64); vals=[]
    for i in range(reps):
        _,ms=timed(gaussian_ldp,x,float(cfg["privacy"]["epsilon"]),float(cfg["privacy"]["delta"]),1.0,42+i); vals.append(ms)
    rows.append({"operation":"local_dp","backend":"gaussian_ldp",**summary(vals)})

    chunks=[payload[i:i+64] for i in range(0,len(payload),64)]; vals=[]
    for _ in range(reps):
        _,ms=timed(merkle_root,chunks); vals.append(ms)
    rows.append({"operation":"integrity_merkle","backend":"blake3_or_blake2b",**summary(vals)})

    bg=[]
    for _ in range(max(10,min(reps,100))):
        bg.append(benchmark_breakglass(secret,n,k,ke)["total_breakglass_ms"])
    rows.append({"operation":"breakglass_total","backend":"research_verifiable_transition",**summary(bg)})

    sec=pd.DataFrame(rows); sec.to_csv("security_benchmark.csv",index=False)

    dag_rows=[]
    dcfg=cfg.get("dag",{})
    for nodes in dcfg.get("node_counts",[4,8,16]):
        workers=max(1,min(int(nodes),int(dcfg.get("workers",8))))
        for count in dcfg.get("transaction_counts",[100,1000]):
            r=benchmark_dag(int(count),workers,int(dcfg.get("payload_bytes",96)))
            r["simulated_nodes"]=nodes; dag_rows.append(r)
    dag=pd.DataFrame(dag_rows); dag.to_csv("dag_scalability.csv",index=False)

    proc=psutil.Process(os.getpid())
    meta={"rss_bytes":proc.memory_info().rss,"cpu_count":psutil.cpu_count(),
          "kem_backend":kem.backend,"kem_algorithm":kem.algorithm}
    open("security_environment.json","w").write(json.dumps(meta,indent=2))
    return sec,dag

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default="config.yaml")
    a=ap.parse_args(); s,d=run(a.config); print(s[["operation","mean_ms","p95_ms"]].to_string(index=False))
