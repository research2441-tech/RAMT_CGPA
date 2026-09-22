from __future__ import annotations
import argparse, json, os, random
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from synthetic_pii import overlay

TARGET_CANDIDATES = ["target","num","label","class","severity","diagnosis","heart_disease"]

def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _guess_target(df, configured=""):
    if configured and configured in df.columns:
        return configured
    lower = {str(c).lower(): c for c in df.columns}
    for c in TARGET_CANDIDATES:
        if c in lower:
            return lower[c]
    return df.columns[-1]

def generate_demo(cfg):
    n = int(cfg["data"].get("demo_samples", 920))
    p = int(cfg["data"].get("demo_features", 13))
    k = int(cfg["data"].get("severity_classes", 5))
    X, y = make_classification(
        n_samples=n, n_features=p, n_informative=max(6,p//2),
        n_redundant=max(2,p//5), n_classes=k, n_clusters_per_class=1,
        weights=None, class_sep=1.2, random_state=int(cfg["seed"])
    )
    cols = [f"clinical_feature_{i+1}" for i in range(p)]
    df = pd.DataFrame(X, columns=cols)
    df["target"] = y.astype(int)
    return df, "target", "synthetic_demo"

def load_primary(cfg, force_demo=False):
    path = str(cfg["data"].get("primary_csv","") or "").strip()
    if force_demo or not path or not Path(path).exists():
        return generate_demo(cfg)
    df = pd.read_csv(path)
    target = _guess_target(df, cfg["data"].get("target_column",""))
    return df, target, Path(path).stem

def normalize_target(y):
    s = pd.Series(y).copy()
    if pd.api.types.is_numeric_dtype(s):
        uniq = sorted(pd.unique(s.dropna()))
        if len(uniq) <= 5:
            mp = {v:i for i,v in enumerate(uniq)}
            return s.map(mp).astype(int)
        q = pd.qcut(s.rank(method="first"), 5, labels=False)
        return q.astype(int)
    cat = s.astype("category")
    if len(cat.cat.categories) > 5:
        raise ValueError("Target has more than five non-numeric classes.")
    return cat.cat.codes.astype(int)

def preprocess_table(df, target, cfg):
    work = df.copy()
    y = normalize_target(work.pop(target))
    id_col = str(cfg["data"].get("id_column","") or "")
    if id_col and id_col in work:
        work = work.drop(columns=[id_col])

    # Keep only clinically usable numeric features; encode categoricals deterministically.
    cat_cols = [c for c in work.columns if not pd.api.types.is_numeric_dtype(work[c])]
    for c in cat_cols:
        work[c] = work[c].astype("category").cat.codes.replace(-1, np.nan)

    imputer = SimpleImputer(strategy=cfg["data"].get("missing_strategy","median"))
    X = pd.DataFrame(imputer.fit_transform(work), columns=work.columns, index=work.index)

    if bool(cfg["data"].get("scale_numeric", True)):
        scaler = StandardScaler()
        X.loc[:, :] = scaler.fit_transform(X)

    out = X.copy()
    out["target"] = y.values
    return out

def prepare(cfg_path="config.yaml", force_demo=False, output="prepared_clinical.csv"):
    cfg = load_config(cfg_path)
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"])
    df, target, source = load_primary(cfg, force_demo=force_demo)
    prepared = preprocess_table(df, target, cfg)
    prepared = overlay(prepared, seed=int(cfg["seed"]))
    prepared["source_cohort"] = source
    prepared.to_csv(output, index=False)
    meta = {
        "source": source,
        "n_rows": int(len(prepared)),
        "n_columns": int(prepared.shape[1]),
        "target_column": "target",
        "synthetic_pii": True,
    }
    Path("prepared_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return prepared, meta

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--output", default="prepared_clinical.csv")
    args = ap.parse_args()
    _, meta = prepare(args.config, args.demo, args.output)
    print(json.dumps(meta, indent=2))
