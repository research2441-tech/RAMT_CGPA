from __future__ import annotations
import hashlib, json
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

PII_NAMES = {
    "national_identity_hash","full_name","contact_number","street_address","zip_code",
    "date_of_birth","admission_timestamp","medical_record_number"
}

def normalized_entropy(s: pd.Series) -> float:
    counts = s.astype(str).value_counts(dropna=False).values.astype(float)
    p = counts / counts.sum()
    h = -(p * np.log2(p + 1e-12)).sum()
    return float(h / max(np.log2(len(p)+1e-12), 1e-12))

def uniqueness_risk(s: pd.Series) -> float:
    vc = s.astype(str).value_counts(dropna=False)
    return float((vc == 1).mean())

def diagnostic_utility_numeric(x: pd.Series, y: pd.Series) -> float:
    a = pd.to_numeric(x, errors="coerce").fillna(pd.to_numeric(x, errors="coerce").median())
    if a.nunique() <= 1:
        return 0.0
    try:
        mi = mutual_info_classif(a.to_numpy().reshape(-1,1), y.to_numpy(), random_state=42)[0]
        return float(mi)
    except Exception:
        return 0.0

def minmax(values):
    a = np.asarray(values, float)
    if len(a) == 0: return a
    lo, hi = np.nanmin(a), np.nanmax(a)
    if hi - lo < 1e-12: return np.zeros_like(a)
    return (a-lo)/(hi-lo)

def assign_tiers(df: pd.DataFrame, target="target", cfg=None):
    privacy = (cfg or {}).get("privacy", {})
    t1 = float(privacy.get("tier1_risk_threshold", 0.65))
    t2u = float(privacy.get("tier2_utility_threshold", 0.20))
    iw = float(privacy.get("identity_weight", 0.60))
    uw = float(privacy.get("utility_weight", 0.40))

    y = df[target].astype(int)
    cols = [c for c in df.columns if c != target and c != "source_cohort"]
    raw = []
    for c in cols:
        pii = 1.0 if c.lower() in PII_NAMES else 0.0
        risk = max(pii, 0.5*normalized_entropy(df[c]) + 0.5*uniqueness_risk(df[c]))
        utility = 0.0 if pii else diagnostic_utility_numeric(df[c], y)
        raw.append((c, risk, utility, pii))
    util_norm = minmax([r[2] for r in raw])
    rows = []
    for (c,risk,utility,pii), un in zip(raw, util_norm):
        score = iw*risk - uw*un
        if pii >= 1 or risk >= t1:
            tier = "T1"
        elif un >= t2u:
            tier = "T2"
        else:
            tier = "T3"
        rows.append({
            "attribute": c,
            "reidentification_risk": float(risk),
            "diagnostic_utility": float(un),
            "risk_utility_score": float(score),
            "tier": tier
        })
    return pd.DataFrame(rows)

def pseudonymize_record_id(index: int, seed=42) -> str:
    return hashlib.sha256(f"RAMT-CGPA::{seed}::{index}".encode()).hexdigest()[:32]

def apply_tiering(df, assignments, target="target"):
    t = {r.attribute:r.tier for r in assignments.itertuples()}
    return {
        "T1": df[[c for c in df.columns if t.get(c)=="T1"]].copy(),
        "T2": df[[c for c in df.columns if t.get(c)=="T2"] + [target]].copy(),
        "T3": df[[c for c in df.columns if t.get(c)=="T3"]].copy(),
    }

if __name__ == "__main__":
    import yaml
    df = pd.read_csv("prepared_clinical.csv")
    cfg = yaml.safe_load(open("config.yaml"))
    out = assign_tiers(df, "target", cfg)
    out.to_csv("tier_assignments.csv", index=False)
    print(out.to_string(index=False))
