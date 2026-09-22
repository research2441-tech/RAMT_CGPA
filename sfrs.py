from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

def relevance_scores(X: pd.DataFrame, y: pd.Series, seed=42):
    arr = X.to_numpy(dtype=float)
    mi = mutual_info_classif(arr, y.to_numpy(), random_state=seed)
    if mi.max() > 0: mi = mi / mi.max()
    return pd.Series(mi, index=X.columns, name="relevance")

def redundancy_matrix(X: pd.DataFrame):
    c = X.corr(method="spearman").abs().fillna(0.0)
    return c

def spherical_membership(x, center, sigma=1.0):
    d2 = float(np.sum((np.asarray(x)-np.asarray(center))**2))
    mu = np.exp(-d2/(2*sigma*sigma + 1e-12))
    nu = min(1.0, np.sqrt(max(0.0, 1.0-mu*mu))*0.70)
    pi = np.sqrt(max(0.0, 1.0-mu*mu-nu*nu))
    return mu, nu, pi

def dependency_proxy(X, y, selected, sigma=1.0):
    if not selected: return 0.0
    arr = X[selected].to_numpy(float)
    yv = np.asarray(y)
    classes = np.unique(yv)
    centers = {c:arr[yv==c].mean(0) for c in classes}
    correct = 0.0
    for row, true in zip(arr,yv):
        scores = {c:spherical_membership(row, centers[c], sigma)[0] for c in classes}
        pred = max(scores, key=scores.get)
        correct += float(pred == true)
    return correct/len(arr)

def select_features(X: pd.DataFrame, y: pd.Series, cfg=None, seed=42):
    c = (cfg or {}).get("sfrs", {})
    maxf = min(int(c.get("max_features",10)), X.shape[1])
    minf = min(int(c.get("min_features",5)), maxf)
    rw = float(c.get("relevance_weight",.70))
    dw = float(c.get("redundancy_weight",.30))
    sigma = float(c.get("similarity_sigma",1.0))
    rel = relevance_scores(X,y,seed)
    red = redundancy_matrix(X)
    selected, history = [], []
    candidates = list(X.columns)
    while candidates and len(selected) < maxf:
        best, best_score = None, -1e18
        for f in candidates:
            redundancy = float(red.loc[f, selected].mean()) if selected else 0.0
            score = rw*float(rel[f]) - dw*redundancy
            if score > best_score:
                best, best_score = f, score
        selected.append(best); candidates.remove(best)
        dep = dependency_proxy(X,y,selected,sigma)
        history.append({"step":len(selected),"feature":best,"score":best_score,"dependency":dep})
        if len(selected) >= minf and len(history)>=2 and history[-1]["dependency"] <= history[-2]["dependency"] + 1e-4:
            break
    return selected, pd.DataFrame(history)
