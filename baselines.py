from __future__ import annotations
import time, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,average_precision_score
from sklearn.preprocessing import label_binarize
from xgboost import XGBClassifier

try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier=None

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

def metrics(y, pred, prob=None):
    p,r,f,_=precision_recall_fscore_support(y,pred,average="macro",zero_division=0)
    out={"accuracy":accuracy_score(y,pred),"macro_precision":p,"macro_recall":r,"macro_f1":f}
    if prob is not None:
        try:
            Y=label_binarize(y,classes=np.arange(prob.shape[1]))
            out["pr_auc"]=average_precision_score(Y,prob,average="macro")
        except Exception: out["pr_auc"]=np.nan
    else: out["pr_auc"]=np.nan
    return out

class MLP(nn.Module):
    def __init__(self,d,k):
        super().__init__(); self.net=nn.Sequential(nn.Linear(d,64),nn.ReLU(),nn.Dropout(.1),
            nn.Linear(64,32),nn.ReLU(),nn.Linear(32,k))
    def forward(self,x): return self.net(x)

class SequenceClassifier(nn.Module):
    def __init__(self,d,k,bidir=False):
        super().__init__(); self.rnn=nn.LSTM(1,32,batch_first=True,bidirectional=bidir)
        self.fc=nn.Linear(64 if bidir else 32,k)
    def forward(self,x):
        o,_=self.rnn(x.unsqueeze(-1)); return self.fc(o[:,-1])

class TransformerTabular(nn.Module):
    def __init__(self,d,k,heads=4,attention_pool=False):
        super().__init__()
        emb=32
        self.proj=nn.Linear(1,emb)
        layer=nn.TransformerEncoderLayer(emb,heads,64,batch_first=True,dropout=.1)
        self.enc=nn.TransformerEncoder(layer,2)
        self.attention_pool=attention_pool
        self.query=nn.Parameter(torch.randn(1,1,emb)) if attention_pool else None
        self.fc=nn.Linear(emb,k)
    def forward(self,x):
        z=self.enc(self.proj(x.unsqueeze(-1)))
        if self.attention_pool:
            q=self.query.expand(x.size(0),-1,-1)
            score=torch.softmax((q@z.transpose(1,2))/np.sqrt(z.size(-1)),dim=-1)
            z=(score@z).squeeze(1)
        else: z=z.mean(1)
        return self.fc(z)

def train_torch(model,Xtr,ytr,Xv,yv,epochs=15,seed=42):
    torch.manual_seed(seed); dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=model.to(dev); opt=torch.optim.AdamW(model.parameters(),lr=1e-3); loss=nn.CrossEntropyLoss()
    dl=DataLoader(TensorDataset(torch.tensor(Xtr,dtype=torch.float32),torch.tensor(ytr,dtype=torch.long)),
                  batch_size=64,shuffle=True)
    best=1e9; state=None
    for _ in range(epochs):
        model.train()
        for xb,yb in dl:
            xb,yb=xb.to(dev),yb.to(dev); opt.zero_grad(); l=loss(model(xb),yb); l.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vv=float(loss(model(torch.tensor(Xv,dtype=torch.float32).to(dev)),
                          torch.tensor(yv,dtype=torch.long).to(dev)).cpu())
        if vv<best: best=vv; state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    if state: model.load_state_dict(state)
    return model

def evaluate_models(Xtr,ytr,Xv,yv,Xte,yte,seed=42):
    k=len(np.unique(np.concatenate([ytr,yv,yte]))); rows=[]
    sk={
      "LogisticRegression":LogisticRegression(max_iter=3000,random_state=seed),
      "SVM":SVC(C=2.0,probability=True,random_state=seed),
      "RandomForest":RandomForestClassifier(n_estimators=300,random_state=seed,n_jobs=-1),
      "XGBoost":XGBClassifier(n_estimators=300,max_depth=6,learning_rate=.05,subsample=.9,
                              colsample_bytree=.9,objective="multi:softprob",eval_metric="mlogloss",
                              random_state=seed,n_jobs=-1,tree_method="hist")
    }
    if CatBoostClassifier:
        sk["CatBoost"]=CatBoostClassifier(iterations=300,depth=6,learning_rate=.05,verbose=False,random_seed=seed)
    for name,m in sk.items():
        t=time.perf_counter(); m.fit(np.vstack([Xtr,Xv]),np.concatenate([ytr,yv])); train=time.perf_counter()-t
        t=time.perf_counter(); pred=m.predict(Xte); infer=(time.perf_counter()-t)*1000/max(len(yte),1)
        pred=np.asarray(pred).reshape(-1).astype(int)
        prob=m.predict_proba(Xte) if hasattr(m,"predict_proba") else None
        rows.append({"model":name,"train_seconds":train,"inference_ms_per_sample":infer,**metrics(yte,pred,prob)})
    torch_models={
      "MLP":MLP(Xtr.shape[1],k),
      "LSTM":SequenceClassifier(Xtr.shape[1],k,False),
      "BiLSTM":SequenceClassifier(Xtr.shape[1],k,True),
      "TransformerEncoder":TransformerTabular(Xtr.shape[1],k,4,False),
      "AttentionTransformer":TransformerTabular(Xtr.shape[1],k,4,True),
    }
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for name,m in torch_models.items():
        t=time.perf_counter(); m=train_torch(m,Xtr,ytr,Xv,yv,epochs=12,seed=seed); train=time.perf_counter()-t
        m.eval(); xt=torch.tensor(Xte,dtype=torch.float32).to(dev)
        t=time.perf_counter()
        with torch.no_grad(): logits=m(xt); prob=torch.softmax(logits,1).cpu().numpy()
        infer=(time.perf_counter()-t)*1000/max(len(yte),1); pred=prob.argmax(1)
        rows.append({"model":name,"train_seconds":train,"inference_ms_per_sample":infer,**metrics(yte,pred,prob)})
    return pd.DataFrame(rows)
