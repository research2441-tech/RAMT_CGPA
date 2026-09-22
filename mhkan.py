from __future__ import annotations
import math, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

class SplineFeatureMap(nn.Module):
    def __init__(self, in_dim, grid_size=8):
        super().__init__()
        self.in_dim = in_dim
        self.grid_size = grid_size
        centers = torch.linspace(-2.5, 2.5, grid_size)
        self.register_buffer("centers", centers)
        self.log_width = nn.Parameter(torch.tensor(-0.2))
        self.coeff = nn.Parameter(torch.randn(in_dim, grid_size)*0.05)

    def forward(self, x):
        width = torch.exp(self.log_width) + 1e-4
        phi = torch.exp(-((x.unsqueeze(-1)-self.centers)**2)/(2*width*width))
        return (phi * self.coeff.unsqueeze(0)).sum(-1)

class KANBlock(nn.Module):
    def __init__(self, in_dim, out_dim, grid=8, dropout=.1):
        super().__init__()
        self.spline = SplineFeatureMap(in_dim, grid)
        self.linear = nn.Linear(in_dim, out_dim)
        self.residual = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        z = self.spline(x)
        h = torch.nn.functional.gelu(self.linear(z))
        return self.norm(self.dropout(h) + self.residual(x))

class MHKAN(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, embedding_dim=32, heads=4,
                 layers=2, grid=8, dropout=.1, n_classes=5):
        super().__init__()
        blocks = []
        d = in_dim
        for _ in range(layers):
            blocks.append(KANBlock(d, hidden_dim, grid, dropout))
            d = hidden_dim
        self.blocks = nn.ModuleList(blocks)
        self.attn = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.proj = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, embedding_dim))
        self.head = nn.Linear(embedding_dim, n_classes)

    def encode(self, x):
        for b in self.blocks: x = b(x)
        seq = x.unsqueeze(1)
        a,_ = self.attn(seq,seq,seq,need_weights=False)
        emb = self.proj(a.squeeze(1))
        return emb

    def forward(self, x):
        e = self.encode(x)
        return self.head(e), e

def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_mhkan(X_train, y_train, X_val, y_val, cfg, n_classes):
    c = cfg.get("mhkan",{})
    torch.manual_seed(int(cfg.get("seed",42))); np.random.seed(int(cfg.get("seed",42)))
    model = MHKAN(
        in_dim=X_train.shape[1],
        hidden_dim=int(c.get("hidden_dim",64)),
        embedding_dim=int(c.get("embedding_dim",32)),
        heads=int(c.get("num_heads",4)),
        layers=int(c.get("num_layers",2)),
        grid=int(c.get("spline_grid_size",8)),
        dropout=float(c.get("dropout",.1)),
        n_classes=n_classes
    ).to(_device())
    opt = torch.optim.AdamW(model.parameters(), lr=float(c.get("learning_rate",1e-3)),
                            weight_decay=float(c.get("weight_decay",1e-4)))
    crit = nn.CrossEntropyLoss()
    ds = TensorDataset(torch.tensor(X_train,dtype=torch.float32), torch.tensor(y_train,dtype=torch.long))
    dl = DataLoader(ds,batch_size=int(c.get("batch_size",64)),shuffle=True)
    best, best_state, wait = float("inf"), None, 0
    for epoch in range(int(c.get("epochs",20))):
        model.train()
        for xb,yb in dl:
            xb,yb=xb.to(_device()),yb.to(_device())
            opt.zero_grad(); logits,_=model(xb); loss=crit(logits,yb); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            xv=torch.tensor(X_val,dtype=torch.float32).to(_device())
            yv=torch.tensor(y_val,dtype=torch.long).to(_device())
            vl=float(crit(model(xv)[0],yv).cpu())
        if vl < best-1e-5:
            best=vl; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; wait=0
        else:
            wait+=1
            if wait>=int(c.get("patience",5)): break
    if best_state: model.load_state_dict(best_state)
    return model

def embed(model, X):
    model.eval()
    with torch.no_grad():
        t=torch.tensor(X,dtype=torch.float32).to(_device())
        return model(t)[1].cpu().numpy()
