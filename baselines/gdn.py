# -*- coding: utf-8 -*-
"""B3: GDN (Deng & Hooi, AAAI 2021) compact re-implementation.

Embedding-based graph learning + GRD attention forecasting + forecast-deviation
scoring; trained on the reference segment, scored on the test segment, with an
FA-matched threshold identical to ours/other baselines.
"""
import numpy as np
import torch
import torch.nn as nn


class GDNModel(nn.Module):
    def __init__(self, n, window=32, emb=16, hid=64):
        super().__init__()
        self.n, self.window = n, window
        self.node_emb = nn.Parameter(torch.randn(n, emb) * 0.1)
        self.weight_vec = nn.Parameter(torch.randn(n, emb) * 0.1)
        self.fc1 = nn.Linear(window + emb, hid)
        self.fc2 = nn.Linear(hid, 1)

    def adj(self):
        a = torch.softmax(self.node_emb @ self.weight_vec.T, dim=1)
        return a

    def forward(self, x):
        """x: (B, window, n) -> predicts every channel at the current time step (B, n)."""
        B = x.shape[0]
        A = self.adj()                                     # (n, n) target row = attention over sources
        xc = x - x.mean(dim=1, keepdim=True)               # local standardization
        neigh = torch.einsum("ij,btj->bti", A, xc)         # (B, window, n) aggregated neighborhoods
        cur = neigh[:, -1, :]                              # (B, n)
        outs = []
        for j in range(self.n):
            z = torch.cat([neigh[:, :, j], self.node_emb[j].expand(B, -1)], dim=1)
            outs.append(self.fc2(torch.relu(self.fc1(z))).squeeze(-1))
        pred = torch.stack(outs, dim=1)                    # (B, n)
        return pred * x.std(dim=1).clamp(min=1e-6) + x.mean(dim=1)


def train_gdn(X_ref, window=32, epochs=6, bs=256, lr=1e-3, seed=0, device="cpu"):
    torch.manual_seed(seed)
    n = X_ref.shape[1]
    model = GDNModel(n, window).to(device)
    mu, sd = X_ref.mean(axis=0), np.maximum(X_ref.std(axis=0), 1e-9)
    Xn = (X_ref - mu) / sd
    P = X_ref.std(axis=0) + 1e-9
    Y = Xn[window:]
    Z = np.stack([Xn[k: k + (len(Xn) - window)] for k in range(window)], axis=1)
    Zt = torch.tensor(Z, dtype=torch.float32, device=device)
    Yt = torch.tensor(Y, dtype=torch.float32, device=device)
    P = torch.tensor(P / sd, dtype=torch.float32, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()
    for ep in range(epochs):
        perm = torch.randperm(len(Zt), device=device)[: bs * 40]
        for lo in range(0, len(perm), bs):
            idx = perm[lo:lo + bs]
            pred = model(Zt[idx])
            loss = lossf(pred / P, Yt[idx] / P)
            opt.zero_grad(); loss.backward(); opt.step()
    return model, mu, sd, device


def gdn_scores(model, X, mu, sd, device="cpu", bs=1024):
    window = model.window
    Xn = (X - mu) / sd
    n = X.shape[1]
    P = np.maximum(sd, 1e-9)
    scores = np.zeros(len(X))
    model.eval()
    with torch.no_grad():
        for lo in range(window, len(X), bs):
            hi = min(lo + bs, len(X))
            Z = np.stack([Xn[k - window: k] for k in range(lo, hi)], axis=0)
            Zt = torch.tensor(Z, dtype=torch.float32, device=device)
            pred = model(Zt).cpu().numpy() * P
            truth = X[lo:hi]
            dev = np.abs(truth - pred) / P
            scores[lo:hi] = dev.max(axis=1)
    return scores
