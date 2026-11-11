# -*- coding: utf-8 -*-
"""Anomaly Transformer baseline on our injection protocol.

Trains AT on reference segment (normal), scores injected test,
calibrates at same FA budget, reports event recall.
Simplified: small model, fast training, same event-matching protocol.
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
# Official Anomaly-Transformer architecture is imported from the public repo
# (github.com/thuml/Anomaly-Transformer): set $AT_ROOT to its checkout, or
# place it at ./Anomaly-Transformer relative to the project root.
AT_ROOT = os.environ.get("AT_ROOT",
                         os.path.normpath(os.path.join(HERE, "..", "..",
                                                       "Anomaly-Transformer")))
CODE = os.path.normpath(os.path.join(HERE, ".."))
PROJECT = os.path.normpath(os.path.join(CODE, ".."))
sys.path.insert(0, CODE)
sys.path.insert(0, HERE)

try:
    sys.path.insert(0, os.path.join(AT_ROOT))
    from model.AnomalyTransformer import AnomalyTransformer
except ImportError as _e:  # optional dependency: fail only at run time
    AnomalyTransformer = None
    AT_IMPORT_ERROR = _e

from rdsg.metrics import match_events, event_metrics
from run_paper_e2 import LEN_BY_SPP, build_channel, make_specs, PAIR_STRATA_E
from rdsg import inject
from baselines.relational import merge_events
def _results_root():
    """Results directory: the nearest ancestor that ships a results/ folder.

    Local repo: code/experiments/x.py -> <project>/results.
    Reviewer package: <pkg>/experiments/x.py -> <pkg>/results (the shipped
    frozen JSONs), so reruns merge with them instead of writing outside."""
    d = os.path.dirname(os.path.abspath(__file__))
    for base in (d, os.path.dirname(d), os.path.dirname(os.path.dirname(d))):
        if os.path.isdir(os.path.join(base, "results")):
            return os.path.join(base, "results")
    return os.path.normpath(os.path.join(d, "..", "..", "results"))



def _require_at():
    if AnomalyTransformer is None:
        raise RuntimeError(
            "The official Anomaly-Transformer checkout was not found. "
            "Clone github.com/thuml/Anomaly-Transformer and set $AT_ROOT "
            f"to its root (current AT_ROOT={AT_ROOT!r}). Original import "
            f"error: {AT_IMPORT_ERROR}")
    return AnomalyTransformer

OUT = os.path.join(_results_root(), "paper", "abl_at.json")
WIN = 128
FA_TARGET = 0.25
SEEDS = range(1, 11)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ATDetector:
    """Wrapper: train on ref, score on test."""

    def __init__(self, n_channels, win=WIN, d_model=64, n_heads=4, e_layers=2):
        self.model = _require_at()(
            win_size=win, enc_in=n_channels, c_out=n_channels,
            d_model=d_model, n_heads=n_heads, e_layers=e_layers,
            d_ff=d_model * 2, dropout=0.0, activation="gelu"
        ).to(DEVICE)
        self.win = win

    def _windows(self, X):
        """Extract overlapping windows: (n_win, win, C)."""
        n = len(X) - self.win + 1
        idx = np.arange(0, n, self.win // 4)  # stride = win/4
        ws = np.stack([X[i:i + self.win] for i in idx])
        return idx, ws.astype(np.float32)

    def _score(self, X):
        """Association discrepancy per window."""
        self.model.eval()
        idx, ws = self._windows(X)
        scores = []
        with torch.no_grad():
            for i in range(0, len(ws), 32):
                batch = torch.tensor(ws[i:i+32]).to(DEVICE)
                out, series, prior, sigma = self.model(batch)
                # Association discrepancy: KL(series || prior) averaged over layers/heads
                kl = 0
                for s_l, p_l in zip(series, prior):
                    # s_l, p_l: (B, H, L, L)
                    kl_l = (s_l * (torch.log(s_l + 1e-8) - torch.log(p_l + 1e-8))).sum(-1)
                    kl += kl_l.mean(axis=(1, 2))  # avg over heads, sum over positions
                scores.append(kl.cpu().numpy())
        scores = np.concatenate(scores)
        return idx, scores

    def train(self, X_ref, epochs=5, lr=1e-4):
        """Simple training: reconstruct + minimize association discrepancy."""
        self.model.train()
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        _, ws = self._windows(X_ref)
        ws_t = torch.tensor(ws).to(DEVICE)
        for ep in range(epochs):
            perm = torch.randperm(len(ws_t))
            total = 0
            for i in range(0, len(ws_t), 32):
                batch = ws_t[perm[i:i+32]]
                opt.zero_grad()
                out, series, prior, sigma = self.model(batch)
                # Reconstruction loss
                rec = F.mse_loss(out, batch)
                # Association loss (minimize for series, maximize for prior — simplified: minimize KL)
                kl = 0
                for s_l, p_l in zip(series, prior):
                    kl_l = (s_l * (torch.log(s_l + 1e-8) - torch.log(p_l + 1e-8))).sum(-1)
                    kl += kl_l.mean()
                loss = rec + 0.01 * kl
                loss.backward()
                opt.step()
                total += loss.item()
            if ep == 0 or (ep + 1) % 2 == 0:
                print(f"    epoch {ep+1}/{epochs} loss={total/max(1, len(ws_t)//32):.4f}", flush=True)
        self.model.eval()


def at_pipeline(X_ref, X_test_inj, spp, fa_target=FA_TARGET):
    """Train AT on ref, score test, calibrate, return events."""
    n_ch = X_ref.shape[1]
    det = ATDetector(n_ch)
    det.train(X_ref, epochs=5)

    # Reference self-scan scores (second half of ref)
    half = len(X_ref) // 2
    _, ref_scores = det._score(X_ref[half:])
    # Test scores
    test_idx, test_scores = det._score(X_test_inj)

    # Rate-based calibration
    span_days = len(ref_scores) * (WIN // 4) * spp / 86400.0
    k_allow = int(np.floor(fa_target * span_days))
    srt = np.sort(ref_scores)
    theta = float(srt[len(srt) - k_allow - 1]) if k_allow < len(srt) else float(srt[0])

    # Convert to events
    alarm = test_scores > theta
    gap = 4 * WIN
    on_idx = test_idx[alarm] if isinstance(alarm, np.ndarray) else np.where(alarm)[0]
    if len(on_idx) == 0:
        return []
    events = []
    t0, prev = on_idx[0], on_idx[0]
    for idx in on_idx[1:]:
        if idx - prev > gap:
            events.append((t0, prev + WIN))
            t0 = idx
        prev = idx
    events.append((t0, prev + WIN))
    return events


def run_dataset(key, loader):
    X, spp, meta = loader()
    X_ref, X_test = X[:12000], X[12000:12000 + 18000]
    lengths = LEN_BY_SPP[spp]
    taus = (0, 1, -1)
    planner = build_channel(X_ref, taus, strata=(16, 24))[0]
    force_R, force_E = planner.strong_pairs_, planner.weak_pairs_
    dR0 = planner
    dE0 = build_channel(X_ref, taus, strata=PAIR_STRATA_E, n_subset=6,
                        force_pairs=(force_R, force_E))[0]

    res = {}
    for seed in SEEDS:
        specs = make_specs(dR0, dE0, X_ref, X_test, seed, lengths)
        X_test_inj = inject(X_test, specs, seed=seed)
        true_events = [(sp["start"], sp["start"] + sp["length"], sp["type"]) for sp in specs]
        t0 = time.time()
        evs = at_pipeline(X_ref, X_test_inj, spp)
        pr3 = [(int(a), int(b), None) for a, b in evs]
        m = event_metrics(true_events, pr3, len(X_test), spp)
        res[str(seed)] = m
        print(f"  [{key} s{seed}] R={m['recall']:.3f} fa/d={m['fa_per_day']:.2f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    return {key: res}


def main():
    from rdsg.datasets import load_metr_la, load_pems_bay, load_beijing_air
    loaders = {"metr_la": load_metr_la, "pems_bay": load_pems_bay,
               "beijing_air": load_beijing_air}
    all_res = {}
    for key, loader in loaders.items():
        print(f"\n===== {key} =====", flush=True)
        all_res.update(run_dataset(key, loader))
        json.dump(all_res, open(OUT, "w"), indent=1)
        recalls = [v["recall"] for v in all_res.get(key, {}).values()]
        if recalls:
            print(f"  [{key}] mean={np.mean(recalls):.3f} ± {np.std(recalls):.3f}")
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
