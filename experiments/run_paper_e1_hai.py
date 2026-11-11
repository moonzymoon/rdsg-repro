# -*- coding: utf-8 -*-
"""E1b: HAI 20.07 real attack labels (operating-point curve protocol, identical
to Gas Pipeline).

Reference segment = train1 (3.6 days attack-free, median downsampled to 10s);
test = test1+test2 (38 labelled attack events). Channels = top-30 process
variables by variance (standardized).
"""
import os

def _results_root():
    """Results directory: nearest ancestor shipping a results/ folder.

    Reviewer package: <pkg>/experiments/x.py -> <pkg>/results (the shipped
    frozen JSONs), so reruns merge with them instead of writing outside;
    local repo layout resolves to <project>/results."""
    d = os.path.dirname(os.path.abspath(__file__))
    for base in (d, os.path.dirname(d), os.path.dirname(os.path.dirname(d))):
        if os.path.isdir(os.path.join(base, "results")):
            return os.path.join(base, "results")
    return os.path.normpath(os.path.join(d, "..", "..", "results"))

import os, sys, json, time
import numpy as np

ROOT = _results_root()
import os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rdsg import RDSGDetector, event_metrics
from rdsg.datasets import load_hai20
from rdsg.metrics import match_events
from baselines.relational import (WindowDivergenceBaseline, VARResidualBaseline,
                                  GrangerBaseline, merge_events, pick_theta_split,
                                  _series_from_stat, _pair_split_series)
from baselines.point_agg import _fit_score

OUT = os.path.normpath(os.path.join(_results_root(), "paper"))
SPP = 10.0
MULTS_Q = [0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995, 0.999]


def evaluate_curve(true_events, times, scores, t_offset, test_len, gap=256):
    pts = []
    days = test_len * SPP / 86400.0
    fin = scores[np.isfinite(scores)]
    if len(fin) == 0:
        return []
    for q in MULTS_Q:
        th = float(np.quantile(fin, q))
        evs = merge_events(times, scores > th, gap)
        pr = [(int(a - t_offset), int(b - t_offset)) for a, b in evs]
        matches, delays, falses = match_events([(a, b, None) for a, b, _ in true_events], pr)
        rec = sum(1 for _, p in matches if p is not None) / max(len(true_events), 1)
        pts.append(dict(quantile=float(q), theta=th, recall=rec,
                        fa_per_day=len(falses) / max(days, 1e-9), n_false=len(falses),
                        n_pred=len(pr),
                        delay_med=float(np.median(delays)) if delays else None))
    return pts


def main():
    os.makedirs(OUT, exist_ok=True)
    X, spp, meta = load_hai20()
    r1 = meta["ref"][1]
    t0t, t1t = meta["test"]
    X_ref = X[:r1]
    true_events = [(e[0], e[1], e[2]) for e in meta["events"]]
    n_events = len(true_events)
    print(f"HAI-20.07: ref={r1} test={t1t-r1} events={n_events}", flush=True)
    res = {"n_events": n_events, "test_len": t1t - r1}
    curves = {}

    # ---- Ours ----
    t0 = time.time()
    dets_R, dets_E = [], []
    for K in (4, 2):
        for w, blk in ((128, 32), (64, 16)):
            for strata, lst in (((12, 0), dets_R), ((0, 16), dets_E)):
                d = RDSGDetector(K=K, w=w, blk=blk, B=199, q=0.05, seed=1,
                                 pair_strata=strata, n_subset=4)
                d.fit(X_ref, channels=list(range(X.shape[1])), taus=(0, 1, -1))
                d.channel_kind_ = "R" if strata[0] > 0 else "E"
                lst.append(d)
    from run_e0_syn2 import channel_scan
    Gcal_R = {id(d): d.granulate(X_ref) for d in dets_R}
    Gcal_E = {id(d): d.granulate(X_ref) for d in dets_E}
    Gf_R = {id(d): d.granulate(X) for d in dets_R}
    Gf_E = {id(d): d.granulate(X) for d in dets_E}
    evs_R, thR, (tqR, TqR) = channel_scan(dets_R, Gcal_R, Gf_R, "R", cal_end=r1,
                                          scan_start=r1, scan_end=len(X),
                                          selfscan_start=0, gap=256, spp=SPP)
    evs_E, thE, (tqE, TqE) = channel_scan(dets_E, Gcal_E, Gf_E, "E", cal_end=r1,
                                          scan_start=r1, scan_end=len(X),
                                          selfscan_start=0, gap=256, spp=SPP)
    grid = sorted(set(tqR.tolist()) | set(tqE.tolist()))
    Tmap = np.full(len(grid), -np.inf)
    for tq, Tq in ((tqR, TqR), (tqE, TqE)):
        pos = np.searchsorted(np.asarray(grid), tq)
        Tmap[pos] = np.maximum(Tmap[pos], Tq)
    fin = Tmap[np.isfinite(Tmap)]
    pts = []
    for q in MULTS_Q:
        th = float(np.quantile(fin, q))
        evs = merge_events(np.asarray(grid), Tmap > th, 256)
        pr = [(int(a - r1), int(b - r1)) for a, b in evs]
        matches, delays, falses = match_events([(a, b, None) for a, b, _ in true_events], pr)
        rec = sum(1 for _, p in matches if p is not None) / max(n_events, 1)
        pts.append(dict(quantile=q, recall=rec,
                        fa_per_day=len(falses) / max((t1t - r1) * SPP / 86400.0, 1e-9),
                        n_false=len(falses), n_pred=len(pr),
                        delay_med=float(np.median(delays)) if delays else None))
    curves["rdsg"] = pts
    res["rdsg_runtime"] = time.time() - t0
    print("rdsg:", [(round(p["recall"], 2), round(p["fa_per_day"], 1)) for p in pts],
          f"{res['rdsg_runtime']:.0f}s", flush=True)

    # ---- Baselines ----
    for name, bl, cal_n in (("b1_corr", WindowDivergenceBaseline("corr", w=128, blk=32, spp=SPP, gap=256), 60),
                            ("b5b_pcorr", WindowDivergenceBaseline("pcorr", w=128, blk=32, spp=SPP, gap=256), 60),
                            ("b5c_var", VARResidualBaseline(w=128, p=3, spp=SPP, gap=256), 40)):
        t0 = time.time()
        t_r, s_r = _pair_split_series(X_ref, bl._stat, 0, r1, cal_n)
        base = pick_theta_split(s_r, 0.25, SPP, 256, bl.w)
        t_q, s_q = _series_from_stat(X, bl.w, bl.blk, bl._stat, r1, len(X))
        curves[name] = evaluate_curve(true_events, t_q, s_q, r1, t1t - r1)
        res[f"{name}_runtime"] = time.time() - t0
        print(name, [(round(p["recall"], 2), round(p["fa_per_day"], 1)) for p in curves[name]], flush=True)

    t0 = time.time()
    gb = GrangerBaseline([(i, j) for i in range(X_ref.shape[1]) for j in range(X_ref.shape[1]) if i != j][:16],
                         w=128, p=3, spp=SPP, gap=256)
    t_r, s_r = _pair_split_series(X_ref, gb._stat, 0, r1, 24)
    base = pick_theta_split(s_r, 0.25, SPP, 256, 128)
    t_q, s_q = _series_from_stat(X, 128, 32, gb._stat, r1, len(X))
    curves["b2_granger"] = evaluate_curve(true_events, t_q, s_q, r1, t1t - r1)
    res["b2_granger_runtime"] = time.time() - t0
    print("b2_granger", [(round(p["recall"], 2), round(p["fa_per_day"], 1)) for p in curves["b2_granger"]], flush=True)

    for name, model in (("b4_if", "if"), ("b4_lof", "lof")):
        t0 = time.time()
        n = X_ref.shape[1]
        half = r1 // 2
        ra = np.full(r1, -np.inf)
        for j in range(n):
            ra[:half] = np.maximum(ra[:half], _fit_score(model, X_ref[half:, j:j+1], X_ref[:half, j:j+1], 0))
            ra[half:] = np.maximum(ra[half:], _fit_score(model, X_ref[:half, j:j+1], X_ref[half:, j:j+1], 0))
        base = pick_theta_split(ra, 0.25, SPP, 256, 128)
        agg = np.full(len(X) - r1, -np.inf)
        for j in range(n):
            agg = np.maximum(agg, _fit_score(model, X_ref[:, j:j+1], X[r1:, j:j+1], 0))
        curves[name] = evaluate_curve(true_events, np.arange(r1, len(X)), agg, r1, t1t - r1)
        res[f"{name}_runtime"] = time.time() - t0
        print(name, [(round(p["recall"], 2), round(p["fa_per_day"], 1)) for p in curves[name]], flush=True)

    res["curves"] = curves
    json.dump(res, open(os.path.join(OUT, "e1_hai.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=float)
    print("saved e1_hai.json")


if __name__ == "__main__":
    main()
