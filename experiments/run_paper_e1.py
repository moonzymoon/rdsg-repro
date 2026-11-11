# -*- coding: utf-8 -*-
"""E1: Gas Pipeline real attack-label experiment (operating-point curve protocol).

The reference segment is only 27 min (the longest contiguous normal segment in
the dataset), and mu/sigma/theta suffer a known transfer amplification on the
9.9h test span (operating-mode drift) -> single-point FA matching is unreliable.
Protocol: for each method take the reference-calibrated base threshold and sweep
it by multiplicative factors to obtain an (FA, recall) operating-point curve;
report recall at matched FA levels and the full curve. The paper states the
short-reference limitation honestly.
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
from rdsg.datasets import load_gas_pipeline
from rdsg.metrics import match_events
from baselines.relational import (WindowDivergenceBaseline, VARResidualBaseline,
                                  GrangerBaseline, merge_events, pick_theta_split,
                                  _series_from_stat, _pair_split_series)
from baselines.point_agg import _fit_score

OUT = os.path.normpath(os.path.join(_results_root(), "paper"))
SPP = 1.0
MULTS = [1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 8.0]


def evaluate_curve(true_events, times, scores, base_theta, mults, t_offset, test_len, gap=64):
    """Test-score quantile sweep (v2): theta = 0.90->0.999 quantile of the test
    scores, yielding a complete (FA, recall) operating-point curve -- method
    agnostic and free of calibration ambiguity; the paper reports the full
    curve and matched-FA points."""
    pts = []
    days = test_len * SPP / 86400.0
    fin = scores[np.isfinite(scores)]
    if len(fin) == 0:
        return pts
    qs = np.linspace(0.90, 0.999, 25)
    for q in qs:
        th = float(np.quantile(fin, q))
        evs = merge_events(times, scores > th, gap)
        pr = [(int(a - t_offset), int(b - t_offset)) for a, b in evs]
        matches, delays, falses = match_events([(a, b, None) for a, b, _ in true_events], pr)
        rec = sum(1 for _, p in matches if p is not None) / max(len(true_events), 1)
        pts.append(dict(quantile=float(q), theta=th, recall=rec,
                        fa_per_day=len(falses) / max(days, 1e-9),
                        n_false=len(falses), n_pred=len(pr),
                        delay_med=float(np.median(delays)) if delays else None))
    return pts


def main():
    os.makedirs(OUT, exist_ok=True)
    X, spp, meta = load_gas_pipeline()
    r0, r1 = meta["ref"]
    t0t, t1t = meta["test"]
    X_ref = X[r0:r1]
    n_ref = len(X_ref)
    true_events = [(e[0], e[1], e[2]) for e in meta["events"]]
    print(f"Gas-Pipeline: ref={n_ref} test={t1t-t0t} events={len(true_events)}")
    res = {"n_events": len(true_events), "test_len": t1t - t0t}
    curves = {}

    # ---- Ours (union score curve over the two channel families) ----
    t0 = time.time()
    dets_R, dets_E = [], []
    for K in (4, 2):
        for w, blk in ((128, 16), (64, 16)):
            for strata, lst in (((12, 0), dets_R), ((0, 16), dets_E)):
                d = RDSGDetector(K=K, w=w, blk=blk, B=499, q=0.05, seed=1,
                                 pair_strata=strata, n_subset=4)
                d.fit(X_ref, channels=list(range(X_ref.shape[1])), taus=(0, 1, -1))
                d.channel_kind_ = "R" if strata[0] > 0 else "E"
                lst.append(d)
    from run_e0_syn2 import channel_scan
    X_full = X
    Gcal_R = {id(d): d.granulate(X_ref) for d in dets_R}
    Gcal_E = {id(d): d.granulate(X_ref) for d in dets_E}
    evs_R, thR, (tqR, TqR) = channel_scan(dets_R, Gcal_R, {id(d): d.granulate(X_full) for d in dets_R}, "R",
                                          cal_end=n_ref, scan_start=n_ref, scan_end=len(X_full),
                                          selfscan_start=0, gap=4 * 64, spp=SPP)
    evs_E, thE, (tqE, TqE) = channel_scan(dets_E, Gcal_E, {id(d): d.granulate(X_full) for d in dets_E}, "E",
                                          cal_end=n_ref, scan_start=n_ref, scan_end=len(X_full),
                                          selfscan_start=0, gap=4 * 64, spp=SPP)
    grid = sorted(set(tqR.tolist()) | set(tqE.tolist()))
    Tmap = np.full(len(grid), -np.inf)
    for tq, Tq in ((tqR, TqR), (tqE, TqE)):
        pos = np.searchsorted(np.asarray(grid), tq)
        Tmap[pos] = np.maximum(Tmap[pos], Tq)
    base = float(np.quantile(Tmap[Tmap > -1], 0.999))
    curves["rdsg"] = evaluate_curve(true_events, np.asarray(grid), Tmap, None, MULTS, n_ref, t1t - n_ref)
    res["rdsg_runtime"] = time.time() - t0
    print("rdsg curve:", [(round(p["recall"], 2), round(p["fa_per_day"], 1)) for p in curves["rdsg"]],
          f"{res['rdsg_runtime']:.0f}s", flush=True)

    # ---- Window-divergence baselines ----
    for name, bl, cal_n in (("b1_corr", WindowDivergenceBaseline("corr", w=64, blk=16, spp=SPP, gap=64), 60),
                            ("b5b_pcorr", WindowDivergenceBaseline("pcorr", w=64, blk=16, spp=SPP, gap=64), 60),
                            ("b5c_var", VARResidualBaseline(w=64, p=3, spp=SPP, gap=64), 40)):
        t0 = time.time()
        t_r, s_r = _pair_split_series(X_ref, bl._stat, 0, n_ref, cal_n)
        base = pick_theta_split(s_r, 0.25, SPP, 64, bl.w)
        t_q, s_q = _series_from_stat(X, bl.w, bl.blk, bl._stat, n_ref, len(X))
        curves[name] = evaluate_curve(true_events, t_q, s_q, None, MULTS, n_ref, t1t - n_ref)
        res[f"{name}_runtime"] = time.time() - t0
        print(name, [(round(p["recall"], 2), round(p["fa_per_day"], 1)) for p in curves[name]],
              flush=True)

    # ---- Granger ----
    t0 = time.time()
    gb = GrangerBaseline([(i, j) for i in range(X_ref.shape[1]) for j in range(X_ref.shape[1]) if i != j][:16],
                         w=64, p=3, spp=SPP, gap=64)
    t_r, s_r = _pair_split_series(X_ref, gb._stat, 0, n_ref, 24)
    base = pick_theta_split(s_r, 0.25, SPP, 64, 64)
    t_q, s_q = _series_from_stat(X, 64, 16, gb._stat, n_ref, len(X))
    curves["b2_granger"] = evaluate_curve(true_events, t_q, s_q, None, MULTS, n_ref, t1t - n_ref)
    res["b2_granger_runtime"] = time.time() - t0
    print("b2_granger", [(round(p["recall"], 2), round(p["fa_per_day"], 1)) for p in curves["b2_granger"]],
          flush=True)

    # ---- IF / LOF (point level) ----
    for name, model in (("b4_if", "if"), ("b4_lof", "lof")):
        t0 = time.time()
        n = X_ref.shape[1]
        half = n_ref // 2
        ra = np.full(n_ref, -np.inf)
        for j in range(n):
            ra[:half] = np.maximum(ra[:half], _fit_score(model, X_ref[half:, j:j+1], X_ref[:half, j:j+1], 0))
            ra[half:] = np.maximum(ra[half:], _fit_score(model, X_ref[:half, j:j+1], X_ref[half:, j:j+1], 0))
        base = pick_theta_split(ra, 0.25, SPP, 64, 64)
        agg = np.full(len(X) - n_ref, -np.inf)
        for j in range(n):
            agg = np.maximum(agg, _fit_score(model, X_ref[:, j:j+1], X[n_ref:, j:j+1], 0))
        curves[name] = evaluate_curve(true_events, np.arange(n_ref, len(X)), agg, None, MULTS, n_ref, t1t - n_ref)
        res[f"{name}_runtime"] = time.time() - t0
        print(name, [(round(p["recall"], 2), round(p["fa_per_day"], 1)) for p in curves[name]],
              flush=True)

    res["curves"] = curves
    json.dump(res, open(os.path.join(OUT, "e1_gas.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=float)
    print("saved e1_gas.json")


if __name__ == "__main__":
    main()
