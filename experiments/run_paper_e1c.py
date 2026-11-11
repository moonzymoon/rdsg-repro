# -*- coding: utf-8 -*-
"""E1c: Gas Pipeline relational-subset decomposition (review hardening,
2026-10-06).

For each of the 53 labelled attacks compute its point footprint (per-channel
max |z| within the attack window relative to the reference segment) and split
the events into two subsets: point-visible (max|z|>5) and relational-only
(<=5). Each method records, at its 25 operating points, the set of matched
true-event ids; output recall@FA<=5/day and max recall on both subsets.

Pre-registered branches (hardening plan 2.1):
(a) RDSG recall on the relational-only subset >= 2x the strongest point
    baseline -> table goes into the main text;
(b) otherwise -> withdraw "dominates"-style claims and downgrade to a
    complementarity narrative.
No method implementation or threshold protocol is modified -- the score
streams are exactly the same source as e1_gas.json.
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
from rdsg import RDSGDetector
from rdsg.datasets import load_gas_pipeline
from rdsg.metrics import match_events
from baselines.relational import (WindowDivergenceBaseline, VARResidualBaseline,
                                  GrangerBaseline, merge_events, pick_theta_split,
                                  _series_from_stat, _pair_split_series)
from baselines.point_agg import _fit_score
from run_e0_syn2 import channel_scan

OUT = os.path.normpath(os.path.join(_results_root(), "paper"))
SPP = 1.0
Z_VISIBLE = 5.0   # pre-registered point-footprint threshold: max|z|>5 within the attack window counts as point-visible
FA5 = 5.0         # recall@FA<=5/day


def evaluate_curve_ids(true_events, times, scores, t_offset, test_len, gap=64):
    """Same 25-quantile sweep as run_paper_e1.evaluate_curve, additionally
    recording the matched true-event ids at each point."""
    pts = []
    days = test_len * SPP / 86400.0
    fin = scores[np.isfinite(scores)]
    if len(fin) == 0:
        return pts
    true3 = [(a, b, None) for a, b, _ in true_events]
    true_index = {(int(a), int(b)): i for i, (a, b, _t) in enumerate(true_events)}
    for q in np.linspace(0.90, 0.999, 25):
        th = float(np.quantile(fin, q))
        evs = merge_events(times, scores > th, gap)
        pr = [(int(a - t_offset), int(b - t_offset)) for a, b in evs]
        matches, delays, falses = match_events(list(true3), pr)
        matched = sorted(true_index[(int(tr[0]), int(tr[1]))] for tr, p in matches if p is not None)
        pts.append(dict(quantile=float(q), theta=th,
                        recall=len(matched) / max(len(true_events), 1),
                        fa_per_day=len(falses) / max(days, 1e-9),
                        matched=matched))
    return pts


def recall_at_fa(pts, fa_cap):
    best = 0.0
    for p in pts:
        if p["fa_per_day"] <= fa_cap:
            best = max(best, p["recall"])
    return best


def main():
    os.makedirs(OUT, exist_ok=True)
    X, spp, meta = load_gas_pipeline()
    r0, r1 = meta["ref"]
    t0t, t1t = meta["test"]
    X_ref = X[r0:r1]
    n_ref = len(X_ref)
    true_events = [(e[0], e[1], e[2]) for e in meta["events"]]
    n_ev = len(true_events)
    mu = X_ref.mean(axis=0); sd = X_ref.std(axis=0) + 1e-12

    # ---- Attack point-footprint classification ----
    sub_id = []
    for (a, b, _lab) in true_events:
        a_, b_ = max(int(a), 0), min(int(b), len(X))
        seg = (X[a_:b_] - mu) / sd
        sub_id.append("relational-only" if float(np.abs(seg).max()) <= Z_VISIBLE else "point-visible")
    n_rel = sub_id.count("relational-only"); n_vis = n_vis = sub_id.count("point-visible")
    print(f"point-footprint classification: relational-only={n_rel}, point-visible={n_vis} / {n_ev}")
    res = {"z_visible": Z_VISIBLE, "fa_target_curve": "test-score quantiles 0.90-0.999",
           "n_events": n_ev, "n_relational_only": n_rel, "n_point_visible": n_vis,
           "subset_of": {i: sub_id[i] for i in range(n_ev)}}

    curves = {}

    # ---- RDSG (exactly the same configuration as run_paper_e1.py) ----
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
    X_full = X
    Gcal_R = {id(d): d.granulate(X_ref) for d in dets_R}
    Gcal_E = {id(d): d.granulate(X_ref) for d in dets_E}
    evs_R, thR, (tqR, TqR) = channel_scan(dets_R, Gcal_R, {id(d): d.granulate(X_full) for d in dets_R}, "R",
                                          cal_end=n_ref, scan_start=n_ref, scan_end=len(X_full),
                                          selfscan_start=0, gap=4*64, spp=SPP)
    evs_E, thE, (tqE, TqE) = channel_scan(dets_E, Gcal_E, {id(d): d.granulate(X_full) for d in dets_E}, "E",
                                          cal_end=n_ref, scan_start=n_ref, scan_end=len(X_full),
                                          selfscan_start=0, gap=4*64, spp=SPP)
    grid = sorted(set(tqR.tolist()) | set(tqE.tolist()))
    Tmap = np.full(len(grid), -np.inf)
    for tq, Tq in ((tqR, TqR), (tqE, TqE)):
        pos = np.searchsorted(np.asarray(grid), tq)
        Tmap[pos] = np.maximum(Tmap[pos], Tq)
    curves["rdsg"] = evaluate_curve_ids(true_events, np.asarray(grid), Tmap, n_ref, t1t - n_ref)
    print(f"rdsg done {time.time()-t0:.0f}s", flush=True)

    # ---- Window-divergence class ----
    for name, bl in (("b1_corr", WindowDivergenceBaseline("corr", w=64, blk=16, spp=SPP, gap=64)),
                     ("b5b_pcorr", WindowDivergenceBaseline("pcorr", w=64, blk=16, spp=SPP, gap=64))):
        t0 = time.time()
        t_q, s_q = _series_from_stat(X, bl.w, bl.blk, bl._stat, n_ref, len(X))
        curves[name] = evaluate_curve_ids(true_events, t_q, s_q, n_ref, t1t - n_ref)
        print(name, f"{time.time()-t0:.0f}s", flush=True)

    # ---- Granger ----
    gb = GrangerBaseline([(i, j) for i in range(X_ref.shape[1]) for j in range(X_ref.shape[1]) if i != j][:16],
                         w=64, p=3, spp=SPP, gap=64)
    t_q, s_q = _series_from_stat(X, 64, 16, gb._stat, n_ref, len(X))
    curves["b2_granger"] = evaluate_curve_ids(true_events, t_q, s_q, n_ref, t1t - n_ref)
    print("b2_granger done", flush=True)

    # ---- IF / LOF ----
    for name, model in (("b4_if", "if"), ("b4_lof", "lof")):
        n = X_ref.shape[1]
        agg = np.full(len(X) - n_ref, -np.inf)
        for j in range(n):
            agg = np.maximum(agg, _fit_score(model, X_ref[:, j:j+1], X[n_ref:, j:j+1], 0))
        curves[name] = evaluate_curve_ids(true_events, np.arange(n_ref, len(X)), agg, n_ref, t1t - n_ref)
        print(name, "done", flush=True)

    # ---- Per-subset summary ----
    summary = {}
    for m, pts in curves.items():
        rel_ids = {i for i in range(n_ev) if sub_id[i] == "relational-only"}
        vis_ids = {i for i in range(n_ev) if sub_id[i] == "point-visible"}
        rel_hit, vis_hit = set(), set()
        for p in pts:
            hit = set(p["matched"])
            rel_hit |= hit & rel_ids
            vis_hit |= hit & vis_ids
        summary[m] = {
            "recall_at_fa5_all": recall_at_fa(pts, FA5),
            "max_recall_all": max(p["recall"] for p in pts),
            "relational_only_union_recall": len(rel_hit) / max(n_rel, 1),
            "point_visible_union_recall": len(vis_hit) / max(n_vis, 1),
            "rel_hit": sorted(rel_hit), "vis_hit": sorted(vis_hit),
        }
    res["curves"] = curves
    res["summary"] = summary
    json.dump(res, open(os.path.join(OUT, "e1c_gas_subset.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=float)

    print("\n=== decomposition summary (relational-only n=%d / point-visible n=%d) ===" % (n_rel, n_vis))
    for m in ("rdsg", "b4_lof", "b4_if", "b5b_pcorr", "b1_corr", "b2_granger"):
        s = summary[m]
        print(f"{m:12s} rel={s['relational_only_union_recall']:.2f} vis={s['point_visible_union_recall']:.2f} "
              f"max={s['max_recall_all']:.2f} @FA5={s['recall_at_fa5_all']:.2f}")
    print("saved e1c_gas_subset.json")


if __name__ == "__main__":
    main()
