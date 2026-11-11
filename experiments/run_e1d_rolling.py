# -*- coding: utf-8 -*-
"""W5: rolling recalibration prototype (Gas Pipeline, residual attack-surface
cleanup plan W5).

Design: split the test segment into 5 segments; before scanning segment k,
extend the reference to the union of [0, all previously alarm-free segments]
and re-run detector.calibrate (re-estimating mu/sigma; block permutation as
before); alarm-raising segments are not merged into the reference.
Control = the original static single-calibration pipeline (E1 unchanged).
Output: (FA, recall) curves for both variants over the same test-quantile
sweep + recall in the FA<=5/day region.
Branches: (a) rolling improves recall by >=20% relative in the FA<=5 region
-> "working prototype"; (b) otherwise -> report honestly that naive
EWMA/segment recalibration does not transfer.
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
from baselines.relational import merge_events
from run_e0_syn2 import channel_scan, _unit_stack, pick_theta, BLK
from run_paper_e1 import SPP

OUT = os.path.normpath(os.path.join(_results_root(), "paper"))
N_SEG = 5


def build_dets(X_ref, n_ch):
    dets_R, dets_E = [], []
    for K in (4, 2):
        for w, blk in ((128, 16), (64, 16)):
            for strata, lst in (((12, 0), dets_R), ((0, 16), dets_E)):
                d = RDSGDetector(K=K, w=w, blk=blk, B=199, q=0.05, seed=1,
                                 pair_strata=strata, n_subset=4)
                d.fit(X_ref, channels=list(range(n_ch)), taus=(0, 1, -1))
                d.channel_kind_ = "R" if strata[0] > 0 else "E"
                lst.append(d)
    return dets_R, dets_E


def score_segment(dets, G, cal_end, scan_start, scan_end, selfscan_start, gap):
    """Scan one segment given a calibration endpoint; return (events,
    per-boundary alarm flags, grid, alarm strength)."""
    for d in dets:
        d.calibrate(G[id(d)], cal_end, cal_t0=0)
        d.prune_units_by_scan(G[id(d)], selfscan_start, scan_start, z_cap=5.5)
    comps = sorted(set(dets[0].components).intersection(*[set(d.components) for d in dets]))
    fa_c = 0.25 / len(comps)
    grid_r, Zr = _unit_stack(dets, G, selfscan_start, scan_start)
    thetas = {}
    for c in comps:
        Tc = Zr[:, :, comps.index(c)].max(axis=1)
        thetas[c], _ = pick_theta(Tc, grid_r, fa_c, SPP, gap=gap)
    grid_q, Zq = _unit_stack(dets, G, scan_start, scan_end)
    alarm = np.zeros(len(grid_q), bool)
    for c in comps:
        Tc_q = Zq[:, :, comps.index(c)].max(axis=1)
        alarm |= Tc_q > thetas[c]
    from baselines.relational import merge_events
    evs = merge_events(grid_q, alarm, gap)
    return evs, alarm, grid_q, Zq.max(axis=(1, 2)), comps, thetas


def curve_from_stream(true_events, times, scores, t_offset, test_len, gap=64):
    pts = []
    days = test_len * SPP / 86400.0
    fin = scores[np.isfinite(scores)]
    for q in np.linspace(0.90, 0.999, 25):
        th = float(np.quantile(fin, q))
        evs = merge_events(times, scores > th, gap)
        pr = [(int(a - t_offset), int(b - t_offset)) for a, b in evs]
        matches, delays, falses = match_events([(a, b, None) for a, b, _ in true_events], pr)
        matched = sum(1 for _, p in matches if p is not None)
        pts.append(dict(quantile=float(q), recall=matched / max(len(true_events), 1),
                        fa_per_day=len(falses) / max(days, 1e-9)))
    return pts


def main():
    X, spp, meta = load_gas_pipeline()
    r0, r1 = meta["ref"]
    n_ref = r1 - r0
    X_full = X
    t1t = meta["test"][1]
    true_events = [(e[0], e[1], e[2]) for e in meta["events"]]
    X_ref0 = X[r0:r1]
    n_ch = X_full.shape[1]
    print(f"ref={n_ref} test={t1t-n_ref} events={len(true_events)}", flush=True)

    # ---- static (E1 unchanged) ----
    t0 = time.time()
    dR_s, dE_s = build_dets(X_ref0, n_ch)
    from run_e0_syn2 import channel_scan as cs
    Gcal_R = {id(d): d.granulate(X_ref0) for d in dR_s}
    Gcal_E = {id(d): d.granulate(X_ref0) for d in dE_s}
    Gs_all_R = {id(d): d.granulate(X_full) for d in dR_s}
    Gs_all_E = {id(d): d.granulate(X_full) for d in dE_s}
    evs_R, thR, (tqR, TqR) = cs(dR_s, Gcal_R, Gs_all_R, "R", cal_end=n_ref,
                                scan_start=n_ref, scan_end=len(X_full),
                                selfscan_start=0, gap=4 * 64, spp=SPP)
    evs_E, thE, (tqE, TqE) = cs(dE_s, Gcal_E, Gs_all_E, "E", cal_end=n_ref,
                                scan_start=n_ref, scan_end=len(X_full),
                                selfscan_start=0, gap=4 * 64, spp=SPP)
    grid = sorted(set(tqR.tolist()) | set(tqE.tolist()))
    Tmap = np.full(len(grid), -np.inf)
    for tq, Tq in ((tqR, TqR), (tqE, TqE)):
        pos = np.searchsorted(np.asarray(grid), tq)
        Tmap[pos] = np.maximum(Tmap[pos], Tq)
    curve_static = curve_from_stream(true_events, np.asarray(grid), Tmap, n_ref, t1t - n_ref)
    print(f"static done {time.time()-t0:.0f}s", flush=True)

    # ---- rolling (per-segment recalibration) ----
    t0 = time.time()
    dR_r, dE_r = build_dets(X_ref0, n_ch)
    bounds = np.linspace(n_ref, len(X_full), N_SEG + 1).astype(int)
    cal_end = n_ref
    all_times, all_scores = [], []
    seg_stats = []
    for k in range(N_SEG):
        s0, s1 = int(bounds[k]), int(bounds[k + 1])
        G_R = {id(d): d.granulate(X_full) for d in dR_r}
        G_E = {id(d): d.granulate(X_full) for d in dE_r}
        Gs_R, Gs_E = G_R, G_E
        Gcal_R, Gcal_E = G_R, G_E
        # Note: the granule sequences cover the full range (G includes the test
        # segment); the calibration endpoint cal_end only controls the mu/sigma
        # estimation window
        for d in dR_r:
            d.calibrate(Gcal_R[id(d)], cal_end, cal_t0=0)
            d.prune_units_by_scan(Gcal_R[id(d)], 0, s0 if k > 0 else cal_end, z_cap=5.5)
        from run_e0_syn2 import pick_theta
        comps = sorted(set(dR_r[0].components).intersection(*[set(d.components) for d in dR_r]))
        grid_r, Zr = _unit_stack(dR_r, Gcal_R, 0, cal_end)
        thetasR = {}
        for c in comps:
            Tc = Zr[:, :, comps.index(c)].max(axis=1)
            thetasR[c], _ = pick_theta(Tc, grid_r, 0.25 / len(comps), SPP, gap=4 * 64)
        gq, Zq = _unit_stack(dR_r, Gs_R, s0, s1)
        alarm = np.zeros(len(gq), bool)
        for c in comps:
            alarm |= Zq[:, :, comps.index(c)].max(axis=1) > thetasR[c]
        from baselines.relational import merge_events
        evsR = merge_events(gq, alarm, 4 * 64)
        # E channel handled the same way
        for d in dE_r:
            d.calibrate(Gcal_E[id(d)], cal_end, cal_t0=0)
            d.prune_units_by_scan(Gcal_E[id(d)], 0, s0 if k > 0 else cal_end, z_cap=5.5)
        grid_rE, ZrE = _unit_stack(dE_r, Gcal_E, 0, cal_end)
        compsE = sorted(set(dE_r[0].components).intersection(*[set(d.components) for d in dE_r]))
        thetasE = {}
        for c in compsE:
            Tc = ZrE[:, :, compsE.index(c)].max(axis=1)
            thetasE[c], _ = pick_theta(Tc, grid_rE, 0.25 / len(compsE), SPP, gap=4 * 64)
        gqE, ZqE = _unit_stack(dE_r, Gs_E, s0, s1)
        alarmE = np.zeros(len(gqE), bool)
        for c in compsE:
            alarmE |= ZqE[:, :, compsE.index(c)].max(axis=1) > thetasE[c]
        evsE = merge_events(gqE, alarmE, 4 * 64)
        # Concatenate score streams (R/E max) + merge alarm-free segments into reference
        gset = sorted(set(gq.tolist()) | set(gqE.tolist()))
        smap = np.full(len(gset), -np.inf)
        for gq_x, Zq_x in ((gq, Zq.max(axis=(1, 2))), (gqE, ZqE.max(axis=(1, 2)))):
            pos = np.searchsorted(np.asarray(gset), gq_x)
            smap[pos] = np.maximum(smap[pos], Zq_x)
        all_times.append(np.asarray(gset))
        all_scores.append(smap)
        n_alarm_here = len(evsR) + len(evsE)
        if n_alarm_here == 0:
            cal_end = s1  # alarm-free segment merged into reference
        seg_stats.append(dict(seg=k, range=[int(s0), int(s1)], n_ev=n_alarm_here,
                              cal_end=int(cal_end)))
        print(f"  seg{k} [{s0},{s1}) ev={n_alarm_here} cal_end={cal_end} ({time.time()-t0:.0f}s)",
              flush=True)
    times_c = np.concatenate(all_times)
    scores_c = np.concatenate(all_scores)
    order = np.argsort(times_c, kind="stable")
    curve_rolling = curve_from_stream(true_events, times_c[order], scores_c[order],
                                      n_ref, t1t - n_ref)

    def at_fa(curve, cap):
        c = [p for p in curve if p["fa_per_day"] <= cap]
        return max((p["recall"] for p in c), default=0.0)

    summary = dict(static=dict(recall_at_fa5=at_fa(curve_static, 5.0),
                               max_recall=max(p["recall"] for p in curve_static)),
                   rolling=dict(recall_at_fa5=at_fa(curve_rolling, 5.0),
                                max_recall=max(p["recall"] for p in curve_rolling)),
                   segments=seg_stats)
    rel = None
    if summary["static"]["recall_at_fa5"] > 0:
        rel = summary["rolling"]["recall_at_fa5"] / summary["static"]["recall_at_fa5"] - 1
    summary["relative_gain_at_fa5"] = None if rel is None else round(rel, 3)
    out = dict(curve_static=curve_static, curve_rolling=curve_rolling, summary=summary)
    json.dump(out, open(os.path.join(OUT, "e1d_rolling.json"), "w"), indent=1, default=float)
    print(json.dumps(summary, indent=1, default=float))
    print("saved e1d_rolling.json")


if __name__ == "__main__":
    main()
