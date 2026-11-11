# -*- coding: utf-8 -*-
"""WADI real-attack runner: RDSG + baselines on WADI A2.

Operating curves (recall vs FA/day) with event-level matching,
mirroring the gas-pipeline protocol (run_paper_e1.py).
Output: results/paper/wadi_curves.json
"""
import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.normpath(os.path.join(HERE, ".."))
PROJECT = os.path.normpath(os.path.join(CODE, ".."))
sys.path.insert(0, CODE)
sys.path.insert(0, HERE)

from rdsg.metrics import event_metrics, match_events
from baselines.relational import (WindowDivergenceBaseline, VARResidualBaseline,
                                  GrangerBaseline, MIGraphBaseline, merge_events,
                                  pick_theta_split)
from baselines.point_agg import _fit_score
from run_e0_syn2 import channel_scan, _merge_alarms, pick_theta, _unit_stack
from run_paper_e2 import build_channel, lengths_gap, PAIR_STRATA_R, PAIR_STRATA_E
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


OUT = os.path.join(_results_root(), "paper", "wadi_curves.json")
FA_TARGET = 0.25
QGRID = np.linspace(0.95, 0.99999, 30)


def load_wadi_local():
    """Load WADI A2 from pre-processed npy.

    Location: $WADI_NPY_DIR if set, else ./datasets/WADI_A2 relative to the
    project root (see README_ANONYMOUS.md for the expected files: ref.npy,
    test.npy, test_label.npy derived from the official WADI A2 release)."""
    npy_dir = os.environ.get("WADI_NPY_DIR",
                             os.path.join(PROJECT, "datasets", "WADI_A2"))
    ref = np.load(os.path.join(npy_dir, "ref.npy"), allow_pickle=True)
    test = np.load(os.path.join(npy_dir, "test.npy"), allow_pickle=True)
    label = np.load(os.path.join(npy_dir, "test_label.npy"), allow_pickle=True).ravel()
    variances = np.nanvar(ref, axis=0)
    ok = variances > np.percentile(variances, 10)
    ok_idx = np.where(ok)[0]
    ranked = ok_idx[np.argsort(-variances[ok_idx])][:40]
    ranked = np.sort(ranked)
    ref_sel = ref[-12000:, ranked].astype(float)
    test_sel = test[:, ranked].astype(float)
    from scipy.interpolate import interp1d
    for arr in (ref_sel, test_sel):
        for j in range(arr.shape[1]):
            col = arr[:, j]
            bad = np.isnan(col)
            if bad.any() and (~bad).sum() > 2:
                f = interp1d(np.where(~bad)[0], col[~bad],
                             bounds_error=False, fill_value="extrapolate")
                col[bad] = f(np.where(bad)[0])
    X = np.vstack([ref_sel, test_sel])
    events = []
    in_a, s = False, 0
    for i in range(len(label)):
        if label[i] == 1 and not in_a:
            in_a, s = True, i
        elif label[i] == 0 and in_a:
            in_a = False
            events.append((s, i, "attack"))
    if in_a:
        events.append((s, len(label), "attack"))
    return X, 1.0, {"name": "WADI", "channels": len(ranked), "events": events,
                    "n_events": len(events), "n_ref": len(ref_sel)}


def evaluate_curve(true_events, times, scores, quantiles, n_ref, test_len, spp, gap):
    pts = []
    for q in quantiles:
        th = np.quantile(scores[0], q)
        evs = merge_events(np.asarray(times), np.asarray(scores[1]) > th, gap)
        pr = [(int(a - n_ref), int(b - n_ref), None) for a, b in evs]
        matches, _, _ = match_events([(a, b, t) for a, b, t in true_events],
                                     [(a, b) for a, b, _ in pr])
        rec = sum(1 for _, p in matches if p is not None) / max(len(true_events), 1)
        # FA: count events outside true events
        n_fa = sum(1 for a, b, _ in pr
                   if not any(a < te[1] and b > te[0] for te in true_events))
        days = test_len * spp / 86400.0
        pts.append(dict(quantile=float(q), theta=float(th), recall=rec, fa_per_day=n_fa / days))
    return pts


def main():
    t_start = time.time()
    X, spp, meta = load_wadi_local()
    n_ref = meta["n_ref"]
    n_test = X.shape[0] - n_ref
    true_events = [(e[0], e[1], e[2]) for e in meta["events"]]
    print(f"WADI: {X.shape} spp={spp} ref={n_ref} test={n_test} events={len(true_events)}")
    print(f"True events: {[(e[0], e[1], e[1]-e[0]) for e in true_events[:5]]}")

    taus = (0, 1, -1)
    gap_ours = 512  # 512 points at 1s = 8.5 min

    # ---- RDSG family ----
    print("\n--- RDSG family ---", flush=True)
    planner = build_channel(X[:n_ref], taus, strata=(16, 24))[0]
    force_R, force_E = planner.strong_pairs_, planner.weak_pairs_
    comp_series = []
    for strata, n_sub in ((PAIR_STRATA_R, None), (PAIR_STRATA_E, 6)):
        dd = build_channel(X[:n_ref], taus, strata=strata, n_subset=n_sub,
                           force_pairs=(force_R, force_E))
        Gcal = {id(d): d.granulate(X[:n_ref]) for d in dd}
        Gs = {id(d): d.granulate(X) for d in dd}
        evs, thetas, _ = channel_scan(dd, Gcal, Gs, "scan", cal_end=n_ref // 2,
                                      scan_start=n_ref, scan_end=len(X),
                                      gap=gap_ours, spp=spp)
        comps = sorted(set(dd[0].components).intersection(*[set(d.components) for d in dd]))
        grid_q, Zu = _unit_stack(dd, Gs, n_ref, len(X))
        Tc = {c: Zu[:, :, ci].max(axis=1) for ci, c in enumerate(comps)}
        comp_series.append((np.asarray(grid_q), Tc, thetas))
    print(f"  RDSG channels built ({time.time()-t_start:.0f}s)", flush=True)

    # RDSG curve: sweep multiplier on theta
    MULTS = (0.3, 0.5, 0.7, 0.85, 1.0, 1.3, 1.7, 2.5, 4.0)
    rdsg_pts = []
    for m in MULTS:
        pr = []
        for grid_q, Tc, thetas in comp_series:
            alarm = np.zeros(len(grid_q), bool)
            for c, s in Tc.items():
                alarm |= s > m * thetas[c]
            for a, b in _merge_alarms(grid_q, alarm, gap=gap_ours):
                pr.append((int(a - n_ref), int(b - n_ref), None))
        matches, _, _ = match_events(true_events, [(a, b) for a, b, _ in pr])
        rec = sum(1 for _, p in matches if p is not None) / max(len(true_events), 1)
        n_fa = sum(1 for a, b, _ in pr
                   if not any(a < te[1] and b > te[0] for te in true_events))
        rdsg_pts.append(dict(quantile=m, recall=rec, fa_per_day=n_fa / (n_test * spp / 86400)))
    print(f"  RDSG curve: {len(rdsg_pts)} pts, max recall={max(p['recall'] for p in rdsg_pts):.3f}")

    # ---- Baselines ----
    curves = {"rdsg": rdsg_pts}
    X_ref, X_test = X[:n_ref], X[n_ref:]

    # Point baselines (IF/LOF)
    for name, mdl in (("b4_if", "if"), ("b4_lof", "lof")):
        print(f"  {name}...", end="", flush=True)
        ref_agg = np.full(n_ref, -np.inf)
        half = n_ref // 2
        for j in range(X_ref.shape[1]):
            ref_agg[:half] = np.maximum(ref_agg[:half],
                _fit_score(mdl, X_ref[half:, j:j+1], X_ref[:half, j:j+1], 1))
            ref_agg[half:] = np.maximum(ref_agg[half:],
                _fit_score(mdl, X_ref[:half, j:j+1], X_ref[half:, j:j+1], 1))
        agg = np.full(n_test, -np.inf)
        for j in range(X_ref.shape[1]):
            agg = np.maximum(agg, _fit_score(mdl, X_ref[:, j:j+1], X_test[:, j:j+1], 1))
        pts = evaluate_curve(true_events, np.arange(n_ref, len(X)),
                            (ref_agg, agg), QGRID, n_ref, n_test, spp, 128)
        curves[name] = pts
        print(f" max={max(p['recall'] for p in pts):.3f}")

    # Relational baselines
    for name, bl in (("b5b_pcorr", WindowDivergenceBaseline("pcorr", spp=spp, fa_target=FA_TARGET)),
                     ("b5c_var", VARResidualBaseline(spp=spp, fa_target=FA_TARGET)),
                     ("b2_granger", GrangerBaseline(force_R[:12], spp=spp, fa_target=FA_TARGET)),
                     ("b5a_mi", MIGraphBaseline(spp=spp, fa_target=FA_TARGET))):
        print(f"  {name}...", end="", flush=True)
        try:
            t_r, s_r, t_q, s_q, gap = bl.run_scores(X_ref, X_test)
            pts = evaluate_curve(true_events, t_q, (s_r, s_q), QGRID, n_ref, n_test, spp, gap)
            curves[name] = pts
            print(f" max={max(p['recall'] for p in pts):.3f}")
        except Exception as e:
            # record the failure in the output instead of silently writing an
            # empty curve: an empty list would read as "no alarms" downstream
            import traceback
            print(f" FAILED: {e}")
            traceback.print_exc()
            curves[name] = {"error": repr(e)}

    json.dump(curves, open(OUT, "w"), indent=1)
    print(f"\n{'='*50}")
    print(f"WADI done ({time.time()-t_start:.0f}s), saved {OUT}")
    for m, pts in curves.items():
        if pts:
            best = max(p["recall"] for p in pts)
            lo_fa = [p for p in pts if p["fa_per_day"] <= 5]
            lo_best = max(p["recall"] for p in lo_fa) if lo_fa else None
            lo_txt = f"{lo_best:.3f}" if lo_best is not None else "NOT-EVALUABLE (no grid point <=5 FA/day)"
            print(f"  {m:12s} max_recall={best:.3f} @<=5FA/d={lo_txt}")


if __name__ == "__main__":
    main()
