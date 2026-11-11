# -*- coding: utf-8 -*-
"""F1 instrumented re-run: METR-LA component-trigger attribution
(F1+F3 approved by the coordinator, 2026-10-16).

Identical configuration and seeds to run_e2d_attr.py; additionally saves
per-event triggering-component counts (number of R-channel dgamma/ddelta
boundaries crossing threshold, E-channel attribution). Zero changes to the
method library.
Reconciliation requirement: the (seed, idx, truth, pred) of each event must
match e2d_attr.json exactly (same source, 63 events).
Output: results/paper/e2d_component.json
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
from rdsg import inject
from rdsg.datasets import load_metr_la
from rdsg.metrics import match_events
from run_e0_syn2 import _unit_stack, pick_theta, BLK
from run_paper_e2 import build_channel, make_specs, PAIR_STRATA_E, LEN_BY_SPP
def merge_events(times, on, gap):
    # Same semantics as run_e2d_attr._merge_alarms: event end = prev + BLK
    # (prerequisite for reconciliation)
    idx = np.where(on)[0]
    if len(idx) == 0:
        return []
    evs, t0, prev = [], times[idx[0]], times[idx[0]]
    for i in idx[1:]:
        if times[i] - prev > gap:
            evs.append((t0, prev + BLK))
            t0 = times[i]
        prev = times[i]
    evs.append((t0, prev + BLK))
    return evs

OUT = os.path.normpath(os.path.join(_results_root(), "paper"))


def instrumented_channel(dd, Gcal, Gs, cal_end, scan_start, scan_end, selfscan_start, spp, gap):
    for d in dd:
        d.calibrate(Gcal[id(d)], cal_end, cal_t0=0)
        d.prune_units_by_scan(Gcal[id(d)], selfscan_start, scan_start, z_cap=5.5)
    comps = sorted(set(dd[0].components).intersection(*[set(d.components) for d in dd]))
    fa_c = 0.25 / len(comps)
    grid_r, Zr = _unit_stack(dd, Gcal, selfscan_start, scan_start)
    thetas = {}
    for c in comps:
        Tc = Zr[:, :, comps.index(c)].max(axis=1)
        thetas[c], _ = pick_theta(Tc, grid_r, fa_c, spp, gap=gap)
    grid_q, Zq = _unit_stack(dd, Gs, scan_start, scan_end)
    trig = np.zeros((len(grid_q), len(comps)), bool)
    for ci, c in enumerate(comps):
        trig[:, ci] = Zq[:, :, comps.index(c)].max(axis=1) > thetas[c]
    alarm = trig.any(axis=1)
    evs = merge_events(grid_q, alarm, gap)
    return evs, comps, trig, grid_q


def main():
    taus = (0, 1, -1)
    X, spp, meta = load_metr_la()
    X_ref, X_test = X[:12000], X[12000:12000 + 18000]
    lengths = LEN_BY_SPP[spp]
    planner = build_channel(X_ref, taus, strata=(16, 24))[0]
    force_R, force_E = planner.strong_pairs_, planner.weak_pairs_
    dE0 = build_channel(X_ref, taus, strata=PAIR_STRATA_E, n_subset=6,
                        force_pairs=(force_R, force_E))[0]
    res = {"seeds": {}}
    for seed in range(1, 11):
        t0 = time.time()
        specs = make_specs(planner, dE0, X_ref, X_test, seed, lengths)
        X_inj = inject(X_test, specs, seed=seed)
        true_events = [(sp["start"], sp["start"] + sp["length"], sp["type"]) for sp in specs]
        dR = build_channel(X_ref, taus, strata=(16, 0))
        dE = build_channel(X_ref, taus, strata=(0, 24), n_subset=6,
                           force_pairs=(force_R, force_E))
        Gcal_R = {id(d): d.granulate(X_ref) for d in dR}
        Gcal_E = {id(d): d.granulate(X_ref) for d in dE}
        Gs_R = {id(d): d.granulate(np.vstack([X_ref, X_inj])) for d in dR}
        Gs_E = {id(d): d.granulate(np.vstack([X_ref, X_inj])) for d in dE}
        evs_R, compsR, trigR, gridR = instrumented_channel(
            dR, Gcal_R, Gs_R, len(X_ref), len(X_ref), len(X_ref) + len(X_test), 0, spp, 4 * 64)
        evs_E, compsE, trigE, gridE = instrumented_channel(
            dE, Gcal_E, Gs_E, len(X_ref), len(X_ref), len(X_ref) + len(X_test), 0, spp, 4 * 64)
        true3 = [(a, b, None) for a, b, _ in true_events]
        tidx = {(int(a), int(b)): k for k, (a, b, _t) in enumerate(true_events)}
        rows = []
        # R-channel events: component counts + pred (same rule as e2d:
        # dominant -> brk, otherwise inv)
        for (a, b) in evs_R:
            m = (gridR >= a) & (gridR < b)
            counts = {c: int(trigR[m][:, i].sum()) for i, c in enumerate(compsR)}
            dom = max(counts, key=counts.get)
            rows.append(dict(chan="R", start=int(a) - len(X_ref), end=int(b) - len(X_ref),
                             counts=counts, pred="brk" if dom == "dgamma" else "inv"))
        for (a, b) in evs_E:
            m = (gridE >= a) & (gridE < b)
            counts = {c: int(trigE[m][:, i].sum()) for i, c in enumerate(compsE)}
            rows.append(dict(chan="E", start=int(a) - len(X_ref), end=int(b) - len(X_ref),
                             counts=counts, pred="emb"))
        # Match against ground truth (same match_events greedy as e2d)
        pred3 = [(r["start"], r["end"], None) for r in rows]
        matches, _, _ = match_events(true3, pred3)
        pmap = {(int(p[0]), int(p[1])): r for p, r in zip(pred3, rows)}
        out_events = []
        for tr, p in matches:
            k = tidx[(int(tr[0]), int(tr[1]))]
            truth = true_events[k][2]
            if p is None:
                out_events.append(dict(idx=k, truth=truth, pred=None, chan=None, counts=None))
            else:
                r = pmap[(int(p[0]), int(p[1]))]
                out_events.append(dict(idx=k, truth=truth, pred=r["pred"], chan=r["chan"],
                                       counts=r["counts"]))
        res["seeds"][str(seed)] = dict(events=out_events)
        print(f"[s{seed}] events={len(out_events)} ({time.time()-t0:.0f}s)", flush=True)
        json.dump(res, open(os.path.join(OUT, "e2d_component.json"), "w"),
                  ensure_ascii=False, indent=1)
    # Summary: truth x triggering-component (R-delta / R-dgamma / E) matrix
    C = {}
    for s, v in res["seeds"].items():
        for e in v["events"]:
            if e["pred"] is None:
                continue
            if e["chan"] == "E":
                col = "E"
            else:
                col = "ddelta" if e["pred"] == "inv" else "dgamma"
            C.setdefault(e["truth"], {}).setdefault(col, 0)
            C[e["truth"]][col] += 1
    res["matrix_truth_x_component"] = C
    json.dump(res, open(os.path.join(OUT, "e2d_component.json"), "w"),
              ensure_ascii=False, indent=1)
    print("matrix:", json.dumps(C))
    print("saved e2d_component.json")


if __name__ == "__main__":
    main()
