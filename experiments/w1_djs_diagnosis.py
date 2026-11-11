# -*- coding: utf-8 -*-
"""W1 diagnosis: measured z of djs/ddelta units at the E0b reshape events vs
the family thresholds.

Problem: in E0b v1 (modal-pinned swap), full ties with gamma+delta. Distinguish
two causes:
  (i) the constructed djs signal is inherently small (insufficient |Delta JS|);
  (ii) djs has signal but is diluted by window contamination / family
  thresholds (z close to the threshold).
If (ii): lengthening the events can rescue it (pre-registered lengthened
re-run); if (i): the negative result is final.
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

import os, sys, json
import numpy as np

ROOT = _results_root()
import os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rdsg import inject
from rdsg.datasets import load_gas_pipeline  # noqa: F401 (interface consistency)
from run_e0_syn import make_synthetic
from run_e0_syn2 import (REF_LEN, TEST_LEN, PAIR_STRATA, build_configs,
                         channel_scan, _unit_stack, BLK)
from run_e0b_reshape import reshape_inject, sample_specs_rs

OUT = os.path.normpath(os.path.join(_results_root(), "e0"))
taus = list(range(-3, 4))


def main():
    seed = 1
    Xs = make_synthetic(REF_LEN + TEST_LEN, seed=100 + seed)
    X_ref, X_test = Xs[:REF_LEN], Xs[REF_LEN:]
    dets = build_configs(X_ref, taus)
    dets_R = build_configs(X_ref, taus, strata=(PAIR_STRATA[0], 0))
    dets_E = build_configs(X_ref, taus, strata=(0, PAIR_STRATA[1]))
    specs = sample_specs_rs(dets_R[0].strong_pairs_, 10, TEST_LEN, 600, seed,
                            lengths=(60, 120))
    X_inj = reshape_inject(X_test, X_ref, specs, seed)

    Gfull = np.vstack([X_ref, X_inj])
    Gcal = {id(d): d.granulate(X_ref) for d in dets_R + dets_E}
    Gs = {id(d): d.granulate(Gfull) for d in dets_R + dets_E}

    # Replicate channel_scan's thresholding logic (to obtain theta_c), then
    # take test-segment component-level z
    from run_e0_syn2 import pick_theta
    report = {}
    for ch, dd in (("R", dets_R), ("E", dets_E)):
        for d in dd:
            d.calibrate(Gcal[id(d)], REF_LEN // 2, cal_t0=0)
            d.prune_units_by_scan(Gcal[id(d)], REF_LEN // 2, REF_LEN, z_cap=5.5)
        comps = sorted(set(dd[0].components).intersection(*[set(x.components) for x in dd]))
        common = set(dd[0].edges_)
        for d in dd:
            common &= set(d.edges_)
        common = sorted(common)
        cpos = {e: k for k, e in enumerate(common)}
        grid_r, Zr = _unit_stack(dd, Gcal, REF_LEN // 2, REF_LEN)
        fa_c = 0.25 / len(comps)
        thetas = {}
        for c in comps:
            Tc = Zr[:, :, comps.index(c)].max(axis=1)
            thetas[c], _ = pick_theta(Tc, grid_r, fa_c, 60.0, gap=8 * BLK)
        grid_q, Zq = _unit_stack(dd, Gs, REF_LEN, REF_LEN + TEST_LEN)
        # Per injected event: per-boundary, per-(edge, comp) z within the
        # window, and whether it crosses the threshold
        ev_report = []
        for sp in specs:
            t0, t1 = REF_LEN + sp["start"], REF_LEN + sp["start"] + sp["length"]
            wmask = (grid_q >= t0 - BLK) & (grid_q < t1 + BLK)
            rows = np.where(wmask)[0]
            best = {}
            for ci, c in enumerate(comps):
                zc = Zq[rows][:, :, ci]
                zmax = float(zc.max())
                # djs reference value of the edge where zmax occurs
                bi, ei = np.unravel_index(np.argmax(zc), zc.shape)
                edge = common[ei]
                best[c] = dict(zmax=zmax, theta=thetas[c], edge=str(edge),
                               margin=zmax - thetas[c])
            ev_report.append(dict(type=sp["type"], length=sp["length"],
                                  start=sp["start"], best=best))
        report[ch] = dict(thetas=thetas, comps=comps, events=ev_report)
        print(f"== {ch} channel theta={ {k: round(v, 2) for k, v in thetas.items()} }")
        for e in ev_report[:6]:
            line = " ".join(f"{c}:z={e['best'][c]['zmax']:.2f}(θ={e['best'][c]['theta']:.2f},"
                            f"{'+' if e['best'][c]['margin'] > 0 else '-'})"
                            for c in report[ch]['comps'])
            print(f"  L={e['length']:3d} @{e['start']:6d}  {line}")
    json.dump(report, open(os.path.join(OUT, "w1_djs_diagnosis.json"), "w"),
              indent=1, default=float)
    print("saved w1_djs_diagnosis.json")


if __name__ == "__main__":
    main()
