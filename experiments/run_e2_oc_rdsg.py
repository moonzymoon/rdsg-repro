# -*- coding: utf-8 -*-
"""Recompute the RDSG operating curve via the PRODUCTION pipeline.

Sweeps the requested FA budget (run_paper_e2.FA_TARGET) over a grid; at each
value the full ours_pipeline runs (calibration, per-channel merge, typing),
guaranteeing exact parity with the main-table protocol. Updates e2_oc.json
in place (rdsg curves + matched summary for all datasets).
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

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, CODE)
sys.path.insert(0, HERE)
from rdsg.datasets import load_metr_la, load_pems_bay, load_beijing_air
from rdsg.metrics import event_metrics
import run_paper_e2 as R
import run_e0_syn2 as E0
from run_paper_e2 import LEN_BY_SPP, build_channel, make_specs, PAIR_STRATA_E
from rdsg import inject

OUT = os.path.join(_results_root(), "paper", "e2_oc.json")
FA_GRID = (0.06, 0.10, 0.15, 0.20, 0.25, 0.40, 0.60, 1.00)
FA_TARGETS = (0.25, 0.5)


def matched(points, fa_t):
    ok = [r for f, r in points if f <= fa_t + 1e-9]
    return max(ok) if ok else None


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all")
    args = ap.parse_args()
    loaders = {"metr_la": load_metr_la, "pems_bay": load_pems_bay,
               "beijing_air": load_beijing_air}
    oc = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
    orig_fa = R.FA_TARGET
    orig_fa_e0 = E0.FA_TARGET
    try:
        for key, loader in loaders.items():
            if args.dataset != "all" and key != args.dataset:
                continue
            X, spp, meta = loader()
            X_ref, X_test = X[:12000], X[12000:12000 + 18000]
            lengths = LEN_BY_SPP[spp]
            taus = (0, 1, -1)
            planner = build_channel(X_ref, taus, strata=(16, 24))[0]
            force_R, force_E = planner.strong_pairs_, planner.weak_pairs_
            dR0 = planner
            dE0 = build_channel(X_ref, taus, strata=PAIR_STRATA_E, n_subset=6,
                                force_pairs=(force_R, force_E))[0]
            curves = {}
            for seed in range(1, 11):
                specs = make_specs(dR0, dE0, X_ref, X_test, seed, lengths)
                X_test_inj = inject(X_test, specs, seed=seed)
                true_events = [(sp["start"], sp["start"] + sp["length"], sp["type"])
                               for sp in specs]
                for fa in FA_GRID:
                    # channel_scan resolves its default budget from run_e0_syn2.FA_TARGET
                    R.FA_TARGET = fa
                    E0.FA_TARGET = fa
                    pred, _ = R.ours_pipeline(X_ref, X_test_inj, X_test, taus, spp=spp,
                                              force_R=force_R, force_E=force_E)
                    m = event_metrics(true_events, pred, len(X_test), spp)
                    curves.setdefault(str(seed), []).append([m["fa_per_day"], m["recall"]])
                print(f"[{key} s{seed}] rdsg(prod) {len(FA_GRID)} pts", flush=True)
            oc.setdefault(key, {}).setdefault("curves", {})["rdsg"] = curves
            summ = {}
            for fa_t in FA_TARGETS:
                vals = [matched([tuple(p) for p in pts], fa_t) for pts in curves.values()]
                vals = [v for v in vals if v is not None]
                if vals:
                    summ[f"matched_{fa_t}"] = round(float(np.mean(vals)), 3)
            oc[key]["matched"]["rdsg"] = summ
            json.dump(oc, open(OUT, "w"), indent=1)
            print(f"[{key}] rdsg matched: {summ}", flush=True)
    finally:
        R.FA_TARGET = orig_fa
        E0.FA_TARGET = orig_fa_e0
    print("updated", OUT)


if __name__ == "__main__":
    main()
