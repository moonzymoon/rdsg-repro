# -*- coding: utf-8 -*-
"""E2abl: ablation pipeline re-run (saving full preds -> unified pooled
by-type accounting, 2026-10-06).

Background: recall_by_type matches independently within each type subset
(preds can be counted multiple times, favouring the greedy matcher) and
systematically overstates by-type recall; rdsg already has detail (pooled),
but the ablations have no events to recompute from. This script re-runs the
three ablation pipelines on METR-LA and PEMS-BAY and outputs pooled by-type
numbers. Pre-registration: the inversion comparison cited in the main text is
the pooled result from this script.
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
from rdsg.datasets import load_metr_la, load_pems_bay
from rdsg.metrics import match_events
from run_paper_e2 import (build_channel, make_specs, ours_pipeline, PAIR_STRATA_E,
                          LEN_BY_SPP, SEEDS)

OUT = os.path.normpath(os.path.join(_results_root(), "paper"))
ABL = (("abl_gamma", ("dgamma",)), ("abl_gd", ("dgamma", "ddelta")), ("abl_gj", ("dgamma", "djs")))
LOADERS = {"metr_la": load_metr_la, "pems_bay": load_pems_bay}


def main():
    taus = (0, 1, -1)
    out = {}
    path = os.path.join(OUT, "e2abl_pooled.json")
    if os.path.exists(path):
        out = json.load(open(path, encoding="utf-8"))
    for key, loader in LOADERS.items():
        X, spp, meta = loader()
        X_ref, X_test = X[:12000], X[12000:12000 + 18000]
        lengths = LEN_BY_SPP[spp]
        planner = build_channel(X_ref, taus, strata=(16, 24))[0]
        force_R, force_E = planner.strong_pairs_, planner.weak_pairs_
        dE0 = build_channel(X_ref, taus, strata=PAIR_STRATA_E, n_subset=6,
                            force_pairs=(force_R, force_E))[0]
        out.setdefault(key, {})
        for seed in SEEDS:
            if str(seed) in out[key].get("abl_gamma", {}):
                continue
            specs = make_specs(planner, dE0, X_ref, X_test, seed, lengths)
            X_test_inj = inject(X_test, specs, seed=seed)
            true_events = [(sp["start"], sp["start"] + sp["length"], sp["type"]) for sp in specs]
            for name, comps in ABL:
                t0 = time.time()
                pred, _rt = ours_pipeline(X_ref, X_test_inj, X_test, taus, comps=comps,
                                          spp=spp, force_R=force_R, force_E=force_E)
                matches, _, _ = match_events(true_events, pred)
                rows = {}
                for (t0e, t1e, t2), p in matches:
                    rows.setdefault(t2, [0, 0])
                    rows[t2][1] += 1
                    rows[t2][0] += 1 if p is not None else 0
                rec = {t: dict(pooled_recall=(h / n if n else None), n=n) for t, (h, n) in rows.items()}
                out[key].setdefault(name, {})[str(seed)] = rec
                print(f"[{key} s{seed}] {name}: " +
                      " ".join(f"{t}={v['pooled_recall']:.2f}({v['n']})" for t, v in rec.items()),
                      flush=True)
            json.dump(out, open(path, "w", encoding="utf-8"), indent=1, default=float)
    print("saved e2abl_pooled.json")


if __name__ == "__main__":
    main()
