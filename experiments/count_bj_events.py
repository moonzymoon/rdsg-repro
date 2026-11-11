# -*- coding: utf-8 -*-
"""Count predicted events per seed on Beijing: full family vs gamma-only.

Verifies the main-text sentence 'predicts fewer events than its gamma-only
ablation on nearly every seed (M versus N at the median)'. Deterministic
rerun of the two frozen arms (same specs, same pipeline, counting len(pred)).
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.normpath(os.path.join(HERE, ".."))
PROJECT = os.path.normpath(os.path.join(CODE, ".."))
sys.path.insert(0, CODE)
sys.path.insert(0, HERE)

from rdsg.datasets import load_beijing_air
from rdsg import inject
from run_paper_e2 import (LEN_BY_SPP, build_channel, make_specs,
                          ours_pipeline, PAIR_STRATA_E)
import statistics
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



def main():

    X, spp, meta = load_beijing_air()
    X_ref, X_test = X[:12000], X[12000:12000 + 18000]
    lengths = LEN_BY_SPP[spp]
    taus = (0, 1, -1)
    planner = build_channel(X_ref, taus, strata=(16, 24))[0]
    force_R, force_E = planner.strong_pairs_, planner.weak_pairs_
    dE0 = build_channel(X_ref, taus, strata=PAIR_STRATA_E, n_subset=6,
                        force_pairs=(force_R, force_E))[0]

    cnt = {"rdsg": [], "abl_gamma": []}
    for seed in range(1, 11):
        specs = make_specs(planner, dE0, X_ref, X_test, seed, lengths)
        X_test_inj = inject(X_test, specs, seed=seed)
        pred, _ = ours_pipeline(X_ref, X_test_inj, X_test, taus, spp=spp,
                                force_R=force_R, force_E=force_E)
        cnt["rdsg"].append(len(pred))
        pred_g, _ = ours_pipeline(X_ref, X_test_inj, X_test, taus,
                                  comps=("dgamma",), spp=spp,
                                  force_R=force_R, force_E=force_E)
        cnt["abl_gamma"].append(len(pred_g))
        print(f"seed {seed}: rdsg={len(pred)} gamma={len(pred_g)}", flush=True)

    print("\nrdsg median:", statistics.median(cnt["rdsg"]), cnt["rdsg"])
    print("gamma median:", statistics.median(cnt["abl_gamma"]), cnt["abl_gamma"])
    fewer = sum(1 for a, b in zip(cnt["rdsg"], cnt["abl_gamma"]) if a < b)
    print(f"rdsg fewer than gamma on {fewer}/10 seeds")
    json.dump(cnt, open(os.path.join(_results_root(), "paper",
                                     "beijing_event_counts.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
