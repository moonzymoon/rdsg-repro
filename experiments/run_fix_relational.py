# -*- coding: utf-8 -*-
"""Recompute the five relational baselines in e2_*.json with the corrected
rate-calibration span (pick_theta_split ts argument; see relational.py).

Only b1_corr / b2_granger / b5a_mi / b5b_pcorr / b5c_var are rewritten:
RDSG arms, point baselines and GDN are deterministic-or-expensive and remain
frozen. Specs per seed are regenerated identically to run_paper_e2 (same
make_specs seed), so true_events match the frozen runs exactly.
"""
import os

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

import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.normpath(os.path.join(HERE, ".."))
PROJECT = os.path.normpath(os.path.join(CODE, ".."))
sys.path.insert(0, CODE)
sys.path.insert(0, HERE)

from rdsg.datasets import load_metr_la, load_pems_bay, load_beijing_air
from rdsg.metrics import event_metrics
from run_paper_e2 import (LEN_BY_SPP, build_channel, make_specs, recall_by_type,
                          PAIR_STRATA_E, FA_TARGET)
from baselines.relational import (WindowDivergenceBaseline, VARResidualBaseline,
                                  GrangerBaseline, MIGraphBaseline)

OUT = os.path.join(_results_root(), "paper")
SEEDS = range(1, 11)
METHODS = ["b5a_mi", "b5b_pcorr", "b5c_var", "b1_corr", "b2_granger"]


def run_dataset(key, loader, taus=(0, 1, -1)):
    X, spp, meta = loader()
    X_ref, X_test = X[:12000], X[12000:12000 + 18000]
    lengths = LEN_BY_SPP[spp]
    print(f"== {meta['name']} {X.shape} spp={spp}", flush=True)
    planner = build_channel(X_ref, taus, strata=(16, 24))[0]
    force_R, force_E = planner.strong_pairs_, planner.weak_pairs_
    dE0 = build_channel(X_ref, taus, strata=PAIR_STRATA_E, n_subset=6,
                        force_pairs=(force_R, force_E))[0]
    out_path = os.path.join(OUT, f"e2_{key}.json")
    res = json.load(open(out_path, encoding="utf-8"))
    old = {m: json.loads(json.dumps(res.get(m, {}))) for m in METHODS}

    for seed in SEEDS:
        t0 = time.time()
        specs = make_specs(planner, dE0, X_ref, X_test, seed, lengths)
        from rdsg import inject
        X_test_inj = inject(X_test, specs, seed=seed)
        true_events = [(sp["start"], sp["start"] + sp["length"], sp["type"]) for sp in specs]

        arms = [
            ("b5a_mi", MIGraphBaseline(spp=spp, fa_target=FA_TARGET)),
            ("b5b_pcorr", WindowDivergenceBaseline("pcorr", spp=spp, fa_target=FA_TARGET)),
            ("b5c_var", VARResidualBaseline(spp=spp, fa_target=FA_TARGET)),
            ("b1_corr", WindowDivergenceBaseline("corr", spp=spp, fa_target=FA_TARGET)),
            ("b2_granger", GrangerBaseline(force_R[:12], spp=spp, fa_target=FA_TARGET)),
        ]
        for name, bl in arms:
            pr, _ = bl.run(X_ref, X_test_inj)
            pr3 = [(int(a), int(b), None) for a, b in pr]
            m = event_metrics(true_events, pr3, len(X_test), spp)
            m["recall_by_type"] = recall_by_type(true_events, pr3)
            res.setdefault(name, {})[str(seed)] = dict(
                events=[[int(a), int(b)] for a, b in pr], theta=None)
            res[name][str(seed)].update(m)
            old_r = old.get(name, {}).get(str(seed), {}).get("recall")
            print(f"  [{key} s{seed}] {name:11s} R={m['recall']:.3f} fa/d={m['fa_per_day']:.2f} "
                  f"(was {old_r if old_r is None else round(old_r,3)})", flush=True)
        json.dump(res, open(out_path, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1, default=float)
        print(f"  seed {seed} done ({time.time()-t0:.0f}s)", flush=True)


def main():
    for key, loader in (("metr_la", load_metr_la), ("pems_bay", load_pems_bay),
                        ("beijing_air", load_beijing_air)):
        run_dataset(key, loader)
    print("ALL DONE")


if __name__ == "__main__":
    main()
