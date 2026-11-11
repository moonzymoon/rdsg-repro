# -*- coding: utf-8 -*-
"""Single-factor non-copula ablation (v2, reviewer-audited).

Factor under test: the IPF copula standardisation inside the edge statistics
(stats.ipf_uniform applied in four_tuple / std_gamma). Everything else is the
production pipeline unchanged: same candidate graph (built by the unpatched
planner, passed via force_pairs), same windows/granularities, same rate
calibration, same merging and event metrics, same per-seed injection specs.

Implementation: rdsg.detector.four_tuple is module-patched to a raw-table
variant that skips ipf_uniform; the downstream statistics (gamma_soft,
grain_direction, rule_consistency, cond_js) already accept raw count tables.
The candidate graph is built BEFORE the patch, so it is bit-identical to the
main run's graph. v1 of this runner was a combined variant (hard gamma_beta,
max-over-edges statistic, fallback edge set, different merge gap) and is kept
in the archive; its numbers are superseded by this run.
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

import rdsg.detector as _det
from rdsg import stats as _st
from rdsg import inject
from rdsg.metrics import event_metrics
from rdsg.datasets import load_metr_la, load_pems_bay, load_beijing_air
from run_paper_e2 import (LEN_BY_SPP, build_channel, make_specs, recall_by_type,
                          ours_pipeline, PAIR_STRATA_E)

OUT = os.path.join(_results_root(), "paper", "abl_nocopula.json")
SEEDS = range(1, 11)

_FOUR_TUPLE_ORIG = _det.four_tuple


def four_tuple_raw(tab_ref, tab_cur, K, beta=0.9):
    """four_tuple without the copula standardisation step: every statistic is
    computed on the raw count tables (each statistic normalises internally),
    so marginal occupancy skew and cross-window marginal drift enter the
    contingency structure unremoved."""
    g_ref = _st.gamma_soft(tab_ref)
    g_cur = _st.gamma_soft(tab_cur)
    d_ref = _st.grain_direction(tab_ref, K)
    d_cur = _st.grain_direction(tab_cur, K)
    return {
        "dgamma": g_cur - g_ref,
        "rho": _st.rule_consistency(tab_ref, tab_cur),
        "ddelta": d_cur - d_ref,
        "djs": _st.cond_js(tab_ref, tab_cur),
        "gamma_ref": g_ref, "gamma_cur": g_cur,
        "delta_ref": d_ref, "delta_cur": d_cur,
    }


def run_dataset(key, loader, taus=(0, 1, -1), acc=None):
    X, spp, meta = loader()
    X_ref, X_test = X[:12000], X[12000:12000 + 18000]
    lengths = LEN_BY_SPP[spp]
    print(f"\n===== {key} {X.shape} spp={spp} =====", flush=True)
    # graph authority: built with the production (copula) statistics
    planner = build_channel(X_ref, taus, strata=(16, 24))[0]
    force_R, force_E = planner.strong_pairs_, planner.weak_pairs_
    dE0 = build_channel(X_ref, taus, strata=PAIR_STRATA_E, n_subset=6,
                        force_pairs=(force_R, force_E))[0]

    res = {}
    for seed in SEEDS:
        specs = make_specs(planner, dE0, X_ref, X_test, seed, lengths)
        X_test_inj = inject(X_test, specs, seed=seed)
        true_events = [(sp["start"], sp["start"] + sp["length"], sp["type"]) for sp in specs]
        t0 = time.time()
        _det.four_tuple = four_tuple_raw          # single-factor patch ON
        try:
            pred, rt = ours_pipeline(X_ref, X_test_inj, X_test, taus, spp=spp,
                                     force_R=force_R, force_E=force_E)
        finally:
            _det.four_tuple = _FOUR_TUPLE_ORIG    # patch OFF
        m = event_metrics(true_events, pred, len(X_test), spp)
        m["recall_by_type"] = recall_by_type(true_events, pred)
        m["runtime"] = rt
        res[str(seed)] = m
        print(f"  [{key} s{seed}] R={m['recall']:.3f} fa/d={m['fa_per_day']:.2f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        if acc is not None:
            acc[key] = res
            json.dump(acc, open(OUT, "w"), indent=1, default=float)
    recalls = [v["recall"] for v in res.values()]
    print(f"  [{key}] mean={np.mean(recalls):.3f} +/- {np.std(recalls):.3f} "
          f"range=[{min(recalls):.3f},{max(recalls):.3f}]")
    return {key: res}


def main():
    out = {}
    if os.path.exists(OUT):
        try:
            out = json.load(open(OUT))
        except Exception:
            out = {}
    for key, loader in (("metr_la", load_metr_la), ("pems_bay", load_pems_bay),
                        ("beijing_air", load_beijing_air)):
        out.update(run_dataset(key, loader, acc=out))
        json.dump(out, open(OUT, "w"), indent=1, default=float)
    print("saved", OUT)


if __name__ == "__main__":
    main()
