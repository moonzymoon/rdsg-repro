# -*- coding: utf-8 -*-
"""W6: neighborhood rough set sample-level outlierness baseline (a time-series
adaptation of the hu2008neighborhood formulation).

Sample-level scoring: for channel j within window W, delta_j = 0.15 x std(v);
the neighborhood coverage of point t is
cov(t) = #{t': |v(t')-v(t)| <= delta_j} / |W|; the outlier score is
out(t) = 1 - cov(t). Aggregated across channels by max (isomorphic to
IF/LOF-agg). Threshold set by half-split self-scan on the reference segment,
FA budget 0.25/day, identical to all other baselines.
Purpose: test "how sample-level rough-set scoring performs on relational
injections" --
Pre-registration: (a) floor (approx. IF) -> the discrimination between monitoring
targets becomes empirical evidence; (b) partial detection -> reported honestly
in the table.
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
from rdsg.datasets import load_metr_la, load_pems_bay, load_beijing_air
from rdsg.metrics import match_events
from baselines.relational import merge_events, pick_theta_split
from run_paper_e2 import build_channel, make_specs, PAIR_STRATA_E, LEN_BY_SPP

OUT = os.path.normpath(os.path.join(_results_root(), "paper"))
LOADERS = {"metr_la": load_metr_la, "pems_bay": load_pems_bay, "beijing_air": load_beijing_air}


def rough_scores(X_fit, X_score):
    """Per-channel neighborhood rough-set outlier score, max-aggregated.
    X_fit provides the std used to set delta; called per window."""
    n = X_score.shape[1]
    agg = np.zeros(len(X_score))
    for j in range(n):
        v_fit = X_fit[:, j]
        delta = 0.15 * (np.std(v_fit) + 1e-12)
        v = X_score[:, j]
        # O(n^2) within-window distance counting (affordable for w=64)
        d = np.abs(v[:, None] - v[None, :])
        cov = (d <= delta).sum(axis=1) / len(v)
        agg = np.maximum(agg, 1.0 - cov)
    return agg


def run_seed(X_ref, X_test_inj, fa_target=0.25, spp=300.0):
    half = len(X_ref) // 2
    ra = np.zeros(len(X_ref))
    for lo, hi, fit_lo, fit_hi in ((0, half, half, len(X_ref)),
                                   (half, len(X_ref), 0, half)):
        seg = rough_scores(X_ref[fit_lo:fit_hi], X_ref[lo:hi])
        ra[lo:hi] = seg
    th = pick_theta_split(ra, fa_target, spp, 64, 64)
    agg = rough_scores(X_ref, X_test_inj)
    evs = merge_events(np.arange(len(agg)), agg > th, 128)
    return [(int(a), int(b)) for a, b in evs], float(th)


def main():
    taus = (0, 1, -1)
    out_path = os.path.join(OUT, "w6_rough_baseline.json")
    res = json.load(open(out_path, encoding="utf-8")) if os.path.exists(out_path) else {}
    for key, loader in LOADERS.items():
        X, spp, meta = loader()
        X_ref, X_test = X[:12000], X[12000:12000 + 18000]
        lengths = LEN_BY_SPP[spp]
        planner = build_channel(X_ref, taus, strata=(16, 24))[0]
        force_R, force_E = planner.strong_pairs_, planner.weak_pairs_
        dE0 = build_channel(X_ref, taus, strata=PAIR_STRATA_E, n_subset=6,
                            force_pairs=(force_R, force_E))[0]
        res.setdefault(key, {"seeds": {}})
        recalls, fas = [], []
        for seed in range(1, 11):
            if str(seed) in res[key]["seeds"]:
                recalls.append(res[key]["seeds"][str(seed)]["recall"])
                fas.append(res[key]["seeds"][str(seed)]["fa_per_day"])
                continue
            t0 = time.time()
            specs = make_specs(planner, dE0, X_ref, X_test, seed, lengths)
            X_inj = inject(X_test, specs, seed=seed)
            true_events = [(sp["start"], sp["start"] + sp["length"], sp["type"]) for sp in specs]
            evs, th = run_seed(X_ref, X_inj, fa_target=0.25, spp=spp)
            matches, delays, falses = match_events([(a, b, None) for a, b, _ in true_events],
                                                   [(a, b, None) for a, b in evs])
            rec = sum(1 for _, p in matches if p is not None) / max(len(true_events), 1)
            days = len(X_test) * spp / 86400.0
            fa = len(falses) / max(days, 1e-9)
            # by-type pooled
            rows = {}
            for (t0e, t1e, t2), p in matches:
                rows.setdefault(t2, [0, 0])
                rows[t2][1] += 1
                rows[t2][0] += 1 if p is not None else 0
            res[key]["seeds"][str(seed)] = dict(
                recall=rec, fa_per_day=fa, theta=th,
                by_type={t: dict(recall=(h / n if n else None), n=n) for t, (h, n) in rows.items()})
            recalls.append(rec)
            fas.append(fa)
            print(f"[{key} s{seed}] R={rec:.2f} fa={fa:.2f} th={th:.3f} ({time.time()-t0:.0f}s)",
                  flush=True)
            json.dump(res, open(out_path, "w", encoding="utf-8"), indent=1, default=float)
        res[key]["summary"] = dict(recall_mean=round(float(np.mean(recalls)), 3),
                                   fa_mean=round(float(np.mean(fas)), 3), n=len(recalls))
        print(f"== {key}: rough-NN recall={res[key]['summary']['recall_mean']:.2f} "
              f"fa={res[key]['summary']['fa_mean']:.2f}")
    json.dump(res, open(out_path, "w", encoding="utf-8"), indent=1, default=float)
    print("saved w6_rough_baseline.json")


if __name__ == "__main__":
    main()
