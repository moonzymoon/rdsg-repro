# -*- coding: utf-8 -*-
"""Phase A: extended metrics — bootstrap CIs, FA intervals, delay, PR-AUC.
Pure post-processing from e2_*.json and e1_gas.json (no new experiments).

Output schema (consumed by gen_supp_stats.py):
  {<cfg>: {<method>: {recall_mean, recall_ci "[lo,hi]", fa_mean, fa_ci "[lo,hi]",
                      delay_med_s?}}, "pr_auc": {<method>: float}}
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

import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
RES = os.path.join(_results_root())
OUT = os.path.join(RES, "paper")

FS = {"metr_la": 300.0, "pems_bay": 300.0, "beijing_air": 3600.0}
DUR = {"metr_la": 18000, "pems_bay": 18000, "beijing_air": 18000}


def load_detail(cfg):
    f = os.path.join(RES, "paper", f"e2_{cfg}.json")
    if not os.path.exists(f):
        return None
    return json.load(open(f, encoding="utf-8"))


def calc_pr_auc(recalls, fars):
    """Trapezoid PR-AUC over the (FA, recall) operating curve."""
    pts = sorted(zip(fars, recalls))
    auc = 0.0
    for i in range(1, len(pts)):
        x0, y0 = pts[i - 1]
        x1, y1 = pts[i]
        auc += (y0 + y1) / 2 * (x1 - x0)
    return auc / max(pts[-1][0] - pts[0][0], 1e-12) if pts[-1][0] > pts[0][0] else pts[0][1]


def bootstrap_ci(vals, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    vals = list(vals)
    meds = []
    for _ in range(n_boot):
        sample = [vals[i] for i in rng.integers(0, len(vals), len(vals))]
        meds.append(float(np.mean(sample)))
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def main():
    results = {}

    for cfg in ("metr_la", "beijing_air", "pems_bay"):
        d = load_detail(cfg)
        if not d:
            continue
        spp = FS[cfg]
        for method in ("rdsg", "b4_if", "b4_lof", "b5b_pcorr", "b1_corr", "b2_granger"):
            if method not in d:
                continue
            seeds = d[method]
            recalls = [v["recall"] for v in seeds.values() if "recall" in v]
            if not recalls:
                continue
            lo, hi = bootstrap_ci(recalls)
            res = results.setdefault(cfg, {}).setdefault(method, {})
            res["recall_mean"] = round(float(np.mean(recalls)), 3)
            res["recall_ci"] = f"[{lo:.3f},{hi:.3f}]"
            fas = [v["fa_per_day"] for v in seeds.values() if "fa_per_day" in v]
            if fas:
                fa_lo, fa_hi = bootstrap_ci(fas)
                res["fa_mean"] = round(float(np.mean(fas)), 2)
                res["fa_ci"] = f"[{fa_lo:.2f},{fa_hi:.2f}]"
            dly = [v["delay_med"] for v in seeds.values() if v.get("delay_med") is not None]
            if dly:
                res["delay_med_s"] = round(float(np.median(dly)) * spp, 1)

    # PR-AUC from the gas-pipeline operating curves (e1_gas.json)
    e1_path = os.path.join(RES, "paper", "e1_gas.json")
    if os.path.exists(e1_path):
        e1 = json.load(open(e1_path, encoding="utf-8"))
        pr = {}
        for m, pts in e1.get("curves", {}).items():
            fars = [p["fa_per_day"] for p in pts]
            recs = [p["recall"] for p in pts]
            if len(fars) > 2:
                pr[m] = round(calc_pr_auc(recs, fars), 3)
        results["pr_auc"] = pr

    os.makedirs(OUT, exist_ok=True)
    json.dump(results, open(os.path.join(OUT, "ext_metrics.json"), "w"), indent=1)
    print(json.dumps(results.get("pr_auc", {}), indent=1))
    print("saved ext_metrics.json")


if __name__ == "__main__":
    main()
