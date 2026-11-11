# -*- coding: utf-8 -*-
"""Regenerate the derived statistics files from per-seed results.

Restores a previously ad-hoc derivation step into the repository:
  - results/paper/pooled_by_type.json : pooled per-type recalls (the pooled
    event-matching convention used by the paper's per-type tables), computed
    from each method's per-seed recall_by_type and the true-event counts per
    type taken from the RDSG detail rows (identical specs for all methods
    within a seed).
  - results/paper/e2_significance.json : one-sided Wilcoxon (ours > strongest
    EXTERNAL baseline; ours > gamma-only ablation) over per-seed recalls.
    'External' excludes the RDSG family itself (rdsg, abl_*).
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
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
RP = os.path.join(_results_root(), "paper")
RE0 = os.path.join(_results_root(), "e0")

EXTERNAL = ("b4_if", "b4_lof", "b5a_mi", "b5b_pcorr", "b5c_var", "b1_corr", "b2_granger", "b3_gdn", "abl_at", "abl_tranad")
GAMMA_KEY = {"e0": "gamma_only", "e2": "abl_gamma"}


def truth_counts_by_type(seed_detail):
    """True events per type in one seed, from RDSG detail rows [start,len,type,...,hit]."""
    cnt = {}
    for row in seed_detail:
        t = row[2]
        cnt[t] = cnt.get(t, 0) + 1
    return cnt


def pooled_block(d):
    """{method: {type: {pooled_recall, n}}} pooled over seeds."""
    out = {}
    if "rdsg" not in d:
        return out
    truth = {}
    for s, v in d["rdsg"].items():
        rows = v.get("detail") or v.get("per_event") or []
        truth[s] = truth_counts_by_type(rows)
    for m, seeds in d.items():
        agg = {}
        for s, v in seeds.items():
            rbt = v.get("recall_by_type") or {}
            for t, n in truth.get(s, {}).items():
                if t not in rbt:
                    continue
                a = agg.setdefault(t, {"matched": 0, "n": 0})
                a["matched"] += round(float(rbt[t]) * n)
                a["n"] += n
        out[m] = {t: {"pooled_recall": round(a["matched"] / a["n"], 4) if a["n"] else 0.0, "n": a["n"]}
                  for t, a in agg.items()}
    return out


def per_seed_recalls(d, m):
    return [float(v["recall"]) for s, v in sorted(d[m].items(), key=lambda kv: int(kv[0]))
            if "recall" in v]


def one_sided_p(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if np.allclose(x, y):
        return 1.0
    try:
        return float(wilcoxon(x, y, alternative="greater", zero_method="pratt").pvalue)
    except ValueError:
        return 1.0




def _load_deep(key):
    import json as _json, os as _os
    fp = _os.path.join(_results_root(), "paper", {"abl_at": "abl_at.json",
                                                  "abl_tranad": "abl_tranad.json"}[key])
    return _json.load(open(fp, encoding="utf-8")) if _os.path.exists(fp) else {}


def _with_deep(cfg, d):
    for k in ("abl_at", "abl_tranad"):
        dd = _load_deep(k).get(cfg)
        if dd:
            d.setdefault(k, {}).update(dd)
    return d

def main():
    # pooled per-type recalls
    pooled = {}
    e0raw_path = os.path.join(RE0, "e0_syn2_raw.json")
    if os.path.exists(e0raw_path):
        pooled["e0"] = pooled_block(json.load(open(e0raw_path, encoding="utf-8")))
    pooled["e2"] = {}
    for cfg in ("metr_la", "beijing_air", "pems_bay"):
        f = os.path.join(RP, f"e2_{cfg}.json")
        if os.path.exists(f):
            pooled["e2"][cfg] = pooled_block(json.load(open(f, encoding="utf-8")))
    json.dump(pooled, open(os.path.join(RP, "pooled_by_type.json"), "w"), indent=1)
    print("saved pooled_by_type.json:",
          {k: list(v.keys()) if k == "e2" else list(v.keys()) for k, v in pooled.items()})

    # significance vs strongest external baseline and vs gamma-only
    sig = {}
    for cfg in ("metr_la", "beijing_air", "pems_bay"):
        f = os.path.join(RP, f"e2_{cfg}.json")
        if not os.path.exists(f):
            continue
        d = json.load(open(f, encoding="utf-8"))
        _with_deep(cfg, d)
        ours = per_seed_recalls(d, "rdsg")
        ext_means = {m: float(np.mean(per_seed_recalls(d, m))) for m in EXTERNAL if m in d}
        if not ext_means:
            continue
        strongest = max(ext_means, key=ext_means.get)
        gam = per_seed_recalls(d, GAMMA_KEY["e2"]) if GAMMA_KEY["e2"] in d else None
        sig[cfg] = dict(
            n_seeds=len(ours),
            strongest_baseline=strongest,
            ours_mean=round(float(np.mean(ours)), 4),
            strongest_mean=round(ext_means[strongest], 4),
            p_vs_strongest=one_sided_p(ours, per_seed_recalls(d, strongest)),
            p_vs_gamma_only=one_sided_p(ours, gam) if gam else None,
            gamma_mean=round(float(np.mean(gam)), 4) if gam else None,
        )
    json.dump(sig, open(os.path.join(RP, "e2_significance.json"), "w"), indent=1)
    print("saved e2_significance.json:")
    print(json.dumps(sig, indent=1))


if __name__ == "__main__":
    main()
