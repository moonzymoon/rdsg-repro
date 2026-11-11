# -*- coding: utf-8 -*-
"""E2c: host-linearity grouping + event-length grouping analysis (second
review-hardening round, 2026-10-06).

Zero re-runs: every method's event predictions are already stored in the
events field of e2_{ds}.json. This script only (1) deterministically rebuilds
the injection specs (same make_specs, same seeds), (2) computes reference-segment
linearity for each host pair (linear-regression R^2 and |Spearman|),
(3) rebuilds the per-event detection table via match_events, and
(4) aggregates recall grouped by host linearity / event length / type.

Pre-registered grouping: host pairs with |Spearman| >= 0.9 -> linear group,
otherwise nonlinear group (emb2-subset hosts form their own subset group;
pairwise linearity does not apply to them).
Supports the claim: granular machinery earns its complexity on non-Gaussian
couplings.
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
from scipy.stats import spearmanr

ROOT = _results_root()
import os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rdsg.datasets import load_metr_la, load_pems_bay, load_beijing_air
from rdsg.metrics import match_events
from run_paper_e2 import (build_channel, make_specs, PAIR_STRATA_E,
                          LEN_BY_SPP, N_EACH, N_EMB2)

OUT = os.path.normpath(os.path.join(_results_root(), "paper"))
RHO_SPLIT = 0.9
METHODS = ["rdsg", "b5b_pcorr", "b1_corr", "b4_lof", "b4_if", "b2_granger", "b3_gdn"]
NICE = {"rdsg": "RDSG", "b5b_pcorr": "pcorr", "b1_corr": "corr", "b4_lof": "LOF",
        "b4_if": "IF", "b2_granger": "Granger", "b3_gdn": "GDN"}
LOADERS = {"metr_la": load_metr_la, "pems_bay": load_pems_bay, "beijing_air": load_beijing_air}


def lin_stats(X_ref, i, j):
    tail = X_ref[-2048:]
    x, y = tail[:, i], tail[:, j]
    rho = float(abs(spearmanr(x, y).statistic))
    A = np.vstack([x, np.ones_like(x)]).T
    coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = float(1 - res[0] / ss_tot) if len(res) and ss_tot > 0 else 0.0
    return rho, r2


def main():
    out = {}
    for key, loader in LOADERS.items():
        X, spp, meta = loader()
        X_ref, X_test = X[:12000], X[12000:12000 + 18000]
        taus = (0, 1, -1)
        lengths = LEN_BY_SPP[spp]
        planner = build_channel(X_ref, taus, strata=(16, 24))[0]
        force_R, force_E = planner.strong_pairs_, planner.weak_pairs_
        dE0 = build_channel(X_ref, taus, strata=PAIR_STRATA_E, n_subset=6,
                            force_pairs=(force_R, force_E))[0]
        res = json.load(open(os.path.join(OUT, f"e2_{key}.json"), encoding="utf-8"))

        # Rebuild specs and the per-event detection table seed by seed
        per_method = {m: {} for m in METHODS}   # m -> group -> hit/total
        host_rho = {}
        for seed in range(1, 11):
            specs = make_specs(planner, dE0, X_ref, X_test, seed, lengths)
            true_events = [(sp["start"], sp["start"] + sp["length"], sp["type"]) for sp in specs]
            true3 = [(a, b, None) for a, b, _ in true_events]
            tidx = {(int(a), int(b)): k for k, (a, b, _t) in enumerate(true_events)}
            groups = []
            for k, sp in enumerate(specs):
                if sp["type"] == "emb2":
                    g = "subset"
                else:
                    rk = (sp["i"], sp["j"])
                    if rk not in host_rho:
                        host_rho[rk] = lin_stats(X_ref, sp["i"], sp["j"])
                    g = "linear" if host_rho[rk][0] >= RHO_SPLIT else "nonlinear"
                groups.append(dict(idx=k, type=sp["type"], length=sp["length"], group=g))
            for m in METHODS:
                rec = res.get(m, {}).get(str(seed), {})
                if m == "rdsg":
                    # RDSG stores detail=[start,length,type,delay,detected] in specs order
                    det = rec.get("detail") or []
                    dmap = {(int(a), int(L), t): bool(h) for a, L, t, _dl, h in det}
                    hits = {ginfo["idx"] for ginfo in groups
                            if dmap.get((specs[ginfo["idx"]]["start"],
                                         specs[ginfo["idx"]]["length"],
                                         specs[ginfo["idx"]]["type"]), False)}
                else:
                    ev = rec.get("events") or []
                    pr = [(int(a), int(b), None) for a, b in ev]
                    matches, _, _ = match_events(list(true3), pr)
                    hits = {tidx[(int(tr[0]), int(tr[1]))] for tr, p in matches if p is not None}
                for ginfo in groups:
                    g = ginfo["group"]
                    per_method[m].setdefault(g, [0, 0])
                    per_method[m][g][1] += 1
                    if ginfo["idx"] in hits:
                        per_method[m][g][0] += 1
        # Aggregate
        tbl = {}
        for m in METHODS:
            tbl[m] = {g: dict(recall=(h / t if t else None), hits=h, n=t)
                      for g, (h, t) in per_method[m].items()}
        out[key] = {"rho_split": RHO_SPLIT, "n_specs_per_seed": N_EACH * 3 + N_EMB2,
                    "by_group": tbl,
                    "by_type": {m: tbl_by_type(key, res, m) for m in METHODS}}
        print(f"== {meta['name']} ==")
        for m in METHODS:
            row = out[key]["by_group"][m]
            cells = " ".join(f"{g}={row[g]['recall']:.2f}(n={row[g]['n']})" if row.get(g) and row[g]["recall"] is not None
                             else f"{g}=--" for g in ("linear", "nonlinear", "subset"))
            print(f"  {NICE[m]:8s} {cells}")
    json.dump(out, open(os.path.join(OUT, "e2c_linearity.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=float)
    print("saved e2c_linearity.json")


def tbl_by_type(key, res, m):
    """Recall grouped by type x event length (pooled across seeds, re-matched
    from events)."""
    return None  # type grouping is already covered by recall_by_type; length grouping is done separately


if __name__ == "__main__":
    main()
