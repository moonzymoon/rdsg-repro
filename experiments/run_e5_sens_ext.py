# -*- coding: utf-8 -*-
"""E5 sensitivity extension: 2 -> 5 seeds per cell (second review-hardening
round). Incrementally runs the new seeds and merges the results."""
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
from run_e0_syn import make_synthetic
from run_e0_syn2 import REF_LEN, TEST_LEN
from run_e5_sens import run_one

OUT = os.path.normpath(os.path.join(_results_root(), "paper"))
NEW_SEEDS = [203, 204, 205]


def main():
    path = os.path.join(OUT, "e5_sensitivity5.json")
    rows = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else []
    done = {(r[0], r[1], r[4]) for r in rows if len(r) > 4}
    for K in (3, 4, 5):
        for w in (128, 256):
            for sd in NEW_SEEDS:
                if (K, w, sd) in done:
                    continue
                t0 = time.time()
                Xs = make_synthetic(REF_LEN + TEST_LEN, seed=100 + sd - 200)
                rec, fa = run_one(Xs, sd, K, w)
                rows.append([K, w, rec, fa, sd])
                print(f"K={K} w={w} seed={sd}: R={rec:.2f} fa={fa:.2f} ({time.time()-t0:.0f}s)",
                      flush=True)
                json.dump(rows, open(path, "w", encoding="utf-8"), indent=1)
    # Aggregation: per-cell means (new 5-seed figures = old 2 + new 3)
    agg = {}
    for r in rows:
        K, w = r[0], r[1]
        old = [x for x in json.load(open(os.path.join(OUT, "e5_sensitivity.json"))) if x[0] == K and x[1] == w]
        recs = [r[2]] + [o[2] for o in old]
        fas = [r[3]] + [o[3] for o in old]
        agg[(K, w)] = (float(np.mean(recs)), float(np.mean(fas)), len(recs))
    print("\n== 5-seed per-cell means ==")
    for (K, w), (r, f, n) in sorted(agg.items()):
        print(f"  K={K} w={w}: recall={r:.2f} fa={f:.2f} (n={n})")
    json.dump({f"{K}_{w}": dict(recall=r, fa=f, n=n) for (K, w), (r, f, n) in agg.items()},
              open(os.path.join(OUT, "e5_sensitivity5_summary.json"), "w", encoding="utf-8"),
              indent=1)


if __name__ == "__main__":
    main()
