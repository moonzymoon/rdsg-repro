# -*- coding: utf-8 -*-
"""Gas-pipeline case study: four-tuple (gamma/delta/D_JS) on monitored edges
before vs during each labelled attack. Writes results/paper/gas_case_tuples.json;
the manuscript case study quotes the strongest gamma-conserved/delta-reversed
signature from this table."""
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
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
import os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from rdsg.datasets import load_gas_pipeline
from rdsg.granulation import Granulator
from rdsg.tables import block_tables
from rdsg.stats import four_tuple

OUT = os.path.join(_results_root(), "paper", "gas_case_tuples.json")
K, blk = 4, 32

def main():
    X, spp, meta = load_gas_pipeline()
    ref0, ref1 = meta["ref"]
    t0, t1 = meta["test"]
    events = meta["events"]
    g = Granulator(K=K).fit(X[ref0:ref1])
    G = g.transform(X)
    rows = []
    for (i, j) in [(0, 1), (1, 2), (2, 3), (0, 2), (1, 3), (2, 4), (3, 4), (5, 6), (6, 7)]:
        edge = ((i,), j, -1)
        tab_ref = block_tables(G, [edge], K, blk, t0=ref0, t1=ref1)[0].sum(axis=0)
        for ei, ev in enumerate(events):
            s_abs, e_abs = t0 + ev[0], t0 + ev[1]
            w0 = max(ref1, s_abs - 64)
            w1 = min(len(X), e_abs + 64)
            if w1 - w0 < 128:
                w0 = max(ref1, w1 - 128)
            if w1 <= w0:
                continue
            tab_cur = block_tables(G, [edge], K, blk, t0=w0, t1=w1)[0].sum(axis=0)
            st = four_tuple(tab_ref[None], tab_cur[None], K)
            rows.append(dict(edge=(i, j), ev=ei,
                             g_ref=float(np.mean(st["gamma_ref"])), g_cur=float(np.mean(st["gamma_cur"])),
                             d_ref=float(np.mean(st["delta_ref"])), d_cur=float(np.mean(st["delta_cur"])),
                             djs=float(np.mean(st["djs"]))))
    json.dump(rows, open(OUT, "w"), indent=1)
    cand = [r for r in rows if abs(r["g_cur"] - r["g_ref"]) < 0.06
            and r["d_ref"] * r["d_cur"] < 0 and r["djs"] > 0.15]
    cand.sort(key=lambda r: -abs(r["d_cur"] - r["d_ref"]))
    print("strongest gamma-conserved delta-reversed signatures:")
    for r in cand[:5]:
        print(f"  edge {r['edge']} attack#{r['ev']}: gamma {r['g_ref']:.2f}->{r['g_cur']:.2f} "
              f"delta {r['d_ref']:+.2f}->{r['d_cur']:+.2f} D_JS {r['djs']:.2f}")

if __name__ == "__main__":
    main()
