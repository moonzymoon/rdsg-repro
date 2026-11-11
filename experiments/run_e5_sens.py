# -*- coding: utf-8 -*-
"""E5 sensitivity: K / w sweep (synthetic factor system, 2 seeds, fast version)."""
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
from rdsg import RDSGDetector, inject, sample_specs, event_metrics
from run_e0_syn import make_synthetic, rank_pairs_by_ref_gamma, recall_by_type
from run_e0_syn2 import build_configs, channel_scan, REF_LEN, TEST_LEN, PAIR_STRATA

OUT = os.path.normpath(os.path.join(_results_root(), "paper"))
BS = chr(92)
taus = [0, 1, -1]


def to_short(t):
    return {"inversion": "inv", "breakdown": "brk", "emergence": "emb"}.get(t, t)


def run_one(Xs, seed, K, w):
    X_ref, X_test = Xs[:REF_LEN], Xs[REF_LEN:]
    strong, weak, gam = rank_pairs_by_ref_gamma(X_ref, list(range(Xs.shape[1])), K, 512)
    vals = np.array(sorted(gam.values()))
    hi, lo = np.quantile(vals, 0.9), np.quantile(vals, 0.1)
    strong = [p for p, v in gam.items() if v >= hi]
    weak = [p for p, v in gam.items() if v <= lo]
    specs = sample_specs(None, strong, weak, n_each=8, t_len=TEST_LEN, margin=600, seed=seed)
    X_inj = inject(X_test, specs, seed=seed)
    true_events = [(sp["start"], sp["start"] + sp["length"], sp["type"]) for sp in specs]
    d = RDSGDetector(K=K, w=w, blk=max(8, w // 8), B=199, q=0.05, seed=1)
    d.fit(X_ref, channels=list(range(Xs.shape[1])), taus=list(range(-3, 4)))
    Gcal = {id(d): d.granulate(X_ref)}
    G = {id(d): d.granulate(np.vstack([X_ref, X_inj]))}
    evs, th, _ = channel_scan([d], Gcal, G, "s", cal_end=REF_LEN // 2,
                              scan_start=REF_LEN, scan_end=REF_LEN + TEST_LEN,
                              gap=8 * 32, spp=60.0)
    pred = [(int(a - REF_LEN), min(int(b - REF_LEN), TEST_LEN), None) for a, b in evs]
    m = event_metrics(true_events, pred, TEST_LEN, 60.0)
    return m["recall"], m["fa_per_day"]


def main():
    seeds = [201, 202]
    rows = []
    for K in (3, 4, 5):
        for w in (128, 256):
            rec, fas = [], []
            for sd in seeds:
                Xs = make_synthetic(REF_LEN + TEST_LEN, seed=sd)
                r, f = run_one(Xs, sd, K, w)
                rec.append(r); fas.append(f)
            rows.append((K, w, float(np.mean(rec)), float(np.mean(fas))))
            print(f"K={K} w={w}: R={np.mean(rec):.2f} fa={np.mean(fas):.2f}", flush=True)
    json.dump(rows, open(os.path.join(OUT, "e5_sensitivity.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
