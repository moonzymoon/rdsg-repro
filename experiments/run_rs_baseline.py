# -*- coding: utf-8 -*-
"""R7-2: rough-set baseline -- granulated-IF implementation plus one experiment run."""
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
from rdsg import Granulator, inject, sample_specs, event_metrics
from rdsg.metrics import match_events
from rdsg.datasets import load_metr_la, load_beijing_air
from baselines.point_agg import _fit_score
from sklearn.ensemble import IsolationForest

OUT = os.path.normpath(os.path.join(_results_root(), "paper"))


def granulated_if_scores(X_ref, X_test, K=4, seed=0):
    """Granulated IF: equal-frequency granulation first, then train IF on
    granule indices as features."""
    gran = Granulator(K).fit(X_ref)
    G_ref = gran.transform(X_ref)
    G_test = gran.transform(X_test)
    # Per-channel IF on granule indices
    agg = np.full(len(G_test), -np.inf)
    for j in range(G_ref.shape[1]):
        m = IsolationForest(n_estimators=200, random_state=seed, contamination="auto")
        m.fit(G_ref[:, j:j+1])
        agg = np.maximum(agg, -m.score_samples(G_test[:, j:j+1]))
    return agg


def granulated_if_events(X_ref, X_test_inj, K=4, gap=128, seed=0, ref_len=12000):
    """Reference-segment FA-matched threshold + test-segment event extraction."""
    gran = Granulator(K).fit(X_ref)
    G_ref = gran.transform(X_ref)
    G_test = gran.transform(X_test_inj)
    # Half-split cross-scoring on the reference segment
    half = len(G_ref) // 2
    ref_agg = np.full(len(G_ref), -np.inf)
    for j in range(G_ref.shape[1]):
        m1 = IsolationForest(n_estimators=200, random_state=seed)
        m1.fit(G_ref[half:, j:j+1])
        ref_agg[:half] = np.maximum(ref_agg[:half], -m1.score_samples(G_ref[:half, j:j+1]))
        m2 = IsolationForest(n_estimators=200, random_state=seed)
        m2.fit(G_ref[:half, j:j+1])
        ref_agg[half:] = np.maximum(ref_agg[half:], -m2.score_samples(G_ref[half:, j:j+1]))
    # FA-matched threshold
    spp = 300.0
    days = len(G_ref) * spp / 86400.0
    allow = max(1, int(np.ceil(0.25 * days)))
    th = np.inf
    for q in np.linspace(0.99, 0.99995, 200):
        cand = np.quantile(ref_agg, q)
        # Simple merging
        on = ref_agg > cand
        idx = np.where(on)[0]
        if len(idx) == 0:
            continue
        n_ev = 1
        for k in range(1, len(idx)):
            if idx[k] - idx[k-1] > gap:
                n_ev += 1
        if n_ev <= allow:
            th = cand
            break
    # Test-segment scoring
    agg = np.full(len(G_test), -np.inf)
    for j in range(G_ref.shape[1]):
        m = IsolationForest(n_estimators=200, random_state=seed)
        m.fit(G_ref[:, j:j+1])
        agg = np.maximum(agg, -m.score_samples(G_test[:, j:j+1]))
    # Event extraction
    on = agg > th
    idx = np.where(on)[0]
    events = []
    if len(idx) > 0:
        t0 = prev = idx[0]
        for t in idx[1:]:
            if t - prev > gap:
                events.append((t0, prev + 1))
                t0 = t
            prev = t
        events.append((t0, prev + 1))
    return events


def run_dataset(loader, name, seed=1):
    X, spp, meta = loader()
    X_ref, X_test = X[:12000], X[12000:12000+18000]
    lengths = (30, 60, 120) if spp == 300.0 else (12, 24, 48)

    # Injection (reuses RDSG's host selection)
    import run_paper_e2 as R
    dR0 = R.build_channel(X_ref, [0,1,-1], strata=R.PAIR_STRATA_R)[0]
    dE0 = R.build_channel(X_ref, [0,1,-1], strata=R.PAIR_STRATA_E, n_subset=6)[0]
    specs = R.make_specs(dR0, dE0, X_ref, X_test, seed, lengths)
    X_test_inj = inject(X_test, specs, seed=seed)
    true_events = [(sp["start"], sp["start"]+sp["length"], sp["type"]) for sp in specs]

    # Granulated IF
    events = granulated_if_events(X_ref, X_test_inj, K=4, seed=seed)
    m = event_metrics(true_events, [(a, b, None) for a, b in events], len(X_test), spp)
    print(f"[{name}] granulated-IF: R={m['recall']:.2f} fa/d={m['fa_per_day']:.2f} n={len(events)}")
    return m


if __name__ == "__main__":
    print("=== R7-2: Granulated-IF (rough-set style baseline) ===")
    results = {}
    for name, loader in [("METR-LA", load_metr_la), ("Beijing", load_beijing_air)]:
        results[name] = run_dataset(loader, name, seed=1)
    json.dump(results, open(os.path.join(OUT, "rs_baseline_granulated_if.json"), "w"),
              ensure_ascii=False, indent=1, default=float)
    print("saved")
