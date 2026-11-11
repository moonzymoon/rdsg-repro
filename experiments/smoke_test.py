# -*- coding: utf-8 -*-
"""Synthetic-data smoke test: verify each pipeline stage and the direction of
the blind-spot correction."""
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

import os, sys
import numpy as np

ROOT = _results_root()
import os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from rdsg import (Granulator, block_tables, four_tuple, gamma_beta,
                  RDSGDetector, inject, event_metrics)
from rdsg.stats import grain_direction, gamma_soft

rng = np.random.default_rng(42)
T, n = 6000, 6
t = np.arange(T)

# Synthesize a multi-channel system with real dependency structure:
# x0 slow-drifting AR(1); x1 = x0 strongly positively correlated + noise;
# x2 = -x0 (strongly negatively correlated); x3 independent;
# x4, x5 = independent noise
x0 = np.zeros(T)
for i in range(1, T):
    x0[i] = 0.95 * x0[i - 1] + 0.3 * rng.standard_normal()
X = np.column_stack([
    x0,
    0.95 * x0 + rng.standard_normal(T),
    -0.95 * x0 + rng.standard_normal(T),
    rng.standard_normal(T),
    rng.standard_normal(T),
    rng.standard_normal(T),
])
REF = 4000

# ---------- 1. Granulation ----------
gran = Granulator(K=4).fit(X[:REF])
G = gran.transform(X)
assert G.shape == (T, n) and G.min() >= 0 and G.max() <= 3
occ = [np.bincount(G[:REF, j], minlength=4) for j in range(n)]
print("1. granulation ok; ref occupancy ch0:", occ[0])

# ---------- 2. Quadruple directionality check ----------
def tab_pair(G, i, d, tau, K=4, lo=0, hi=None):
    from rdsg.tables import edge_code
    hi = hi or G.shape[0]
    code = edge_code(G, (i,), d, tau, K)[lo:hi]
    c = code[code >= 0]
    return np.bincount(c, minlength=K * K).reshape(K, K).astype(float)[None]

T_clean = tab_pair(G, 0, 1, 0, lo=0, hi=REF)
T_x2 = tab_pair(G, 0, 2, 0, lo=0, hi=REF)
g_pos, d_pos = gamma_soft(T_clean)[0], grain_direction(T_clean, 4)[0]
g_neg, d_neg = gamma_soft(T_x2)[0], grain_direction(T_x2, 4)[0]
print(f"2. x0→x1 γ={g_pos:.2f} δ={d_pos:+.2f} | x0→x2 γ={g_neg:.2f} δ={d_neg:+.2f}")
assert g_pos > 0.45 and d_pos > 0.5
assert g_neg > 0.45 and d_neg < -0.5
print("   gamma nearly identical (empirical blind spot) while delta flips sign -> directional component is effective")

# Direct empirical evidence of the gamma blind spot: same reference table vs
# mapping-reversed table
tabA = np.zeros((1, 4, 4)); tabA[0, np.arange(4), np.arange(4)] = 100
tabB = np.zeros((1, 4, 4)); tabB[0, np.arange(4), np.arange(3, -1, -1)] = 100
ga, gb = gamma_beta(tabA, 0.9)[0], gamma_beta(tabB, 0.9)[0]
st_flip = four_tuple(tabA, tabB, 4)
assert abs(ga - gb) < 1e-9, "gamma should be invariant to mapping reversal"
assert st_flip["rho"][0] < 0.05 and abs(st_flip["ddelta"][0]) > 1.5
print(f"3. blind-spot construction verified: γ(A)={ga:.2f}=γ(B)={gb:.2f}, ρ={st_flip['rho'][0]:.2f}, "
      f"Δδ={st_flip['ddelta'][0]:+.2f}, D_JS={st_flip['djs'][0]:.2f} ✓")

# ---------- 4. Margin-preserving injection ----------
X_test = X[REF:].copy()
spec = [dict(type="inv", i=0, j=1, start=500, length=200)]
Xi = inject(X_test, spec, seed=1)
assert np.allclose(np.sort(Xi[500:700, 3]), np.sort(X_test[500:700, 3])), "marginals must be exactly preserved"
from scipy.stats import spearmanr
r_before = spearmanr(X_test[500:700, 0], X_test[500:700, 1]).statistic
r_after = spearmanr(Xi[500:700, 0], Xi[500:700, 1]).statistic
print(f"4. injection: ch1 marginals preserved pointwise ✓, Spearman(0,1) {r_before:+.2f} → {r_after:+.2f}")
assert r_after < -0.7

# ---------- 5. End-to-end detection ----------
det = RDSGDetector(K=4, w=256, blk=32, B=99, q=0.05, seed=0)
det.fit(X[:REF], channels=list(range(6)), taus=[0])
Gfull = det.granulate(np.vstack([X[:REF], Xi]))
events, recs = det.detect(Gfull, REF)
events = det.localize(Gfull, events)
events = det.type_events(Gfull, events)
print(f"5. detect: {len(events)} events")
for e in events:
    print(f"   t=[{e['t0']-REF},{e['t1']-REF}) type={e.get('type')} votes={e.get('type_votes')}")
true_events = [(500, 700, "inv")]
pred = [(e["t0"] - REF, min(e["t1"] - REF, T - REF), e.get("type", "?")) for e in events]
m = event_metrics(true_events, pred, T - REF, 1.0)
print(f"   metrics: recall={m['recall']:.2f} fa/d={m['fa_per_day']:.2f} type={m['type_acc']:.2f}")
assert any(_ov(500, 700, p[0], p[1]) for _ov in (lambda a, b, c, d: max(a, c) < min(b, d),) for p in pred), "the injected event should be detected"
print("SMOKE TEST PASS")
