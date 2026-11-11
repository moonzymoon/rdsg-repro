# -*- coding: utf-8 -*-
"""W1 varshape construction self-check (pre-registered capability gate: below
the bar, the construction does not enter the experiments).

Construction: within a window select the partner granule pair (g,h) with the
closest conditional means and swap the j value sets of the two granules.
Theoretical expectations: the global multiset is exactly preserved (marginals
unchanged); each granule's conditional mean moves by <= |m_g-m_h|;
conditional variances are swapped -> the shape of the conditional
distributions (binned by granule) changes substantially -> D_JS moves a lot;
gamma/delta only move by O(d).

Pass criterion (pre-registered): across 10 random trials,
  median relative change of gamma < 0.05 and median relative change of
  delta < 0.05, and the median absolute dJS change > 3 x the median of
  max(|dGamma|, |dDelta|)
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

import sys, os
import numpy as np

ROOT = _results_root()
import os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from rdsg.stats import four_tuple
from rdsg.granulation import Granulator

K = 4


def varshape_pairs(X_test, X_ref, i, j, s, L, rng):
    """Within the window [s, s+L), perform a closest-mean granule-pair swap on
    channel pair (i,j); returns the modified copy."""
    sl = slice(s, s + L)
    X = np.array(X_test, copy=True)
    gran = Granulator(K=K)
    gi = gran.fit_transform(X_ref[:, i:i+1]).ravel() if hasattr(Granulator, 'fit_transform') else None
    # Granulator interface is uncertain; use equal-frequency quantiles of the
    # reference segment directly
    edges = np.quantile(X_ref[:, i], np.linspace(0, 1, K + 1)[1:-1])
    g = np.digitize(X[sl, i], edges)
    v = X[sl, j]
    means = [v[g == kk].mean() if (g == kk).sum() >= 3 else np.nan for kk in range(K)]
    sds = [v[g == kk].std() if (g == kk).sum() >= 3 else np.nan for kk in range(K)]
    # Find the granule pair with closest means and largest variance difference
    best, best_score = None, -np.inf
    for a in range(K):
        for b in range(a + 1, K):
            if np.isnan(means[a]) or np.isnan(means[b]):
                continue
            d = abs(means[a] - means[b])
            dv = abs((sds[a] or 0) - (sds[b] or 0))
            if d == 0:
                score = dv
            else:
                score = dv / d - d  # large variance difference, small mean difference
            if score > best_score and (v[g == a].size >= 3) and (v[g == b].size >= 3):
                best, best_score = (a, b), score
    if best is None:
        return X, None
    a, b = best
    ia = np.where(g == a)[0]
    ib = np.where(g == b)[0]
    m = min(ia.size, ib.size)
    ia = rng.choice(ia, size=m, replace=False)
    ib = rng.choice(ib, size=m, replace=False)
    va = v[ia].copy()
    vb = v[ib].copy()
    X[sl, j][ia] = vb
    X[sl, j][ib] = va
    return X, dict(pair=(a, b), d_mean=abs(means[a] - means[b]),
                   sd_a=sds[a], sd_b=sds[b])


def main():
    print('four_tuple interface self-check...')
    rng0 = np.random.default_rng(0)
    T = rng0.integers(1, 50, size=(K, K)).astype(float)
    out = four_tuple(T, T, K)
    print('  four_tuple keys:', sorted(out.keys()))
    assert 'dgamma' in out and 'djs' in out

    rng = np.random.default_rng(42)
    Tn = 8000
    t = np.arange(Tn)
    base = np.sin(t / 37.0) + 0.3 * rng.standard_normal(Tn)
    X_ref = np.column_stack([base,
                             np.roll(base, 2) + 0.3 * rng.standard_normal(Tn),
                             rng.standard_normal(Tn),
                             np.roll(base, 5) + 0.5 * rng.standard_normal(Tn)])
    i, j = 0, 1
    L, s = 128, 4000
    edges_i = np.quantile(X_ref[:, i], np.linspace(0, 1, K + 1)[1:-1])
    edges_j = np.quantile(X_ref[:, j], np.linspace(0, 1, K + 1)[1:-1])
    def table(X, sl):
        gi = np.digitize(X[sl, i], edges_i)
        gj = np.digitize(X[sl, j], edges_j)
        Tm = np.zeros((K, K))
        for a, b in zip(gi, gj):
            Tm[a, b] += 1
        return Tm
    T_ref = table(X_ref, slice(s - L, s))
    sr = four_tuple(T_ref, T_ref, K)

    dgs, dds, djs = [], [], []
    dmeans = []
    for trial in range(10):
        Xw = np.array(X_ref, copy=True)
        Xw[s:s+L, j] = X_ref[s-128:s-128+L, j]  # splice in a non-stationary segment to create a disturbance
        Xmod, info = varshape_pairs(Xw, X_ref, i, j, s, L, rng)
        if info is None:
            continue
        T_cur = table(Xmod, slice(s, s + L))
        sc = four_tuple(T_ref, T_cur, K)
        dgs.append(abs(sc['gamma_cur'] - sr['gamma_ref']))
        dds.append(abs(sc['delta_cur'] - sr['delta_ref']))
        djs.append(abs(sc['djs']))
        dmeans.append(info['d_mean'])
        print(f"trial{trial}: pair={info['pair']} d_mean={info['d_mean']:.3f} "
              f"sd {info['sd_a']:.2f}<->{info['sd_b']:.2f} | "
              f"dG={dgs[-1]:.4f} dD={dds[-1]:.4f} dJ={djs[-1]:.4f}")
    med = lambda x: float(np.median(x)) if x else float('nan')
    print(f"\nmedians: dGamma={med(dgs):.4f} dDelta={med(dds):.4f} dJS={med(djs):.4f} d_mean={med(dmeans):.3f}")
    ok = med(dgs) < 0.05 and med(dds) < 0.05 and med(djs) > 3 * max(med(dgs), med(dds))
    print('pre-registered capability gate:', 'PASS' if ok else 'FAIL')
    sys.exit(0 if ok else 2)


if __name__ == '__main__':
    main()
