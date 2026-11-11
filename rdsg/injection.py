# -*- coding: utf-8 -*-
"""Margin-preserving relational-structure injection protocol (construction basis for the
E0/E2 mechanism-specific experiments; Reviewer Comment 4).

Principle (methods draft, Sec. 5, Proposition 2): only the observed values of channel j
within the window are permuted -- the empirical marginal distribution is unchanged
pointwise (same set of values, only their order changes), and only the joint dependence
changes. For point detectors (which consume only per-channel marginals / per-channel
residuals), recall on such injections equals the false-alarm rate.
"""
import numpy as np
from scipy.stats import norm


def _copula_reorder(x_parent: np.ndarray, y_vals: np.ndarray, rho_target: float, rng) -> np.ndarray:
    """Gaussian copula rank mapping: preserves the multiset of y_vals (marginal exactly
    unchanged) while targeting Spearman ~= rho_target."""
    L = len(y_vals)
    u = (np.argsort(np.argsort(x_parent)) + 0.5) / L
    z = norm.ppf(np.clip(u, 1e-6, 1 - 1e-6))
    z2 = rho_target * z + np.sqrt(max(1 - rho_target ** 2, 1e-12)) * rng.standard_normal(L)
    v = norm.cdf(z2)
    order = np.argsort(v)                      # target ranks
    y_sorted = np.sort(y_vals)
    out = np.empty(L)
    out[order] = y_sorted                      # reorder the order statistics of y by target rank
    return out


def inject(X_test: np.ndarray, specs: list, seed: int = 0) -> np.ndarray:
    """specs: [{'type': 'inv'|'brk'|'emb'|'emb2', 'i': i, 'j': j, 'i2': i2(emb2), 'start': s, 'length': L}, ...]
    Returns an injected copy (the input array is untouched). Every type permutes the values
    of j within the window, so the marginals are exactly preserved.
    """
    rng = np.random.default_rng(seed)
    X = np.array(X_test, copy=True)
    for sp in specs:
        s, L = sp["start"], sp["length"]
        sl = slice(s, s + L)
        i, j = sp["i"], sp["j"]
        t = sp["type"]
        if t == "inv":
            # Sign-inversion protocol (v3): the inversion target is the negated
            # reference-segment rank correlation (floor |rho*| >= 0.6 preserves the flip
            # magnitude); fixes the protocol bug where the fixed -0.9 target was a no-op
            # for pairs natively negatively correlated.
            rt = sp.get("rho_target")
            if rt is None:
                rt = -0.9
            X[sl, j] = _copula_reorder(X[sl, i], X[sl, j], rt, rng)
        elif t == "brk":
            X[sl, j] = rng.permutation(X[sl, j])
        elif t == "emb":
            X[sl, j] = _copula_reorder(X[sl, i], X[sl, j], +0.9, rng)
        elif t == "emb2":
            # Subset joint emergence, parity construction instantiating Prop 1(iii):
            # j is reordered so that its rank order follows (grain(i1)+grain(i2)) mod K.
            # Given either single parent granule the key classes are exactly uniform
            # (Latin-square property), so pair tables stay at their independence floors,
            # while the subset table over both parent granules concentrates strongly.
            # The values of j are only permuted: the window marginal is preserved exactly.
            Lw = L
            K = int(sp.get("K", 4))
            g1 = (np.argsort(np.argsort(X[sl, sp["i2"][0]])) * K) // Lw
            g2 = (np.argsort(np.argsort(X[sl, sp["i2"][1]])) * K) // Lw
            key = (g1 + g2) % K + 0.01 * rng.standard_normal(Lw)  # tie-break within class
            X[sl, j] = np.sort(X[sl, j])[np.argsort(np.argsort(key))]
        else:
            raise ValueError(t)
    return X


def sample_specs(X_ref_tail, strong_pairs, weak_pairs, n_each: int,
                 t_len: int, margin: int, lengths=(30, 60, 120), seed: int = 0):
    """Samples injection events from channel pairs binned by reference-segment gamma:
    strong pairs -> INV/BRK, weak pairs -> EMB. Events are non-overlapping with margins kept."""
    rng = np.random.default_rng(seed)
    specs = []
    occupied = np.zeros(t_len, bool)
    def _place(L):
        for _ in range(200):
            s = rng.integers(margin, max(margin + 1, t_len - margin - L))
            if not occupied[s - margin:s + L + margin].any():
                occupied[s:s + L] = True
                return int(s)
        return None
    plans = ([("inv", p) for p in strong_pairs[:n_each]] +
             [("brk", p) for p in strong_pairs[n_each:2 * n_each]] +
             [("emb", p) for p in weak_pairs[:n_each]])
    plans = [plans[i] for i in rng.permutation(len(plans))]  # shuffle: keeps later types from being starved of space
    for t, (i, j) in plans:
        L = int(rng.choice(lengths))
        s = _place(L)
        if s is None:
            continue
        specs.append(dict(type=t, i=int(i), j=int(j), start=s, length=L))
    return specs
