# -*- coding: utf-8 -*-
"""Tuple edge statistics: γ_β (variable-precision rough-set dependency degree),
ρ (rule-mapping preservation), δ (granular-rank direction), D_JS (Jensen-Shannon
divergence of conditional residual distributions).

All are computed from contingency tables and can be batch-vectorized over (E, ...).
s(A→d) = (γ, ρ, δ, D_JS); γ measures dependency strength, while ρ/δ/D_JS cover the
mapping-inversion blind spot of γ (Reviewer Comment 1).
"""
import numpy as np

_EPS = 1e-12


def _as_prob(tab: np.ndarray):
    """counts (..., KA, K) -> probabilities and row marginals."""
    tot = tab.sum(axis=(-2, -1), keepdims=True)
    P = tab / np.maximum(tot, _EPS)
    pia = P.sum(axis=-1)  # (..., KA) row occupancy
    return P, pia


def gamma_beta(tab: np.ndarray, beta: float = 0.9) -> np.ndarray:
    """VPRS dependency degree (hard-threshold version, Ziarko 1993): row g enters the
    β-positive region iff the maximum within-row decision share >= β."""
    P, pia = _as_prob(tab)
    rowmax = P.max(axis=-1)
    pure = rowmax >= beta * np.maximum(pia, _EPS) * (pia > 0)
    return np.where(pia > 0, pure * pia, 0.0).sum(axis=-1)


def gamma_soft(tab: np.ndarray) -> np.ndarray:
    """Softened variable-precision dependency degree (default for real-valued time series):

    γ = Σ_g π_g · max_v P(v|g) -- the row-occupancy-weighted average of the modal
    decision-rule confidences, in [1/K, 1]. Under independence it sits at the noise floor
    (slightly above 1/K for finite window length w; the permutation test calibrates this
    automatically); under deterministic dependency it approaches 1. Under inversion
    (Y=X -> Y=-X) rows remain concentrated on a single column, so γ is unchanged (the
    blind spot is preserved; ρ/δ/D_JS provide the backstop).
    """
    P, pia = _as_prob(tab)
    rowmax = P.max(axis=-1) / np.maximum(pia, _EPS)  # conditional modal confidence
    return np.where(pia > 0, rowmax * pia, 0.0).sum(axis=-1)


def modal_map(tab: np.ndarray) -> np.ndarray:
    """Modal rule mapping φ(g) = argmax_v P(v|g). Returns (..., KA); empty rows return -1."""
    P, pia = _as_prob(tab)
    amap = P.argmax(axis=-1)
    return np.where(pia > 0, amap, -1)


def rule_consistency(tab_ref: np.ndarray, tab_cur: np.ndarray) -> np.ndarray:
    """ρ = Σ_g π_ref(g)·1[φ_ref(g)=φ_cur(g)] (weighted by reference row occupancy)."""
    _, pia_ref = _as_prob(tab_ref)
    phi_ref = modal_map(tab_ref)
    phi_cur = modal_map(tab_cur)
    agree = (phi_ref == phi_cur) & (phi_ref >= 0) & (phi_cur >= 0)
    return np.where(pia_ref > 0, agree * pia_ref, 0.0).sum(axis=-1)


def grain_direction(tab: np.ndarray, K: int) -> np.ndarray:
    """δ = weighted Pearson correlation after centering the granule indices (a
    linear-linear association that captures the sign of an inversion)."""
    P, _ = _as_prob(tab)
    g = np.arange(tab.shape[-2]) - (K - 1) / 2.0
    v = np.arange(tab.shape[-1]) - (K - 1) / 2.0
    tot = P.sum(axis=(-2, -1)) + _EPS  # (...)
    pg = P.sum(axis=-1)  # (..., KA)
    pv = P.sum(axis=-2)  # (..., K)
    mg = (pg * g).sum(-1) / tot
    mv = (pv * v).sum(-1) / tot
    cov = np.einsum("...gv,g,v->...", P, g, v) / tot - mg * mv
    sg = np.sqrt(np.maximum((pg * (g ** 2)).sum(-1) / tot - mg ** 2, _EPS))
    sv = np.sqrt(np.maximum((pv * (v ** 2)).sum(-1) / tot - mv ** 2, _EPS))
    return cov / np.maximum(sg * sv, _EPS)


def cond_js(tab_ref: np.ndarray, tab_cur: np.ndarray) -> np.ndarray:
    """D_JS = Σ_g π_ref(g)·JSD(P_ref(·|g), P_cur(·|g)), with the natural logarithm as base."""
    P_ref, pia = _as_prob(tab_ref)
    P_cur, _ = _as_prob(tab_cur)
    cond_r = P_ref / np.maximum(pia[..., None], _EPS)
    cond_c = P_cur / np.maximum(np.maximum(P_cur.sum(-1, keepdims=True), _EPS), _EPS)
    # Only rows with reference occupancy > 0 are counted; for rows absent from the current
    # window the conditional distribution is undefined -> computed against the reference
    # itself (JSD = 0 contribution)
    valid = (pia > 0) & (P_cur.sum(-1) > 0)
    M = 0.5 * (cond_r + cond_c)
    def kl(p, m):
        return np.where(p > 0, p * (np.log(np.maximum(p, _EPS)) - np.log(np.maximum(m, _EPS))), 0.0)
    jsd = 0.5 * kl(cond_r, M).sum(-1) + 0.5 * kl(cond_c, M).sum(-1)
    jsd = np.nan_to_num(jsd, nan=0.0, posinf=0.0, neginf=0.0)
    return np.where(valid, pia * jsd, 0.0).sum(-1)


def ipf_uniform(tab: np.ndarray, iters: int = 2, alpha: float = 8.0) -> np.ndarray:
    """Table -> marginal-standardized probability table (a discrete copula transform via
    smoothing + IPF).

    Removes two artifacts: (a) γ_soft mistaking the target channel's marginal granule-
    occupancy skew for dependence; (b) inter-window marginal drift leaking into
    contingency-table noise. After standardization the independence floor is 1/K.
    First apply within-row additive-α smoothing (small-count rows shrink toward uniform;
    as n_g -> 0 the conditionals -> 1/K), then run a few IPF iterations to equalize the
    column marginals -- pure IPF divides by noisy row sums, which amplifies sampling noise
    on small windows (row n ≈ 30) and inflates the family maximum.
    """
    n_g = tab.sum(axis=-1, keepdims=True)                       # row counts
    P = (tab + alpha / tab.shape[-1]) / (n_g + alpha)           # within-row additive-α smoothing
    KA, K = tab.shape[-2], tab.shape[-1]
    for _ in range(iters):
        rs = P.sum(axis=-1, keepdims=True)
        P = P / np.maximum(rs, _EPS) / KA
        cs = P.sum(axis=-2, keepdims=True)
        P = P / np.maximum(cs, _EPS) / K
    return P


def std_gamma(tab: np.ndarray) -> np.ndarray:
    """γ_soft after marginal standardization (used for binning/anchors; consistent with
    the four_tuple convention)."""
    return gamma_soft(ipf_uniform(tab))


def four_tuple(tab_ref: np.ndarray, tab_cur: np.ndarray, K: int, beta: float = 0.9):
    """Returns a dict: per-component (E,) arrays + γ_ref/γ_cur/δ_ref/δ_cur (for typing).

    γ uses the softened variable-precision form (gamma_soft); the beta argument is kept
    only for compatibility (hard version in gamma_beta). Tables are first IPF
    marginal-standardized (copula scale: all four statistics are decoupled from the
    marginals).
    """
    tab_ref = ipf_uniform(tab_ref)
    tab_cur = ipf_uniform(tab_cur)
    g_ref = gamma_soft(tab_ref)
    g_cur = gamma_soft(tab_cur)
    d_ref = grain_direction(tab_ref, K)
    d_cur = grain_direction(tab_cur, K)
    return {
        "dgamma": g_cur - g_ref,
        "rho": rule_consistency(tab_ref, tab_cur),
        "ddelta": d_cur - d_ref,
        "djs": cond_js(tab_ref, tab_cur),
        "gamma_ref": g_ref, "gamma_cur": g_cur,
        "delta_ref": d_ref, "delta_cur": d_cur,
    }
