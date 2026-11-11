# -*- coding: utf-8 -*-
"""Block permutation test + stratified multiple comparison (Reviewer Comment 2).

Stratified design:
- Event alarms (family level): Westfall-Young maxT -- each permutation takes the maximum
  statistic over all edges, and its (1-α_maxt) quantile is the alarm threshold
  (controls the FWER over the edge family; obtained for free from the same batch of
  permutations);
- Edge localization (local): BH-FDR q=0.05, selecting significant edges only within
  already-alarmed boundaries (serves typing).

Edge statistic z_e = max_c |Δs_c / σ_c^π| (components standardized by the permutation
null distribution, with ρ negated).
"""
import numpy as np

_ALL_COMPS = ["dgamma", "rho", "ddelta", "djs"]
_SIGN = {"dgamma": 1.0, "rho": -1.0, "ddelta": 1.0, "djs": 1.0}  # a drop in ρ is a breakdown signal -> negate it


class BlockPermutationTest:
    def __init__(self, B: int = 499, q: float = 0.05, maxt_alpha: float = 0.01):
        self.B, self.q, self.maxt_alpha = B, q, maxt_alpha

    def run(self, stat_obs: dict, stat_perm: dict, components=None):
        """stat_obs/stat_perm are keyed by dgamma/rho/ddelta/djs (rho is rule-mapping
        preservation, negated for testing: a drop in ρ = a structural-break signal).
        components: a subset of components (for ablation).
        """
        comps = list(components) if components else _ALL_COMPS
        obs = np.stack([_SIGN.get(c, 1.0) * stat_obs[c] for c in comps], axis=1)          # (E, C)
        per = np.stack([_SIGN.get(c, 1.0) * stat_perm[c] for c in comps], axis=2)         # (B, E, C)
        sd = per.std(axis=0)                                                             # (E, C)
        z_obs = np.abs(obs / np.maximum(sd, 1e-8)).max(axis=1)                            # (E,)
        z_perm = np.abs(per / np.maximum(sd, 1e-8)).max(axis=2)                           # (B, E)

        # Event alarms: maxT (edge-family FWER)
        zmax_obs = float(z_obs.max())
        null_max = z_perm.max(axis=1)
        maxt_thresh = float(np.quantile(null_max, 1.0 - self.maxt_alpha))
        alarm = zmax_obs > maxt_thresh

        # Edge localization: BH-FDR (only serves typing after alarms)
        m = z_obs.shape[0]
        pvals = (1.0 + (z_perm >= z_obs[None, :]).sum(axis=0)) / (self.B + 1.0)
        order = np.argsort(pvals)
        ranked = pvals[order] * m / (np.arange(m) + 1)
        ok = ranked <= self.q
        sig = np.zeros(m, bool)
        if ok.any():
            sig[order[: np.max(np.where(ok)[0]) + 1]] = True
        return dict(pvals=pvals, sig_fdr=sig, z_obs=z_obs, zmax_obs=zmax_obs,
                    maxt_thresh=maxt_thresh, alarm=bool(alarm), null_max=null_max)


def permuted_window_pairs(nblocks: int, half: int, B: int, seed: int = 0):
    """B pairs of (ref_idx, cur_idx): nblocks blocks randomly split in half."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(B):
        perm = rng.permutation(nblocks)
        out.append((np.sort(perm[:half]), np.sort(perm[half:])))
    return out
