# -*- coding: utf-8 -*-
"""RDSG detector: sliding-window structural-break detection + event-aligned typing
(two stages).

Null-distribution calibration (reference segment, preventing surrogate contamination):
- Detection statistic z_e = max_c |Δs_c / σ_c|; σ_c and the maxT threshold are all taken
  from adjacent sliding window pairs within the reference segment (H0 holds, no injection
  contamination);
- Alarm = z_max crossing the reference-segment maxT threshold (family-level FWER,
  α = maxt_alpha);
- Only at alarmed boundaries is a local block permutation run -> BH-FDR edge localization
  (serves typing).
Typing: the tuple is recomputed on event-aligned windows (detection handles recall,
typing handles the type).
"""
import numpy as np
from typing import List, Optional
from .granulation import Granulator
from .tables import Edge, block_tables, window_sum, edge_code
from .stats import four_tuple, gamma_soft, std_gamma
from .testing import BlockPermutationTest, permuted_window_pairs

INV, BRK, EMB = "inversion", "breakdown", "emergence"
_COMPS = ("dgamma", "rho", "ddelta", "djs")


def build_pair_edges(channels: List[int], taus: List[int]) -> List[Edge]:
    """All pairwise edges: every ordered channel pair at every lag."""
    edges = []
    for d in channels:
        for i in channels:
            if i == d:
                continue
            for tau in taus:
                edges.append(((i,), d, tau))
    return edges


class RDSGDetector:
    def __init__(self, K=4, w=512, blk=None, B=499, q=0.05,
                 maxt_alpha=0.01, step_blocks=1, gap_blocks=None, seed=0,
                 null_q=0.95, components=None, cal_pairs=200, pair_strata=None,
                 k_roll=4, n_subset=0):
        self.K, self.w, self.B, self.q = K, w, B, q
        self.blk = blk if blk is not None else max(8, w // 8)
        # Default event merge radius = one window length (absorbs trailing alarms produced
        # by the exit-side contrast)
        self.gap_blocks = gap_blocks if gap_blocks is not None else self.w // self.blk
        self.step_blocks = step_blocks
        self.seed, self.null_q, self.maxt_alpha = seed, null_q, maxt_alpha
        self.components = components or _COMPS
        self.cal_pairs = cal_pairs
        self.pair_strata = pair_strata  # (m_strong, m_weak) stratification of candidate pairs; None = all pairs
        self.k_roll = k_roll            # window length (in boundaries) for fixed-window mean evidence aggregation
        self.n_subset = n_subset        # number of binary subset edges per target for channel E (EMB-2 detection)

    # ---------- fitting ----------
    def fit(self, X_ref: np.ndarray, channels=None, taus=None, force_pairs=None):
        self.gran_ = Granulator(self.K).fit(X_ref)
        cand = channels if channels is not None else list(range(X_ref.shape[1]))
        # Drop dead channels whose granules collapse on the reference segment (constant /
        # single-valued sensors): their edge tuples degenerate (sd=0) and would contaminate
        # the maxT family threshold with z ~ Δ/ε. Channels dropped for E0 are reported
        # faithfully in the preprocessing section of the paper.
        G_all = self.gran_.transform(X_ref)
        alive = [c for c in cand
                 if len(np.unique(G_all[:, c])) >= 2]
        self.dead_channels_ = sorted(set(cand) - set(alive))
        self.channels_ = alive
        Gref = self.gran_.transform(X_ref[-2 * self.w:])
        # Use a long window for stratification / best-tau selection (2w can be as few as
        # 256 points; small-sample gamma noise would rank independent pairs into the strong pool)
        Gsel = self.gran_.transform(X_ref[-max(2048, 2 * self.w):])
        self.taus_ = taus if taus is not None else list(range(-3, 4))
        if self.pair_strata is not None and force_pairs is not None:
            # Candidate graph supplied externally (v5.2: the graph is defined once at the
            # main granularity; multi-granularity configurations reuse the same graph)
            strong, weak = force_pairs
            self.strong_pairs_ = strong
            self.weak_pairs_ = weak
            self.candidate_pairs_ = strong + weak
            best_tau = {}
            for (i, j) in strong:
                taus_l, tv = [], []
                for tau in self.taus_:
                    code = edge_code(Gsel, (i,), j, tau, self.K)
                    c = code[code >= 0]
                    tab = np.bincount(c, minlength=self.K * self.K).reshape(self.K, self.K).astype(float)
                    taus_l.append(tau); tv.append(std_gamma(tab[None])[0])
                best_tau[(i, j)] = int(taus_l[int(np.argmax(tv))])
            edges = []
            for (i, j) in strong:
                t0 = best_tau[(i, j)]
                for tau in (t0 - 1, t0, t0 + 1):
                    if tau in self.taus_:
                        edges.append(((i,), j, tau))
                        if tau != 0:
                            edges.append(((j,), i, tau))
            self.edges_R_ = edges
            edges_E = []
            for (i, j) in weak:
                edges_E.append(((i,), j, 0))
            from collections import defaultdict
            by_t = defaultdict(list)
            for (i, j) in weak:
                by_t[j].append(i)
            self.subset_parents_ = {}
            for d, parents in by_t.items():
                ps = sorted(set(parents))[: self.n_subset]
                self.subset_parents_[d] = ps
                for a in range(len(ps)):
                    for b in range(a + 1, len(ps)):
                        edges_E.append(((ps[a], ps[b]), d, 0))
            self.edges_E_ = edges_E
            self.edges_ = edges + edges_E
        elif self.pair_strata is not None:
            # Two-channel edge sets (v3): binned by reference-segment gamma_soft.
            # Channel R (breakdown/inversion): top-m strong pairs, each keeping only the
            #   reference-segment best tau and its ±1 (compact family; strong-pair breakdown
            #   signal z ≈ 17 vs family floor ≈ 5). Channel E (emergence): bottom-m' weak
            #   pairs, tau = 0 only.
            m_s, m_w = self.pair_strata
            gam = {}
            for a in range(len(alive)):
                for b in range(a + 1, len(alive)):
                    i, j = alive[a], alive[b]
                    code = edge_code(Gsel, (i,), j, 0, self.K)
                    c = code[code >= 0]
                    tab = np.bincount(c, minlength=self.K * self.K).reshape(self.K, self.K).astype(float)
                    gam[(i, j)] = float(std_gamma(tab[None])[0])
            order = sorted(gam, key=gam.get, reverse=True)
            strong = order[:m_s]
            weak = order[-m_w:] if m_w > 0 else []
            self.strong_pairs_ = strong
            self.weak_pairs_ = weak
            self.candidate_pairs_ = strong + weak
            # Strong pairs: pick the best lag by reference-segment gamma_soft across taus
            best_tau = {}
            for (i, j) in strong:
                taus, tv = [], []
                for tau in self.taus_:
                    code = edge_code(Gsel, (i,), j, tau, self.K)
                    c = code[code >= 0]
                    tab = np.bincount(c, minlength=self.K * self.K).reshape(self.K, self.K).astype(float)
                    taus.append(tau); tv.append(std_gamma(tab[None])[0])
                best_tau[(i, j)] = int(taus[int(np.argmax(tv))])
            edges = []
            for (i, j) in strong:
                t0 = best_tau[(i, j)]
                for tau in (t0 - 1, t0, t0 + 1):
                    if tau in self.taus_:
                        edges.append(((i,), j, tau))
                        if tau != 0:
                            edges.append(((j,), i, tau))
            self.edges_R_ = edges
            edges_E = []
            for (i, j) in weak:
                edges_E.append(((i,), j, 0))
            # EMB-2 subset edges: attach binary subset edges of the top-m weak partners to
            # each weak target (detects subset joint emergence)
            from collections import defaultdict
            by_t = defaultdict(list)
            for (i, j) in weak:
                by_t[j].append(i)
            self.subset_parents_ = {}
            for d, parents in by_t.items():
                ps = sorted(set(parents))[: self.n_subset]
                self.subset_parents_[d] = ps
                for a in range(len(ps)):
                    for b in range(a + 1, len(ps)):
                        edges_E.append(((ps[a], ps[b]), d, 0))
            self.edges_E_ = edges_E
            self.edges_ = edges + edges_E
        else:
            self.candidate_pairs_ = None
            self.edges_ = build_pair_edges(self.channels_, self.taus_)
        # Data-driven typing anchors v2: computed from the detector's own edge set
        # (channel R -> median of strong pairs = high anchor; channel E -> the floor itself
        # = low anchor). The all-pair median would fall on the floor and stall the EMB rule.
        pair_set = {}
        for (A, d, tau) in self.edges_:
            pair_set[(A[0], d)] = None
        vals = []
        for (i, d) in pair_set:
            code = edge_code(Gsel, (i,), d, 0, self.K)
            c = code[code >= 0]
            tab = np.bincount(c, minlength=self.K * self.K).reshape(self.K, self.K).astype(float)
            vals.append(float(std_gamma(tab[None])[0]))
        v = np.asarray(vals)
        self.gamma_hi_ = max(float(np.median(v)), float(np.quantile(v, 0.6)))
        self.gamma_lo_ = float(np.quantile(v, 0.25)) + 0.02
        if self.gamma_lo_ >= self.gamma_hi_:
            self.gamma_lo_ = self.gamma_hi_ - 0.04
        return self

    def granulate(self, X):
        return self.gran_.transform(X)

    # ---------- reference-segment calibration ----------
    def _obs_stack(self, obs):
        """Component stack: rho is negated (a drop = breakdown signal); D_JS uses log1p to
        tame the non-negative right-skewed tail."""
        sign = {"dgamma": 1.0, "rho": -1.0, "ddelta": 1.0, "djs": 1.0}
        return np.stack([sign.get(c, 1.0) * (np.log1p(obs[c]) if c == "djs" else obs[c])
                         for c in self.components], axis=1)  # (E, C)

    def calibrate(self, G: np.ndarray, t_ref_end: int, verbose=False, cal_span=None,
                  cal_t0=None):
        """Reference-segment calibration (H0 guaranteed to hold; no surrogate contamination).

        v2 changes:
        - cal_t0: start of the mu/sigma estimation span (kept separate from the self-scan span,
          preventing the in-sample optimism of the self-scan);
        - unit validity mask: degenerate units are dropped from the family by the per-component
          median sigma (sigma < 5% of the median), and the sigma floor is 0.2 x the component
          median -- heavy-tailed units no longer contaminate the family maximum.
        """
        K, blk, half = self.K, self.blk, self.w // self.blk
        E = len(self.edges_)
        if cal_t0 is None:
            cal_t0 = max(0, t_ref_end - cal_span) if cal_span else 0
        # ---- mu/sigma estimation: sliding window pairs ----
        tab = block_tables(G, self.edges_, K, blk, t0=cal_t0, t1=t_ref_end)
        nb = tab.shape[1]
        starts = np.arange(0, nb - 2 * half + 1)
        if len(starts) > self.cal_pairs:
            starts = starts[np.linspace(0, len(starts) - 1, self.cal_pairs).astype(int)]
        deltas = []
        for s in starts:
            Tr = window_sum(tab, np.arange(s, s + half))
            Tc = window_sum(tab, np.arange(s + half, s + 2 * half))
            st = four_tuple(Tr, Tc, K)
            deltas.append(self._obs_stack(st))  # (E, C)
        D = np.stack(deltas, axis=0)  # (J, E, C)
        # Per-component center+scale calibration: rho/djs are level quantities (rho ≈ 1 on
        # deterministic edges), dgamma/ddelta are difference quantities; standardize (x-μ)/σ
        # uniformly per edge and component, otherwise dividing a level quantity by its own
        # variance blows up into spurious z.
        self.mu_ = D.mean(axis=0)                       # (E, C)
        # v2 unit mask: units with sigma < 5% x the component median are dropped from the
        # family; sigma floor = 0.2 x the component median -- heavy-tailed/degenerate units
        # no longer contaminate the family maximum (a main cause of v1 threshold inflation)
        sd_raw = D.std(axis=0)
        med_c = np.median(sd_raw, axis=0)               # (C,)
        self.valid_units_ = sd_raw > (0.05 * np.maximum(med_c, 1e-9))[None, :]
        self.sd_ = np.maximum(sd_raw, (0.2 * np.maximum(med_c, 1e-9))[None, :])
        # ---- main null distribution: sliding split samples (same construction as the
        # detection statistic: adjacent-window contrast, preserving autocorrelation) ----
        zsplit = (np.abs((D - self.mu_) / self.sd_) * self.valid_units_).max(axis=2)  # (J, E)
        self.znull_ = zsplit
        self.zmax_null_ = zsplit.max(axis=1)
        self.alarm_thr_ = float(np.quantile(self.zmax_null_, 1.0 - self.maxt_alpha))
        # ---- block-permutation null samples (sensitivity variant: reshuffling is more
        # conservative; reported in parallel as-is) ----
        J2 = 4
        anchors = starts[np.linspace(0, len(starts) - 1, J2).astype(int)]
        znull = []  # (J2*B, E)
        for a in anchors:
            poolT = np.ascontiguousarray(
                tab[:, a:a + 2 * half].transpose(1, 0, 2, 3)
                .reshape(2 * half, E * tab.shape[2] * K)).astype(np.float32)
            perms = permuted_window_pairs(2 * half, half, self.B, self.seed + a)
            Wref = np.zeros((self.B, 2 * half), dtype=np.float32)
            Wcur = np.zeros((self.B, 2 * half), dtype=np.float32)
            for b, (ri, ci) in enumerate(perms):
                Wref[b, ri] = 1.0
                Wcur[b, ci] = 1.0
            PREF = (Wref @ poolT).reshape(self.B, E, tab.shape[2], K)
            PCUR = (Wcur @ poolT).reshape(self.B, E, tab.shape[2], K)
            st_perm = self._batch_stats(PREF, PCUR)
            z = np.abs((self._obs_stack_perm(st_perm) - self.mu_) / self.sd_) * self.valid_units_  # (B,E,C)
            znull.append(z.max(axis=2))
            if verbose:
                print(f"  perm anchor {a}/{starts[-1]}", flush=True)
        self.znull_blk_ = np.concatenate(znull, axis=0)       # (N, E) sensitivity
        self.alarm_thr_blk_ = float(np.quantile(self.znull_blk_.max(axis=1),
                                                1.0 - self.maxt_alpha))
        # ---- unit-level standardization parameters for rolling-mean evidence aggregation
        # (lever 1) ----
        # k-term rolling mean of the calibration-span z series (stride=1 blocks)
        # -> mu-bar / sigma-bar per edge
        tc, Zc = self.z_series(G, cal_t0, t_ref_end)
        if len(tc) >= self.k_roll:
            Zm = np.vstack([Zc[max(0, i - self.k_roll + 1):i + 1].mean(axis=0)
                            for i in range(len(tc))])
            self.roll_mu_ = Zm.mean(axis=0)
            sd_roll = Zm.std(axis=0)
            med_r = float(np.median(sd_roll)) or 1e-6
            self.roll_sd_ = np.maximum(sd_roll, 0.05 * med_r)
        else:  # calibration span too short; fall back to point statistics
            self.roll_mu_ = np.zeros(len(self.edges_))
            self.roll_sd_ = np.ones(len(self.edges_))
        if verbose:
            print(f"  calibrated: J={len(starts)} splits -> alarm_thr={self.alarm_thr_:.2f} "
                  f"(block-perm sensitivity thr={self.alarm_thr_blk_:.2f})", flush=True)
        return self

    def _obs_stack_perm(self, st_perm):
        sign = {"dgamma": 1.0, "rho": -1.0, "ddelta": 1.0, "djs": 1.0}
        return np.stack([sign.get(c, 1.0) * (np.log1p(st_perm[c]) if c == "djs" else st_perm[c])
                         for c in self.components], axis=2)  # (B,E,C)

    def _pvals(self, z_obs: np.ndarray) -> np.ndarray:
        """Edge-level p-values: parametric two-sided tails from the calibrated (μ, σ)
        (the empirical sample granularity cannot support BH over tens of thousands of edges)."""
        from scipy.stats import norm as _norm
        return np.maximum(2.0 * _norm.sf(z_obs), 1e-12)

    # ---------- Stage 1: sliding-window detection ----------
    def detect(self, G: np.ndarray, t_ref_end: int, verbose=False):
        K, blk, half = self.K, self.blk, self.w // self.blk
        self._ref_len = t_ref_end
        if not hasattr(self, "sd_"):
            self.calibrate(G, t_ref_end, verbose)
        span0 = max(0, t_ref_end - self.w)
        span1 = G.shape[0]
        tab = block_tables(G, self.edges_, K, blk, t0=span0, t1=span1)
        E, nb = len(self.edges_), tab.shape[1]
        self._tab_, self._span0_ = tab, span0
        records = []
        k0 = (t_ref_end - span0) // blk
        k = k0
        while k + half <= nb:
            Tr = window_sum(tab, np.arange(k - half, k))
            Tc = window_sum(tab, np.arange(k, k + half))
            st = four_tuple(Tr, Tc, K)
            z = np.abs((self._obs_stack(st) - self.mu_) / self.sd_) * self.valid_units_
            zmax = float(z.max())
            if zmax > self.alarm_thr_:
                records.append(dict(t=span0 + k * blk, zmax=zmax, thr=self.alarm_thr_,
                                    sig_idx=None, n_sig=0, obs=None))
            k += self.step_blocks
            if verbose and (k - k0) % 80 == 0:
                print(f"  boundary {k}/{nb} (alarms={len(records)})", flush=True)
        events = self._merge_events(records)
        return events, records

    def zmean_series(self, G: np.ndarray, t_start: int, t_end: int):
        """Fixed-window mean evidence aggregation series (lever 1):

        The k_roll-boundary rolling mean of each edge's z, standardized by the
        calibration-span rolling distribution (zbar - mubar) / sigmabar. H0 spikes are
        diluted by the averaging (family floor drops) while sustained event elevations keep
        their signal (power rises). Returns (times, Zstd).
        """
        t, Z = self.z_series(G, t_start, t_end)
        if len(t) == 0:
            return t, Z
        Zm = np.vstack([Z[max(0, i - self.k_roll + 1):i + 1].mean(axis=0)
                        for i in range(len(t))])
        Zstd = (Zm - self.roll_mu_) / self.roll_sd_
        return t, Zstd

    def zfull_series(self, G: np.ndarray, t_start: int, t_end: int):
        """Unit-level full series (for component-family calibration): (times, Z), where
        Z[i, e, c] = standardized z of component c of edge e at boundary i."""
        K, blk, half = self.K, self.blk, self.w // self.blk
        span0 = max(0, t_start - self.w)
        tab = block_tables(G, self.edges_, K, blk, t0=span0, t1=t_end)
        nb = tab.shape[1]
        self._tab_, self._span0_ = tab, span0
        k0 = (t_start - span0 + blk - 1) // blk
        times, Zs = [], []
        k = k0
        while k + half <= nb:
            Tr = window_sum(tab, np.arange(k - half, k))
            Tc = window_sum(tab, np.arange(k, k + half))
            st = four_tuple(Tr, Tc, K)
            z = np.abs((self._obs_stack(st) - self.mu_) / self.sd_) * self.valid_units_
            times.append(span0 + k * blk)
            Zs.append(z)
            k += self.step_blocks
        return np.asarray(times), np.asarray(Zs)

    def prune_units_by_scan(self, G: np.ndarray, t0: int, t1: int, z_cap: float = 6.5):
        """Reference-segment self-scan pruning: edges with any boundary z > z_cap during the
        self-scan are dropped from the family (edge level).

        These edges actually violate the (μ, σ) standardization assumption when migrating
        from the calibration span to the scan span (heavy-tailed sources); the decision uses
        reference-segment information only, which is legitimate, and θ falls accordingly.
        The number of pruned edges is recorded in pruned_edges_.
        """
        t, Z = self.zmean_series(G, t0, t1)
        noisy = Z.max(axis=0) > z_cap
        self.valid_units_[noisy, :] = False
        self.pruned_edges_ = [self.edges_[i] for i in np.where(noisy)[0]]
        return int(noisy.sum())

    # ---------- v2: standard scan-series interface (for reference-segment self-scan to set
    # θ / multi-configuration union) ----------
    def T_series(self, G: np.ndarray, t_start: int, t_end: int):
        """Returns (times, T): boundaries b in [t_start, t_end), window pair [b-w, b) vs
        [b, b+w); T(b) = maximum standardized statistic over valid units. Also caches the
        block table for reuse by localize."""
        K, blk, half = self.K, self.blk, self.w // self.blk
        span0 = max(0, t_start - self.w)
        tab = block_tables(G, self.edges_, K, blk, t0=span0, t1=t_end)
        E, nb = len(self.edges_), tab.shape[1]
        self._tab_, self._span0_ = tab, span0
        k0 = (t_start - span0 + blk - 1) // blk
        times, Ts = [], []
        k = k0
        while k + half <= nb:
            Tr = window_sum(tab, np.arange(k - half, k))
            Tc = window_sum(tab, np.arange(k, k + half))
            st = four_tuple(Tr, Tc, K)
            z = np.abs((self._obs_stack(st) - self.mu_) / self.sd_) * self.valid_units_
            times.append(span0 + k * blk)
            Ts.append(float(z.max()))
            k += self.step_blocks
        return np.asarray(times), np.asarray(Ts)

    def z_series(self, G: np.ndarray, t_start: int, t_end: int):
        """Returns (times, Ze): Ze[i, e] = maximum valid-component z of edge e at boundary i
        (after masking)."""
        K, blk, half = self.K, self.blk, self.w // self.blk
        span0 = max(0, t_start - self.w)
        tab = block_tables(G, self.edges_, K, blk, t0=span0, t1=t_end)
        nb = tab.shape[1]
        self._tab_, self._span0_ = tab, span0
        k0 = (t_start - span0 + blk - 1) // blk
        times, Zs = [], []
        k = k0
        while k + half <= nb:
            Tr = window_sum(tab, np.arange(k - half, k))
            Tc = window_sum(tab, np.arange(k, k + half))
            st = four_tuple(Tr, Tc, K)
            z = np.abs((self._obs_stack(st) - self.mu_) / self.sd_) * self.valid_units_
            times.append(span0 + k * blk)
            Zs.append(z.max(axis=1))
            k += self.step_blocks
        return np.asarray(times), np.asarray(Zs)
    def localize(self, G: np.ndarray, events):
        K, blk, half = self.K, self.blk, self.w // self.blk
        tab = self._tab_
        for ev in events:
            first = min(ev["members"], key=lambda m: m["t"])  # onset-side alarm: direction is correct
            k = (first["t"] - self._span0_) // blk
            Tr = window_sum(tab, np.arange(k - half, k))
            Tc = window_sum(tab, np.arange(k, k + half))
            st = four_tuple(Tr, Tc, K)
            z = (np.abs((self._obs_stack(st) - self.mu_) / self.sd_) * self.valid_units_).max(axis=1)  # (E,)
            pvals = self._pvals(z)
            m = len(pvals)
            order = np.argsort(pvals)
            ranked = pvals[order] * m / (np.arange(m) + 1)
            ok = ranked <= self.q
            sig = np.zeros(m, bool)
            if ok.any():
                sig[order[: np.max(np.where(ok)[0]) + 1]] = True
            else:
                # When FDR is overly conservative (BH selects nothing), fall back to the
                # top-10 z edges for attribution -- alarms are already guaranteed by the
                # family-level threshold; attribution only serves typing
                top = np.argsort(-z)[:10]
                sig[top] = True
            for mm in ev["members"]:
                mm["sig_idx"] = np.where(sig)[0]
                mm["n_sig"] = int(sig.sum())
                mm["obs"] = {kk: vv.copy() for kk, vv in st.items()}
        return events

    def _batch_stats(self, PREF, PCUR):
        out = {}
        for key in ("dgamma", "rho", "ddelta", "djs"):
            vals = []
            for lo in range(0, PREF.shape[0], 64):
                st = four_tuple(PREF[lo:lo + 64], PCUR[lo:lo + 64], self.K)
                vals.append(st[key])
            out[key] = np.concatenate(vals, axis=0)
        return out

    def _merge_events(self, records):
        if not records:
            return []
        events, cur = [], dict(t0=records[0]["t"], t1=records[0]["t"] + self.blk,
                               members=[records[0]])
        for r in records[1:]:
            if r["t"] - cur["t1"] <= self.gap_blocks * self.blk:
                cur["t1"] = r["t"] + self.blk
                cur["members"].append(r)
            else:
                events.append(cur)
                cur = dict(t0=r["t"], t1=r["t"] + self.blk, members=[r])
        events.append(cur)
        return events

    # ---------- Stage 2: event-aligned typing (anchored at the first alarm boundary,
    # two-scale windows) ----------
    def type_events(self, G: np.ndarray, events, min_len=None, anchors=None):
        """v5.2: dual anchors (event onset boundary + peak boundary within the event) x
        two-scale window search for typing.

        Rolling-mean evidence makes alarms persist, so merged events span far more than the
        injection length; with a single onset-boundary anchor the aligned windows mostly
        contain non-event data and the typing statistics get diluted. For each candidate
        anchor x both scales, the contrast z is computed and the strongest is used for
        typing.
        """
        min_len = min_len or max(2 * self.blk, 128)
        cgamma, crho, cdelta, cjs = self._typing_thresholds(G)
        typed = []
        for ev in events:
            b0 = min(ev["members"], key=lambda m: m["t"])["t"]  # onset-side alarm: the reference side is clean
            b_list = [b0] + [a for a in (anchors or []) if a > b0]
            idx = self._top_edges(ev, k=10)
            if not idx or b0 - self.w < 0:
                ev["type"] = "untyped"
                typed.append(ev)
                continue
            best = None
            for b in b_list:
              for Lw in (self.w // 2, self.w // 4):
                c0, c1 = b, b + Lw
                r0 = b - Lw
                if c1 > G.shape[0]:
                    continue
                zs = []
                for ei in idx:
                    A, d, tau = self.edges_[ei]
                    code = edge_code(G, A, d, tau, self.K)
                    KA = self.K ** len(A)
                    Tc = _code_table(code, np.arange(c0, c1), KA, self.K)
                    Tr = _code_table(code, np.arange(r0, c0), KA, self.K)
                    if Tr.sum() < self.blk or Tc.sum() < self.blk:
                        continue
                    st = four_tuple(Tr[None], Tc[None], self.K)
                    z = float((np.abs((self._obs_stack(st)[0] - self.mu_[ei])
                                     / self.sd_[ei]) * self.valid_units_[ei]).max())
                    zs.append((z, st, ei))
                if zs and (best is None or np.mean([x[0] for x in zs]) > best[0]):
                    best = (float(np.mean([x[0] for x in zs])), zs)
            votes = []
            vote_detail = []
            if best:
                _, zs = best
                # Peak-edge typing (channel-aware, v5.1): channel R only sees breakdown/
                # inversion (hosted on strong pairs); channel E only sees emergence (a
                # significant rise of γ on a weak-pair host means a new dependency forms).
                zs_sorted = sorted(zs, key=lambda x: -x[0])
                def _branch(st):
                    """v3 component-attribution typing: detection has already confirmed a
                    structural change; attribute the type by the direction of the dominant signal."""
                    dg = float(st["dgamma"][0])
                    channel = getattr(self, "channel_kind_", "R")
                    if channel == "E":
                        return EMB  # weak-pair emergence pool: any change is a new dependency
                    if dg < 0:
                        return BRK   # γ drop = loss of dependency
                    return INV   # γ not down but other components changed = direction/conditional-distribution reshaping
                peak_type = _branch(zs_sorted[0][1])
                ev["peak_edge"] = self.edges_[zs_sorted[0][2]]
                for zval, st, ei in zs_sorted[:5]:
                    vote_detail.append(_branch(st))
                if peak_type is None and vote_detail:
                    nn = [v for v in vote_detail if v]
                    peak_type = nn[0] if nn else None
                if peak_type:
                    votes.append(peak_type)
                    ev["type_votes"] = {str(v): int(vote_detail.count(v))
                                        for v in set(vote_detail) if v}
            if votes:
                vals, cnts = np.unique(votes, return_counts=True)
                ev["type"] = str(vals[np.argmax(cnts)])
                ev["type_votes"] = {str(v): int(c) for v, c in zip(vals, cnts)}
            else:
                ev["type"] = "untyped"
            typed.append(ev)
        return typed

    def _top_edges(self, ev, k=10):
        cnt = {}
        for m in ev["members"]:
            for ei in m["sig_idx"]:
                cnt[ei] = cnt.get(ei, 0) + 1
        rank = sorted(cnt, key=lambda e: -cnt[e])
        return rank[:k]

    def _typing_thresholds(self, G):
        """Quantiles of adjacent-window-pair tuple differences within the reference segment
        (no test-set parameter selection)."""
        K, L = self.K, self.w
        ref_end = getattr(self, "_ref_len", G.shape[0] - self.w)
        ds = {kk: [] for kk in ("dgamma", "rho", "ddelta", "djs")}
        t = max(0, ref_end - 4 * L)
        rng = np.random.default_rng(self.seed + 7)
        while t + 2 * L <= ref_end:
            sel1 = np.arange(t, t + L)
            sel2 = np.arange(t + L, t + 2 * L)
            for ei in rng.choice(len(self.edges_), size=min(60, len(self.edges_)), replace=False):
                A, d, tau = self.edges_[ei]
                code = edge_code(G, A, d, tau, K)
                KA = K ** len(A)
                T1 = _code_table(code, sel1, KA, K)
                T2 = _code_table(code, sel2, KA, K)
                if T1.sum() < self.blk or T2.sum() < self.blk:
                    continue
                st = four_tuple(T1[None], T2[None], K)
                for kk in ds:
                    ds[kk].append(float(st[kk][0]))
            t += L
        q = self.null_q
        cgamma = max(float(np.quantile(np.abs(ds["dgamma"]), q)) if ds["dgamma"] else 0.05, 0.03)
        crho = max(float(np.quantile(ds["rho"], 1 - q)) if ds["rho"] else 0.3, 0.05)
        cdelta = max(float(np.quantile(np.abs(ds["ddelta"]), q)) if ds["ddelta"] else 0.3, 0.1)
        cjs = max(float(np.quantile(np.abs(ds["djs"]), q)) if ds["djs"] else 0.2, 0.05)
        return cgamma, crho, cdelta, cjs


def _code_table(code: np.ndarray, sel: np.ndarray, KA: int, K: int) -> np.ndarray:
    c = code[sel]
    c = c[c >= 0]
    return np.bincount(c, minlength=KA * K).reshape(KA, K).astype(float)
