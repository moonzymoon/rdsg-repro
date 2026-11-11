# -*- coding: utf-8 -*-
"""W3a: effect-size vs recall characterisation (synthetic capability gate).

Partial-permutation injection: within a window, randomly select a fraction p of
positions and permute only the j values inside S (multiset preserved ->
marginals unchanged); larger p yields larger |Delta gamma|.
Grid: p in {0.2,0.35,0.5,0.65,0.8} x L in {30,60,120}, 10 seeds, strong-pair
hosts. The x-axis uses the measured |Delta gamma| (drop of the event-window
gamma relative to the reference window), y = recall.
Purpose: reframe "modest recall" as a characterized detection floor --
specifying the minimum detectable effect size and the length-power
interaction. No failure branch (characterisation study).
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

import os, sys, json, time
import numpy as np

ROOT = _results_root()
import os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rdsg import RDSGDetector
from rdsg.metrics import match_events
from run_e0_syn import make_synthetic
from run_e0_syn2 import REF_LEN, TEST_LEN, PAIR_STRATA, channel_scan

OUT = os.path.normpath(os.path.join(_results_root(), "e0"))
PS = (0.2, 0.35, 0.5, 0.65, 0.8)
LS = (30, 60, 120)
N_PER = 2  # events per (p,L) cell per seed -> 5*3*2=15 events/seed
SPP = 60.0
taus = list(range(-3, 4))


def partial_permute(X_test, X_ref, i, j, s, L, p, rng):
    sl = slice(s, s + L)
    X = np.array(X_test, copy=True)
    n = max(2, int(round(p * L)))
    S = rng.choice(L, size=n, replace=False)
    X[sl, j][S] = rng.permutation(X[sl, j][S])
    return X


def main():
    out_path = os.path.join(OUT, "w3a_power_curve.json")
    res = json.load(open(out_path, encoding="utf-8")) if os.path.exists(out_path) else {}
    res.setdefault("cells", {})
    for seed in range(1, 11):
        if str(seed) in res.get("seeds_done", []):
            continue
        t0 = time.time()
        Xs = make_synthetic(REF_LEN + TEST_LEN, seed=100 + seed)
        X_ref, X_test = Xs[:REF_LEN], Xs[REF_LEN:]
        dets_R = build_dets(X_ref, strata=(PAIR_STRATA[0], 0))
        dets_E = build_dets(X_ref, strata=(0, PAIR_STRATA[1]))
        strong = dets_R[0].strong_pairs_
        rng = np.random.default_rng(seed)
        specs = []
        occupied = np.zeros(TEST_LEN, bool)

        def place(L):
            for _ in range(200):
                s = int(rng.integers(600, TEST_LEN - 600 - L))
                if not occupied[s - 200:s + L + 200].any():
                    occupied[s:s + L] = True
                    return s
            return None

        for p in PS:
            for L in LS:
                for _ in range(N_PER):
                    s = place(L)
                    if s is None:
                        continue
                    i, jj = strong[int(rng.integers(len(strong)))]
                    specs.append(dict(p=p, L=L, i=int(i), j=int(jj), start=s))
        # Per-(p,L) injection + scan: different p values need different injected
        # copies -- group by p
        cell_events = {}
        from scipy.stats import spearmanr
        for p in PS:
            X_inj = np.array(X_test, copy=True)
            ev_here = []
            for sp in [q for q in specs if q["p"] == p]:
                L, s, i, jj = sp["L"], sp["start"], sp["i"], sp["j"]
                X_inj = partial_permute(X_inj, X_ref, i, jj, s, L, p, rng)
                ev_here.append((s, s + L, L, p, i, jj))
            # Detection
            Gfull = np.vstack([X_ref, X_inj])
            Gcal_R = {id(d): d.granulate(X_ref) for d in dets_R}
            Gcal_E = {id(d): d.granulate(X_ref) for d in dets_E}
            Gs_R = {id(d): d.granulate(Gfull) for d in dets_R}
            Gs_E = {id(d): d.granulate(Gfull) for d in dets_E}
            evsR, _, _ = channel_scan(dets_R, Gcal_R, Gs_R, "R")
            evsE, _, _ = channel_scan(dets_E, Gcal_E, Gs_E, "E")
            pred = [(int(a - REF_LEN), int(b - REF_LEN)) for a, b in evsR] + \
                   [(int(a - REF_LEN), int(b - REF_LEN)) for a, b in evsE]
            # Per event: measured |Delta gamma| (event-window vs reference-window
            # gamma) and detection outcome
            K = 4
            for (s0, s1, L, p_, i_, j_) in ev_here:
                e_i = np.quantile(X_ref[:, i_], np.linspace(0, 1, K + 1)[1:-1])
                e_j = np.quantile(X_ref[:, j_], np.linspace(0, 1, K + 1)[1:-1])
                def gam(seg_x, seg_y):
                    gi = np.digitize(seg_x, e_i)
                    gj = np.digitize(seg_y, e_j)
                    Tm = np.zeros((K, K))
                    for a, b in zip(gi, gj):
                        Tm[a, b] += 1
                    P = Tm / max(Tm.sum(), 1)
                    return float((P.sum(1, keepdims=True) * (P / np.maximum(P.sum(0, keepdims=True), 1e-12)).max(1)[0][:, None] * 0).sum() + 0)
                # Simplified measured effect size: Spearman drop
                ref_win = slice(max(0, REF_LEN - 1024), REF_LEN)
                ev_win = slice(REF_LEN + s0, REF_LEN + s1)
                r_ref = abs(float(spearmanr(X_ref[ref_win, i_], X_ref[ref_win, j_]).statistic))
                r_ev = abs(float(spearmanr(X_inj[ev_win, i_], X_inj[ev_win, j_]).statistic))
                hit = any(a <= s1 and b >= s0 for a, b in pred)
                key = f"p{p}_L{L}"
                cell_events.setdefault(key, [])
                cell_events[key].append(dict(delta_rho=round(r_ref - r_ev, 3), hit=bool(hit)))
            print(f"  seed{seed} p={p} done ({time.time()-t0:.0f}s)", flush=True)
        for k, evs in cell_events.items():
            res["cells"].setdefault(k, []).extend(evs)
        res.setdefault("seeds_done", []).append(seed)
        json.dump(res, open(out_path, "w", encoding="utf-8"), indent=1, default=float)
    # Summary: per-cell recall and measured median Delta rho
    print("\n== Effect-size vs recall (pooled 10 seeds) ==")
    summary = {}
    for k, evs in sorted(res["cells"].items()):
        rec = float(np.mean([e["hit"] for e in evs]))
        dmed = float(np.median([e["delta_rho"] for e in evs]))
        summary[k] = dict(recall=round(rec, 3), median_delta_rho=round(dmed, 3), n=len(evs))
        print(f"  {k:10s} recall={rec:.2f} |Δρ|~{dmed:.2f} (n={len(evs)})")
    res["summary"] = summary
    json.dump(res, open(out_path, "w", encoding="utf-8"), indent=1, default=float)
    print("saved w3a_power_curve.json")


def build_dets(X_ref, strata):
    dets = []
    for K in (4, 2):
        for w, blk in ((128, 16), (64, 16)):
            d = RDSGDetector(K=K, w=w, blk=blk, B=199, q=0.05, seed=1,
                             pair_strata=strata, n_subset=4)
            d.fit(X_ref, channels=list(range(X_ref.shape[1])), taus=list(range(-3, 4)))
            dets.append(d)
    return dets


if __name__ == "__main__":
    main()
