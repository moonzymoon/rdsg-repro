# -*- coding: utf-8 -*-
"""Typing diagnosis: replay 1 seed each on METR-LA and Beijing, dump the peak-edge
four-tuple values vs typing thresholds for detected events, tally the three-branch
outcomes, and locate the root cause of typing failures."""
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

ROOT = _results_root()
import os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_paper_e2 as R
from rdsg import inject, sample_specs, event_metrics
from rdsg.tables import edge_code
from rdsg.stats import four_tuple
from rdsg.detector import _code_table
from rdsg.datasets import load_metr_la, load_beijing_air

REF_LEN = 12000


def run(ds_loader, name, seed=1, spp=300.0):
    print(f"\n======== {name} seed={seed} ========")
    X, sppv, meta = ds_loader()
    X_ref, X_test = X[:REF_LEN], X[REF_LEN:REF_LEN + 18000]
    lengths = (30, 60, 120) if sppv == 300.0 else (12, 24, 48)
    taus = [0, 1, -1]
    dR = R.build_channel(X_ref, taus, strata=R.PAIR_STRATA_R)
    dE = R.build_channel(X_ref, taus, strata=R.PAIR_STRATA_E, n_subset=6)
    strong = dR[0].strong_pairs_
    weak = dE[0].weak_pairs_
    specs = R.make_specs(dR[0], dE[0], X_ref, X_test, seed, lengths)
    from scipy.stats import spearmanr as _sr
    for sp in specs:
        if sp["type"] == "inv":
            r = float(_sr(X_ref[-2048:, sp["i"]], X_ref[-2048:, sp["j"]]).statistic)
            sp["rho_target"] = float(np.clip(-r, -1, -0.6)) if r >= 0 else float(np.clip(-r, 0.6, 1))
    X_inj = inject(X_test, specs, seed=seed)
    true_events = [(sp["start"], sp["start"] + sp["length"], sp["type"]) for sp in specs]
    X_full = np.vstack([X_ref, X_inj])

    evs_R, thR, _ = R.scan_channel(dR, X_ref, X_full, spp)
    evs_E, thE, _ = R.scan_channel(dE, X_ref, X_full, spp)
    print(f"θ_R={ {k: round(v,2) for k,v in thR.items()} } θ_E={ {k: round(v,2) for k,v in thE.items()} }")
    print(f"anchors: R hi={dR[0].gamma_hi_:.3f} lo={dR[0].gamma_lo_:.3f} | "
          f"E hi={dE[0].gamma_hi_:.3f} lo={dE[0].gamma_lo_:.3f}")
    cg, crho, cd, cjs = dR[0]._typing_thresholds(
        dR[0].granulate(np.vstack([X_ref, X_inj])))
    print(f"typing thresholds: cγ={cg:.3f} cρ={crho:.3f} cδ={cd:.3f} cJS={cjs:.3f}")

    # Dump typing details for each detected event
    n_hit = {t: [0, 0] for t in ("inv", "brk", "emb", "emb2")}
    for ch, dd, evs in (("R", dR, evs_R), ("E", dE, evs_E)):
        d0 = dd[0]
        G = d0.granulate(X_full)
        d0.z_series(G, REF_LEN, len(X_full))
        for (a, b) in evs:
            hit_t = next((t for (s, e, t) in true_events
                          if max(a - REF_LEN, s) < min(b - REF_LEN, e)), None)
            if hit_t is None:
                continue
            ev = dict(t0=a, t1=b, scale=d0.w,
                      members=[dict(t=a, zmax=1.0, sig_idx=None, n_sig=0)])
            ev = d0.localize(G, [ev])[0]
            idx = d0._top_edges(ev, k=10)
            ev = d0.type_events(G, [ev])[0]
            # dump peak edge
            if idx:
                ei = idx[0]
                A, dch, tau = d0.edges_[ei]
                code = edge_code(G, A, dch, tau, d0.K)
                KA = d0.K ** len(A)
                Lw = d0.w // 2
                c0, c1 = b, b + Lw
                r0 = c0 - Lw
                Tc = _code_table(code, np.arange(c0, min(c1, len(G))), KA, d0.K)
                Tr = _code_table(code, np.arange(max(0, r0), c0), KA, d0.K)
                st = four_tuple(Tr[None], Tc[None], d0.K)
                print(f"  [{ch}] true={hit_t} t={a-REF_LEN} L={b-a} peak_edge={A}->{dch},τ{tau}: "
                      f"γref={st['gamma_ref'][0]:.2f} Δγ={st['dgamma'][0]:+.2f} "
                      f"ρ={st['rho'][0]:.2f} Δδ={st['ddelta'][0]:+.2f} JS={st['djs'][0]:.2f} "
                      f"→ typed={ev.get('type','?')}")
            else:
                print(f"  [{ch}] true={hit_t} t={a-REF_LEN} L={b-a}: no significant edges (top empty)")
    print(f"Hit type tally: {n_hit}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("both", "metr"):
        run(load_metr_la, "METR-LA", seed=1)
    if which in ("both", "beijing"):
        run(load_beijing_air, "Beijing", seed=1, spp=3600.0)
