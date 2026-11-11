# -*- coding: utf-8 -*-
"""Phase 1A: delta-only ablation (3 networks x 10 seeds, detection family = ddelta single component)."""
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
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from rdsg import RDSGDetector
import numpy as np
import os
from rdsg import inject as do_inject
from rdsg.datasets import load_metr_la, load_pems_bay, load_beijing_air
from run_paper_e2 import make_specs as _make_specs, build_channel as _build_ch, PAIR_STRATA_E as _PSE, LEN_BY_SPP as _LSP

OUT = os.path.join(_results_root(), "paper")
CONFIGS = {
    "metr_la": load_metr_la,
    "beijing_air": load_beijing_air,
    "pems_bay": load_pems_bay,
}
taus = (0, 1, -1)

def main():
    res = {}
    out_path = os.path.join(OUT, "abl_delta.json")
    if os.path.exists(out_path):
        res = json.load(open(out_path, encoding="utf-8"))
    for cfg_key, loader in CONFIGS.items():
        if cfg_key in res and len(res[cfg_key]) >= 10:
            print(f"[{cfg_key}] already done ({len(res[cfg_key])} seeds)")
            continue
        X, spp, meta = loader()
        X_ref, X_test = X[:12000], X[12000:12000+18000]
        res.setdefault(cfg_key, {})
        d = RDSGDetector(K=4, w=128, blk=32, B=499, q=0.05, seed=1,
                         components=("ddelta",),
                         pair_strata=(16, 24), n_subset=4)
        for seed in range(1, 11):
            if str(seed) in res[cfg_key]:
                continue
            t0 = time.time()
            pass  # imports moved to top
            _planner = _build_ch(X_ref, taus, strata=(16, 24))[0]
            _fR, _fE = _planner.strong_pairs_, _planner.weak_pairs_
            _dE0 = _build_ch(X_ref, taus, strata=_PSE, n_subset=6, force_pairs=(_fR, _fE))[0]
            specs = _make_specs(_planner, _dE0, X_ref, X_test, seed, _LSP[spp])
            X_inj = do_inject(X_test, specs, seed=seed)
            true_events = [(sp["start"], sp["start"] + sp["length"], sp["type"]) for sp in specs]
            # delta-only detection
            from rdsg.metrics import match_events, event_metrics
            d2 = RDSGDetector(K=4, w=128, blk=32, B=499, q=0.05, seed=1,
                              components=("ddelta",),
                              pair_strata=(16, 24), n_subset=4)
            d2.fit(X_ref, channels=list(range(X_ref.shape[1])), taus=list(taus))
            G_ref = d2.granulate(X_ref)
            d2.calibrate(G_ref, 12000 // 2, cal_t0=0)
            G_full = d2.granulate(np.vstack([X_ref, X_inj]))
            from run_e0_syn2 import channel_scan
            evs, thetas, _ = channel_scan([d2], {id(d2): G_ref}, {id(d2): G_full}, "R",
                                          cal_end=len(X_ref), scan_start=len(X_ref),
                                          scan_end=len(X_full := np.vstack([X_ref, X_inj])),
                                          selfscan_start=0, gap=4*32, spp=spp)
            pred = [(int(a - len(X_ref)), int(b - len(X_ref))) for a, b in evs]
            matches, delays, falses = match_events(true_events, [(a, b, None) for a, b in pred])
            recall = sum(1 for _, p in matches if p is not None) / max(len(true_events), 1)
            days = 18000 * spp / 86400.0
            fa = len(falses) / max(days, 1e-9)
            res[cfg_key][str(seed)] = dict(recall=recall, fa_per_day=fa, n_pred=len(pred))
            print(f"[{cfg_key} s{seed}] δ-only R={recall:.2f} fa={fa:.2f} ({time.time()-t0:.0f}s)", flush=True)
            json.dump(res, open(out_path, "w"), indent=1)
    # summary
    print("\n=== delta-only ablation summary ===")
    for cfg_key in CONFIGS:
        if cfg_key not in res:
            continue
        recalls = [v["recall"] for v in res[cfg_key].values()]
        fas = [v["fa_per_day"] for v in res[cfg_key].values()]
        print(f"  {cfg_key:12s} δ-only: recall={np.mean(recalls):.3f}±{np.std(recalls):.3f} fa={np.mean(fas):.2f}")
    json.dump(res, open(out_path, "w"), indent=1)
    print("saved", out_path)

if __name__ == "__main__":
    main()
