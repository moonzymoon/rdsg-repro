# -*- coding: utf-8 -*-
"""E2 operating-curve runner: per-method test-score sweeps -> realised-FA-matched recall.

For each dataset and seed, every method's reference and test score streams are
computed once, then a threshold grid is swept to trace (realised FA/day, recall)
points. The manuscript reports recall at a common *realised* budget (0.25/day),
closing the requested-vs-realised gap. Output: results/paper/e2_oc.json.
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

import os, sys, json, time, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.normpath(os.path.join(HERE, ".."))            # code/ (rdsg, baselines)
PROJECT = os.path.normpath(os.path.join(HERE, "..", ".."))   # project root (results/)
sys.path.insert(0, CODE)
sys.path.insert(0, HERE)
from rdsg import inject
from rdsg.datasets import load_metr_la, load_pems_bay, load_beijing_air
from rdsg.metrics import event_metrics
from run_paper_e2 import (LEN_BY_SPP, SEEDS, FA_TARGET, build_channel, make_specs,
                          lengths_gap, PAIR_STRATA_R, PAIR_STRATA_E,
                          point_baseline_scores)
from run_e0_syn2 import channel_scan, _unit_stack, _merge_alarms
from baselines.relational import (WindowDivergenceBaseline, VARResidualBaseline,
                                  GrangerBaseline, MIGraphBaseline, merge_events)
from baselines.gdn import train_gdn, gdn_scores

OUT = os.path.join(_results_root(), "paper", "e2_oc.json")
MULTS = (0.4, 0.55, 0.7, 0.85, 1.0, 1.25, 1.6, 2.2, 3.0)   # ours: theta multiplier
QGRID = np.linspace(0.90, 0.99995, 45)                      # baselines: ref-score quantiles
FA_TARGETS = (0.25, 0.5)
METHODS = ("rdsg", "b5b_pcorr", "b5c_var", "b1_corr", "b2_granger", "b5a_mi",
           "b4_if", "b4_lof", "b3_gdn")


def matched_recall(points, fa_target):
    ok = [r for f, r in points if f <= fa_target + 1e-9]
    return max(ok) if ok else None


def run_dataset(key, loader):
    X, spp, meta = loader()
    X_ref, X_test = X[:12000], X[12000:12000 + 18000]
    lengths = LEN_BY_SPP[spp]
    n_ref = len(X_ref)
    taus = (0, 1, -1)
    t_start = time.time()
    planner = build_channel(X_ref, taus, strata=(16, 24))[0]
    force_R, force_E = planner.strong_pairs_, planner.weak_pairs_
    dR0 = planner
    dE0 = build_channel(X_ref, taus, strata=PAIR_STRATA_E, n_subset=6,
                        force_pairs=(force_R, force_E))[0]
    # GDN: train once, calibrate scores on the held-out reference tail
    tr_end = int(n_ref * 0.8)
    dev = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    model, mu, sd, dv = train_gdn(X_ref[:tr_end], device=dev)
    gdn_sc_ref = gdn_scores(model, X_ref[tr_end:], mu, sd, dv)
    gap_ours = max(lengths_gap(spp), 256)
    out = {m: {} for m in METHODS}
    for seed in SEEDS:
        specs = make_specs(dR0, dE0, X_ref, X_test, seed, lengths)
        X_test_inj = inject(X_test, specs, seed=seed)
        true_events = [(sp["start"], sp["start"] + sp["length"], sp["type"]) for sp in specs]
        # ---- ours (family): calibrate once per channel, sweep a common theta multiplier ----
        comp_series = []
        for strata, n_sub, fp in ((PAIR_STRATA_R, None, (force_R, force_E)),
                                  (PAIR_STRATA_E, 6, (force_R, force_E))):
            dd = build_channel(X_ref, taus, strata=strata, n_subset=n_sub, force_pairs=fp)
            X_full = np.vstack([X_ref, X_test_inj])
            Gcal = {id(d): d.granulate(X_ref) for d in dd}
            Gs = {id(d): d.granulate(X_full) for d in dd}
            comps = sorted(set(dd[0].components).intersection(*[set(d.components) for d in dd]))
            evs, thetas, _ = channel_scan(dd, Gcal, Gs, "scan", cal_end=n_ref // 2,
                                          scan_start=n_ref, scan_end=len(X_full),
                                          gap=gap_ours, spp=spp)
            grid_q, Zu = _unit_stack(dd, Gs, n_ref, len(X_full))
            Tc = {c: Zu[:, :, ci].max(axis=1) for ci, c in enumerate(comps)}
            comp_series.append((np.asarray(grid_q), Tc, thetas))
        # per-channel merge then union of event lists (matches the production
        # pipeline, where R and E alarms are merged separately before union)
        for m in MULTS:
            pr = []
            for grid_q, Tc, thetas in comp_series:
                alarm = np.zeros(len(grid_q), bool)
                for c, s in Tc.items():
                    alarm |= s > m * thetas[c]
                for a, b in _merge_alarms(np.asarray(grid_q), alarm, gap=gap_ours):
                    pr.append((int(a - n_ref), int(b - n_ref), None))
            mm = event_metrics(true_events, pr, len(X_test), spp)
            out["rdsg"].setdefault(str(seed), []).append([mm["fa_per_day"], mm["recall"]])
        print(f"[{key} s{seed}] rdsg sweep done {time.time()-t_start:.0f}s", flush=True)
        # ---- relational baselines ----
        rel = (("b5b_pcorr", WindowDivergenceBaseline("pcorr", spp=spp, fa_target=FA_TARGET)),
               ("b5c_var", VARResidualBaseline(spp=spp, fa_target=FA_TARGET)),
               ("b1_corr", WindowDivergenceBaseline("corr", spp=spp, fa_target=FA_TARGET)),
               ("b2_granger", GrangerBaseline(dR0.strong_pairs_[:12], spp=spp, fa_target=FA_TARGET)),
               ("b5a_mi", MIGraphBaseline(spp=spp, fa_target=FA_TARGET)))
        for name, bl in rel:
            t_r, s_r, t_q, s_q, gap = bl.run_scores(X_ref, X_test_inj)
            for q in QGRID:
                th = np.quantile(s_r, q)
                evs = merge_events(np.asarray(t_q), np.asarray(s_q) > th, gap)
                pr = [(int(a - n_ref), int(b - n_ref), None) for a, b in evs]
                mm = event_metrics(true_events, pr, len(X_test), spp)
                out[name].setdefault(str(seed), []).append([mm["fa_per_day"], mm["recall"]])
        print(f"[{key} s{seed}] relational sweeps done {time.time()-t_start:.0f}s", flush=True)
        # ---- point baselines + GDN ----
        for name, mdl in (("b4_if", "if"), ("b4_lof", "lof")):
            ref_agg, agg = point_baseline_scores(X_ref, X_test_inj, mdl, seed=seed)
            for q in QGRID:
                th = np.quantile(ref_agg, q)
                evs = merge_events(np.arange(len(agg)), agg > th, 128)
                pr = [(int(a), int(b), None) for a, b in evs]
                mm = event_metrics(true_events, pr, len(X_test), spp)
                out[name].setdefault(str(seed), []).append([mm["fa_per_day"], mm["recall"]])
        sc_q = gdn_scores(model, X_test_inj, mu, sd, dv)
        for q in QGRID:
            th = np.quantile(gdn_sc_ref, q)
            evs = merge_events(np.arange(len(sc_q)), sc_q > th, 128)
            pr = [(int(a), int(b), None) for a, b in evs]
            mm = event_metrics(true_events, pr, len(X_test), spp)
            out["b3_gdn"].setdefault(str(seed), []).append([mm["fa_per_day"], mm["recall"]])
        print(f"[{key} s{seed}] point+GDN sweeps done {time.time()-t_start:.0f}s", flush=True)
    summary = {}
    for m in METHODS:
        for fa_t in FA_TARGETS:
            vals = [matched_recall([tuple(p) for p in pts], fa_t)
                    for pts in out[m].values() if pts]
            vals = [v for v in vals if v is not None]
            if vals:
                summary.setdefault(m, {})[f"matched_{fa_t}"] = round(float(np.mean(vals)), 3)
    return {"curves": out, "matched": summary}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all")
    args = ap.parse_args()
    loaders = {"metr_la": load_metr_la, "pems_bay": load_pems_bay,
               "beijing_air": load_beijing_air}
    res = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
    for key, loader in loaders.items():
        if args.dataset != "all" and key != args.dataset:
            continue
        print(f"===== {key} start =====", flush=True)
        res[key] = run_dataset(key, loader)
        json.dump(res, open(OUT, "w"), indent=1)
        print(f"[{key}] matched@0.25: "
              f"{ {m: v.get('matched_0.25') for m, v in res[key]['matched'].items()} }",
              flush=True)
    print("saved", OUT)


if __name__ == "__main__":
    main()
