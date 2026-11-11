# -*- coding: utf-8 -*-
"""Phase 2A: CUSUM baseline + Phase 2B: distance-correlation baseline (3 networks, same FA-matching protocol)."""
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
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from rdsg.datasets import load_metr_la, load_pems_bay, load_beijing_air
from rdsg.metrics import match_events, event_metrics
from rdsg import inject as do_inject
from run_paper_e2 import build_channel, make_specs, PAIR_STRATA_E, LEN_BY_SPP

OUT = os.path.join(_results_root(), "paper")
os.makedirs(OUT, exist_ok=True)
LOADERS = {"metr_la": load_metr_la, "pems_bay": load_pems_bay, "beijing_air": load_beijing_air}
taus = (0, 1, -1)


def cusum_scores(X_ref, X_test, edges, threshold_h=0.5, drift=0.25):
    """Two-sided CUSUM per edge; returns the max positive|negative CUSUM statistic series.
    edges: [(i, j)] channel pairs. For each edge, CUSUM of |corr(i, j)|.
    Simplification: per-edge change detection via CUSUM over sliding-window Pearson correlation."""
    from scipy.stats import spearmanr
    n_edges = len(edges)
    T = len(X_test)
    W = 128
    blk = 32
    n_blocks = T // blk
    # per-block, per-edge |Spearman|
    block_corr = np.zeros((n_blocks, n_edges))
    for bi in range(n_blocks):
        seg = X_test[bi * blk:(bi + 1) * blk]
        for ei, (i, j) in enumerate(edges):
            if i < seg.shape[1] and j < seg.shape[1]:
                block_corr[bi, ei] = abs(spearmanr(seg[:, i], seg[:, j]).statistic or 0)
    # reference-segment baseline mean
    ref_corr = np.zeros(n_edges)
    ref_blocks = len(X_ref) // blk
    for bi in range(ref_blocks):
        seg = X_ref[bi * blk:(bi + 1) * blk]
        for ei, (i, j) in enumerate(edges):
            if i < seg.shape[1] and j < seg.shape[1]:
                ref_corr[ei] += abs(spearmanr(seg[:, i], seg[:, j]).statistic or 0)
    ref_corr /= max(ref_blocks, 1)
    # CUSUM
    scores = np.zeros(T)
    cusum = np.zeros(n_edges)
    for t in range(T):
        bi = min(t // blk, n_blocks - 1)
        dev = block_corr[bi] - ref_corr  # positive = correlation deviating from baseline
        cusum = np.maximum(0, cusum + np.abs(dev) - 0.05)
        scores[t] = cusum.max()
    return scores


def dcor_scores(X_ref, X_test, edges, W=128, blk=32):
    """Distance-correlation baseline: sliding-window delta-dcor per edge pair."""
    from scipy.spatial.distance import cdist
    T = len(X_test)
    scores = np.zeros(T)
    n_feat = min(X_test.shape[1], 12)
    Xr = X_ref[:, :n_feat]
    # reference dcor matrix
    D_ref = cdist(Xr.T, Xr.T, metric='euclidean')
    dc_ref = _dcor_matrix(Xr)
    for start in range(0, T - W, blk):
        seg = X_test[start:start + W]
        dc_cur = _dcor_matrix(seg[:, :n_feat])
        diff = np.abs(dc_cur - dc_ref).max()
        scores[start:start + W] = diff
    return scores


def _dcor_matrix(X):
    """Simplified distance-correlation matrix (n_feat x n_feat)."""
    from scipy.spatial.distance import cdist
    n = X.shape[0]
    dmat = cdist(X.T, X.T, metric='euclidean')
    # center
    A = dmat - dmat.mean(axis=0) - dmat.mean(axis=1, keepdims=True) + dmat.mean()
    B = A.copy()
    # dCov_ij = sqrt(mean(A_ij * A_ji))  simplified: pairwise across columns
    dc = np.zeros((X.shape[1], X.shape[1]))
    for i in range(X.shape[1]):
        for j in range(X.shape[1]):
            if i == j:
                dc[i, j] = 1.0
            else:
                di = cdist(X[:, i:i+1], X[:, i:i+1])
                dj = cdist(X[:, j:j+1], X[:, j:j+1])
                Ai = di - di.mean() - di.mean(axis=0) + di.mean()
                Bj = dj - dj.mean() - dj.mean(axis=0) + dj.mean()
                dcov = np.sqrt(max(np.mean(Ai * Bj), 0))
                dvar_i = np.sqrt(max(np.mean(Ai * Ai), 0))
                dvar_j = np.sqrt(max(np.mean(Bj * Bj), 0))
                dc[i, j] = dcov / max(dvar_i * dvar_j, 1e-12)
    return dc


def main():
    out_path = os.path.join(OUT, "bl_cusum_dcor.json")
    res = json.load(open(out_path, encoding="utf-8")) if os.path.exists(out_path) else {}
    for cfg_key, loader in LOADERS.items():
        X, spp, meta = loader()
        X_ref, X_test = X[:12000], X[12000:12000+18000]
        lengths = LEN_BY_SPP[spp]
        planner = build_channel(X_ref, taus, strata=(16, 24))[0]
        force_R, force_E = planner.strong_pairs_, planner.weak_pairs_
        dE0 = build_channel(X_ref, taus, strata=PAIR_STRATA_E, n_subset=6,
                            force_pairs=(force_R, force_E))[0]
        res.setdefault(cfg_key, {})
        edges_cusum = [(i, j) for i in range(min(12, X_ref.shape[1])) for j in range(i+1, min(12, X_ref.shape[1]))][:24]

        for seed in range(1, 11):
            if str(seed) in res[cfg_key] and "cusum" in res[cfg_key][str(seed)] and "dcor" in res[cfg_key][str(seed)]:
                continue
            t0 = time.time()
            np.random.seed(seed * 1000 + seed)
            rng = np.random.default_rng(seed)
            specs = []
            occupied = np.zeros(18000, bool)
            strong = [(i, j) for i in range(min(12, X_ref.shape[1])) for j in range(i+1, min(12, X_ref.shape[1]))][:16]
            weak = [(i, j) for i in range(min(12, X_ref.shape[1])) for j in range(i+1, min(12, X_ref.shape[1]))][16:40]
            for typ, pairs, L_range, n_ev in (("inv", strong, (30, 120), 10),
                                               ("brk", strong, (30, 120), 10),
                                               ("emb", weak, (30, 120), 10),
                                               ("emb2", weak, (30, 120), 5)):
                for _ in range(n_ev):
                    L = int(rng.choice(L_range))
                    s = int(rng.integers(600, 18000 - 600 - L))
                    if occupied[s-200:s+L+200].any():
                        continue
                    occupied[s:s+L] = True
                    pi, pj = pairs[int(rng.integers(len(pairs)))]
                    if typ == "emb2":
                        specs.append(dict(type=typ, i=pi, j=pj, i2=(pi, (pi+3) % X_ref.shape[1]), start=s, length=L))
                    else:
                        specs.append(dict(type=typ, i=pi, j=pj, start=s, length=L))
            X_inj = do_inject(X_test, specs, seed=seed)
            true_events = [(sp["start"], sp["start"] + sp["length"], sp["type"]) for sp in specs]
            true3 = [(a, b, None) for a, b, _ in true_events]
            seed_res = {}

            # --- CUSUM ---
            from run_paper_e2 import pick_theta_split
            cusum_scores = cusum_scores_fn(X_ref, X_inj, edges_cusum)
            th = pick_theta_split(cusum_scores[:12000], 0.25, spp, 64, 64)
            evs = merge_evs(cusum_scores > th, 128)
            pred = evs
            m = event_metrics(true_events, pred, 18000, spp)
            seed_res["cusum"] = dict(recall=m["recall"], fa_per_day=m["fa_per_day"])

            # --- dcor ---
            dcor_scores = dcor_scores_fn(X_ref, X_inj, n_feat=min(12, X_ref.shape[1]))
            th2 = pick_theta_split(dcor_scores[:12000], 0.25, spp, 64, 64)
            evs2 = merge_evs(dcor_scores > th2, 128)
            m2 = event_metrics(true_events, evs2, 18000, spp)
            seed_res["dcor"] = dict(recall=m2["recall"], fa_per_day=m2["fa_per_day"])

            res[cfg_key][str(seed)] = seed_res
            print(f"[{cfg_key} s{seed}] cusum={seed_res['cusum']['recall']:.2f} dcor={seed_res['dcor']['recall']:.2f} ({time.time()-t0:.0f}s)", flush=True)
            json.dump(res, open(out_path, "w"), indent=1)

    # summary
    print("\n=== baseline summary ===")
    for cfg_key in LOADERS:
        if cfg_key not in res:
            continue
        for m in ("cusum", "dcor"):
            recalls = [v[m]["recall"] for v in res[cfg_key].values() if m in v]
            if recalls:
                print(f"  {cfg_key:12s} {m:6s}: recall={np.mean(recalls):.3f}")
    json.dump(res, open(out_path, "w"), indent=1)
    print("saved")


def merge_evs(flags, gap):
    idx = np.where(flags)[0]
    if len(idx) == 0:
        return []
    evs, t0, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev > gap:
            evs.append((t0, prev + 1))
            t0 = i
        prev = i
    evs.append((t0, prev + 1))
    return [(int(a), int(b)) for a, b in evs]


def cusum_scores_fn(X_ref, X_test, edges, blk=64):
    """Multi-edge CUSUM score series."""
    ref_corr = []
    for i, j in edges:
        if i < X_ref.shape[1] and j < X_ref.shape[1]:
            from scipy.stats import spearmanr
            ref_corr.append(abs(spearmanr(X_ref[:, i], X_ref[:, j]).statistic or 0))
        else:
            ref_corr.append(0)
    ref_corr = np.array(ref_corr)
    T = len(X_test)
    scores = np.zeros(T)
    cusum = np.zeros(len(edges))
    for t in range(T):
        if t < 128:
            scores[t] = 0
            continue
        seg = X_test[max(0, t-128):t]
        for ei, (i, j) in enumerate(edges):
            if i < seg.shape[1] and j < seg.shape[1]:
                from scipy.stats import spearmanr
                cur = abs(spearmanr(seg[:, i], seg[:, j]).statistic or 0)
                dev = abs(cur - ref_corr[ei])
                cusum[ei] = max(0, cusum[ei] + dev - 0.05)
        scores[t] = cusum.max()
    return scores


def dcor_scores_fn(X_ref, X_test, n_feat=12, blk=64):
    """Multi-edge distance-correlation change score."""
    from scipy.spatial.distance import cdist
    ref_dc = _pair_dcor(X_ref[:, :n_feat])
    T = len(X_test)
    scores = np.zeros(T)
    for start in range(0, T, blk):
        end = min(start + blk, T)
        seg = X_test[start:end]
        cur_dc = _pair_dcor(seg[:, :n_feat])
        diff = np.abs(cur_dc - ref_dc[:cur_dc.shape[0], :cur_dc.shape[0]]).max() if cur_dc.shape == ref_dc[:cur_dc.shape[0], :cur_dc.shape[0]].shape else 0
        scores[start:end] = diff
    return scores


def _pair_dcor(X):
    from scipy.spatial.distance import cdist
    n = X.shape[0]
    dc = np.eye(X.shape[1])
    for i in range(X.shape[1]):
        for j in range(i+1, X.shape[1]):
            di = cdist(X[:, i:i+1], X[:, i:i+1])
            dj = cdist(X[:, j:j+1], X[:, j:j+1])
            Ai = di - di.mean() - di.mean(axis=0) + di.mean()
            Bj = dj - dj.mean() - dj.mean(axis=0) + dj.mean()
            dcov = np.sqrt(max(np.mean(Ai * Bj), 0))
            dvi = np.sqrt(max(np.mean(Ai * Ai), 0))
            dvj = np.sqrt(max(np.mean(Bj * Bj), 0))
            dc[i, j] = dc[j, i] = dcov / max(dvi * dvj, 1e-12)
    return dc


if __name__ == "__main__":
    main()
