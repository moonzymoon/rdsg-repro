# -*- coding: utf-8 -*-
"""补强A: MI 替换 γ 做边选择消融（三网×10种子，检测族=γ+δ 不变，仅换选边准则）。"""
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import os

def _results_root():
    """Results directory: nearest ancestor shipping a results/ folder
    (local: <project>/results; reviewer package: <pkg>/results)."""
    d = os.path.dirname(os.path.abspath(__file__))
    for base in (d, os.path.dirname(d), os.path.dirname(os.path.dirname(d))):
        if os.path.isdir(os.path.join(base, "results")):
            return os.path.join(base, "results")
    return os.path.normpath(os.path.join(d, "..", "..", "results"))

OUT = os.path.normpath(os.path.join(_results_root(), "paper"))
os.makedirs(OUT, exist_ok=True)

from rdsg import RDSGDetector
from rdsg.datasets import load_metr_la, load_pems_bay, load_beijing_air
from rdsg.metrics import match_events, event_metrics
from rdsg import inject as do_inject
LOADERS = {"metr_la": load_metr_la, "pems_bay": load_pems_bay, "beijing_air": load_beijing_air}
taus = (0, 1, -1)


def mi_score(x_i, x_j):
    """列联表互信息（等频 K=4 粒化后）。"""
    K = 4
    qs_i = np.quantile(x_i, np.linspace(0, 1, K + 1)[1:-1])
    qs_j = np.quantile(x_j, np.linspace(0, 1, K + 1)[1:-1])
    gi = np.searchsorted(qs_i, x_i, side="right")
    gj = np.searchsorted(qs_j, x_j, side="right")
    T = np.zeros((K, K))
    for a, b in zip(gi, gj):
        T[a, b] += 1
    P = T / max(T.sum(), 1)
    pi = P.sum(axis=1)
    pj = P.sum(axis=0)
    mi = 0.0
    for a in range(K):
        for b in range(K):
            if P[a, b] > 0:
                mi += P[a, b] * np.log(P[a, b] / max(pi[a] * pj[b], 1e-12))
    return mi


def select_edges_mi(X_ref, n_strong=16, n_weak=24):
    """MI 替换 γ：按列联表 MI 排名通道对。"""
    n = X_ref.shape[1]
    scores = []
    for i in range(n):
        for j in range(i + 1, n):
            s = mi_score(X_ref[:, i], X_ref[:, j])
            scores.append((s, i, j))
    scores.sort(reverse=True)
    strong = [(i, j) for _, i, j in scores[:n_strong]]
    weak = [(i, j) for _, i, j in scores[-n_weak:]]
    return strong, weak


def main():
    out_path = os.path.join(OUT, "abl_mi_select.json")
    res = json.load(open(out_path, encoding="utf-8")) if os.path.exists(out_path) else {}
    for cfg_key, loader in LOADERS.items():
        X, spp, meta = loader()
        X_ref, X_test = X[:12000], X[12000:12000 + 18000]
        n_test = len(X_test)  # Beijing provides only 5520 rows
        res.setdefault(cfg_key, {})
        if len(res[cfg_key]) >= 10:
            print(f"[{cfg_key}] done")
            continue
        # MI 选边
        strong_mi, weak_mi = select_edges_mi(X_ref[-2048:], 16, 24)
        # γ 选边（原版）
        det = RDSGDetector(K=4, w=128, blk=32, B=499, q=0.05, seed=1,
                           components=("dgamma", "ddelta"),
                           pair_strata=(16, 24), n_subset=4)
        det.fit(X_ref, channels=list(range(X_ref.shape[1])), taus=list(taus))
        strong_gamma = det.strong_pairs_[:16] if hasattr(det, 'strong_pairs_') else strong_mi
        weak_gamma = det.weak_pairs_[:24] if hasattr(det, 'weak_pairs_') else weak_mi
        # 为公平，两者用同一种边数
        for seed in range(1, 11):
            if str(seed) in res[cfg_key]:
                continue
            t0 = time.time()
            rng = np.random.default_rng(seed)
            specs = []
            occupied = np.zeros(n_test, bool)
            all_pairs = strong_mi[:8] + weak_mi[:8]
            for typ, pairs, n_ev in (("inv", strong_mi[:8], 10),
                                     ("brk", strong_mi[8:16], 10),
                                     ("emb", weak_mi[:8], 10)):
                for _ in range(n_ev):
                    L = int(rng.choice([30, 60, 120]))
                    s = int(rng.integers(600, n_test - 600 - L))
                    if occupied[s - 200:s + L + 200].any():
                        continue
                    occupied[s:s + L] = True
                    pi, pj = pairs[int(rng.integers(len(pairs)))]
                    specs.append(dict(type=typ, i=pi, j=pj, start=s, length=L))
            X_inj = do_inject(X_test, specs, seed=seed)
            true_events = [(sp["start"], sp["start"] + sp["length"], sp["type"]) for sp in specs]
            true3 = [(a, b, None) for a, b, _ in true_events]

            # MI-selected detector: the MI ranking now drives the DETECTOR's
            # candidate edges via force_pairs (not only the host selection) --
            # this is the single factor under test
            from run_e0_syn2 import channel_scan
            d_mi = RDSGDetector(K=4, w=128, blk=32, B=499, q=0.05, seed=1,
                                components=("dgamma", "ddelta"),
                                pair_strata=(16, 24), n_subset=4)
            d_mi.fit(X_ref, channels=list(range(X_ref.shape[1])), taus=list(taus),
                     force_pairs=(strong_mi[:16], weak_mi[:24]))
            G_ref = d_mi.granulate(X_ref)
            d_mi.calibrate(G_ref, 12000 // 2, cal_t0=0)
            G_full = d_mi.granulate(np.vstack([X_ref, X_inj]))
            evs_mi, _, _ = channel_scan([d_mi], {id(d_mi): G_ref}, {id(d_mi): G_full}, "R",
                                        cal_end=len(X_ref), scan_start=len(X_ref),
                                        scan_end=len(X_ref) + len(X_test),
                                        selfscan_start=0, gap=4 * 32, spp=spp)
            pred_mi = [(int(a - len(X_ref)), int(b - len(X_ref))) for a, b in evs_mi]
            m_mi = event_metrics(true_events, pred_mi, n_test, spp)

            # matched comparator: IDENTICAL detector and identical injection
            # hosts, with the gamma-ranked edges forced instead
            d_g = RDSGDetector(K=4, w=128, blk=32, B=499, q=0.05, seed=1,
                               components=("dgamma", "ddelta"),
                               pair_strata=(16, 24), n_subset=4)
            d_g.fit(X_ref, channels=list(range(X_ref.shape[1])), taus=list(taus),
                    force_pairs=(strong_gamma[:16], weak_gamma[:24]))
            G_ref_g = d_g.granulate(X_ref)
            d_g.calibrate(G_ref_g, 12000 // 2, cal_t0=0)
            G_full_g = d_g.granulate(np.vstack([X_ref, X_inj]))
            evs_g, _, _ = channel_scan([d_g], {id(d_g): G_ref_g}, {id(d_g): G_full_g}, "R",
                                       cal_end=len(X_ref), scan_start=len(X_ref),
                                       scan_end=len(X_ref) + len(X_test),
                                       selfscan_start=0, gap=4 * 32, spp=spp)
            pred_g = [(int(a - len(X_ref)), int(b - len(X_ref))) for a, b in evs_g]
            m_g = event_metrics(true_events, pred_g, n_test, spp)

            res[cfg_key][str(seed)] = dict(recall=m_mi["recall"], fa_per_day=m_mi["fa_per_day"],
                                           gamma_sel_recall=m_g["recall"],
                                           gamma_sel_fa_per_day=m_g["fa_per_day"])
            print(f"[{cfg_key} s{seed}] MI-sel R={m_mi['recall']:.2f} gamma-sel R={m_g['recall']:.2f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            json.dump(res, open(out_path, "w"), indent=1)

    # 汇总
    print("\n=== MI 替换消融汇总 ===")
    for cfg_key in LOADERS:
        if cfg_key not in res:
            continue
        recalls = [v["recall"] for v in res[cfg_key].values()]
        print(f"  {cfg_key:12s} MI-select: recall={np.mean(recalls):.3f}±{np.std(recalls):.3f}")
    json.dump(res, open(out_path, "w"), indent=1)
    print("saved", out_path)


if __name__ == "__main__":
    import time
    main()
