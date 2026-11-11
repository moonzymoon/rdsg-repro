# -*- coding: utf-8 -*-
"""论文主实验 E2（机制专属注入套件）+ E3（消融）+ E6（判型混淆）。

数据：METR-LA / PEMS-BAY / Beijing-Air（换血数据集，SWaT 系零进入）。
注入：INV/BRK/EMB 各10事件 + EMB2 5事件 × 5 种子（宿主取自检测器自身候选图）。
方法：本文（全四元组）vs γ-only vs B1 相关矩阵 / B2 Granger / B4 IF·LOF /
      B5a MI / B5b 偏相关 / B5c VAR残差 / B3 GDN（torch 复现）。
协议：全部方法走同一"参考段自扫描 FA 匹配"阈值协议（FA_TARGET/天），事件级指标。
E3：分量消融（γ / γ+δ / γ+JS / 全）仅在 METR-LA。
"""
import os, sys, json, time, argparse
import numpy as np

import os

def _results_root():
    """Results directory: nearest ancestor shipping a results/ folder.

    Local repo: code/experiments/x.py -> <project>/results; reviewer
    package: <pkg>/experiments/x.py -> <pkg>/results (shipped frozen
    JSONs), so reruns merge with them instead of writing outside."""
    d = os.path.dirname(os.path.abspath(__file__))
    for base in (d, os.path.dirname(d), os.path.dirname(os.path.dirname(d))):
        if os.path.isdir(os.path.join(base, "results")):
            return os.path.join(base, "results")
    return os.path.normpath(os.path.join(d, "..", "..", "results"))

ROOT = _results_root()
import os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rdsg import RDSGDetector, inject, sample_specs, event_metrics
from rdsg.metrics import match_events
from rdsg.datasets import load_metr_la, load_pems_bay, load_beijing_air
from baselines.relational import (WindowDivergenceBaseline, VARResidualBaseline,
                                  GrangerBaseline, MIGraphBaseline, merge_events,
                                  pick_theta_split)
from baselines.point_agg import _fit_score
from baselines.gdn import train_gdn, gdn_scores
OUT = os.path.normpath(os.path.join(_results_root(), "paper"))
FA_TARGET = 0.25
KS, WS = (4, 2), (128, 64, 32)
BLK, B = 64, 499
N_EACH, N_EMB2 = 10, 5
SEEDS = range(1, 11)
TYPE_MAP = {"inversion": "inv", "breakdown": "brk", "emergence": "emb", "emergence2": "emb2"}
LEN_BY_SPP = {300.0: (30, 60, 120), 3600.0: (12, 24, 48)}
PAIR_STRATA_R = (16, 0)
PAIR_STRATA_E = (0, 24)


def _to_short(t):
    return TYPE_MAP.get(t, t)


def recall_by_type(true_events, pred_events):
    out = {}
    for typ in ("inv", "brk", "emb", "emb2"):
        te = [e for e in true_events if e[2] == typ]
        if not te:
            continue
        matches, _, _ = match_events(te, [(p[0], p[1]) for p in pred_events])
        out[typ] = sum(1 for _, p in matches if p is not None) / len(te)
    return out


def detail_with_type(true_events, pred_events):
    # v2 修复：直接用完整三元组匹配（match_events 只按重叠区间配对，多余元素无损），
    # 预测类型得以保留——v1 在此剥成二元组导致预测类型全为 None（252/252）
    matches, _, _ = match_events(true_events, pred_events)
    return [[int(t0), int(t1 - t0), typ,
             (pred[2] if pred is not None and len(pred) > 2 else None),
             bool(pred is not None)]
            for (t0, t1, typ), pred in matches]


def build_channel(X_ref, taus, comps=None, strata=(16, 0), n_subset=0, w_list=WS,
                  force_pairs=None):
    dets = []
    for K in KS:
        for w in w_list:
            c = comps or ("dgamma", "ddelta", "djs")
            if K == 2:
                c = tuple(x for x in c if x != "djs")
            d = RDSGDetector(K=K, w=w, blk=min(BLK, w // 2), B=B, q=0.05, seed=1,
                             components=c, pair_strata=strata, n_subset=n_subset)
            d.fit(X_ref, channels=list(range(X_ref.shape[1])), taus=taus,
                  force_pairs=force_pairs)
            d.channel_kind_ = "R" if strata[0] > 0 else "E"
            dets.append(d)
    return dets


def make_specs(dR0, dE0, X_ref, X_test, seed, lengths):
    strong = dR0.strong_pairs_
    weak = dE0.weak_pairs_
    specs = sample_specs(None, strong, weak, n_each=N_EACH,
                         lengths=lengths, t_len=len(X_test), margin=max(lengths) * 4, seed=seed)
    from scipy.stats import spearmanr as _sr
    for sp in specs:
        if sp["type"] == "inv":
            r = float(_sr(X_ref[-2048:, sp["i"]], X_ref[-2048:, sp["j"]]).statistic)
            sp["rho_target"] = float(np.clip(-r, -1, -0.6)) if r >= 0 else float(np.clip(-r, 0.6, 1))
    # EMB-2：从 E 通道子集 parent 图取宿主
    rng = np.random.default_rng(seed * 17 + 3)
    sp_map = dE0.subset_parents_
    keys = sorted(sp_map.keys())
    for k in range(N_EMB2):
        j = keys[int(rng.integers(len(keys)))]
        ps = sp_map[j]
        if len(ps) < 2:
            continue
        i1, i2 = ps[0], ps[len(ps) // 2]
        L = int(rng.choice(lengths))
        s = int(rng.integers(max(lengths) * 2, len(X_test) - max(lengths) * 2 - L))
        specs.append(dict(type="emb2", i=i1, j=j, i2=(i1, i2), start=s, length=L))
    return specs


def scan_channel(dd, X_ref, X_full, spp):
    n_ref = len(X_ref)
    Gcal = {id(d): d.granulate(X_ref) for d in dd}
    Gs = {id(d): d.granulate(X_full) for d in dd}
    from run_e0_syn2 import channel_scan  # 复用分量族实现
    evs, thetas, (tgrid, Tser) = channel_scan(dd, Gcal, Gs, "scan",
                                  cal_end=n_ref // 2, scan_start=n_ref, scan_end=len(X_full),
                                  gap=max(lengths_gap(spp), 8 * 32), spp=spp)
    return evs, thetas, (tgrid, Tser)


def lengths_gap(spp):
    return {300.0: 512, 3600.0: 96}[spp] if spp in (300.0, 3600.0) else 512


def ours_pipeline(X_ref, X_test_inj, X_test, taus, comps=None, spp=300.0,
                  force_R=None, force_E=None):
    global _spp
    _spp = [spp]
    t0 = time.time()
    dR = build_channel(X_ref, taus, comps=comps, strata=PAIR_STRATA_R,
                       force_pairs=(force_R, force_E))
    dE = build_channel(X_ref, taus, comps=comps, strata=PAIR_STRATA_E, n_subset=6,
                       force_pairs=(force_R, force_E))
    X_full = np.vstack([X_ref, X_test_inj])
    evs_R, thR, (tqR, TqR) = scan_channel(dR, X_ref, X_full, _spp[0])
    evs_E, thE, (tqE, TqE) = scan_channel(dE, X_ref, X_full, _spp[0])
    tq_map = {"R": (np.asarray(tqR), np.asarray(TqR)), "E": (np.asarray(tqE), np.asarray(TqE))}
    evs = [(a, b, "R") for a, b in evs_R] + [(a, b, "E") for a, b in evs_E]
    # 定位+判型：事件落在哪个通道就用该通道的主配置（K=4 首个窗）
    typed = []
    n_ref = len(X_ref)
    for (t0e, t1e, ch) in evs:
        dd = dR if ch == "R" else dE
        d = dd[0]
        G = d.granulate(X_full)
        d.z_series(G, n_ref, len(X_full))       # 恢复缓存块表
        # v5.2：锚点=事件内峰值边界（滚动均值证据 T 最大处）
        tq, Tq = tq_map[ch]
        m = (tq >= t0e) & (tq < t1e)
        peak = int(tq[m][int(np.argmax(Tq[m]))]) if m.any() else int(t0e)
        ev = dict(t0=t0e, t1=t1e, scale=d.w,
                  members=[dict(t=peak, zmax=1.0, sig_idx=None, n_sig=0)])
        ev = d.localize(G, [ev])[0]
        ev = d.type_events(G, [ev], anchors=[peak])[0]
        ev["channel"] = ch
        typed.append(ev)
    pred = [(e["t0"] - n_ref, min(e["t1"] - n_ref, len(X_test)), _to_short(e.get("type", "?")))
            for e in typed]
    return pred, time.time() - t0


def point_baseline(X_ref, X_test_inj, model, fa_target=FA_TARGET, seed=0, spp=300.0):
    t0 = time.time()
    n = X_ref.shape[1]
    half = len(X_ref) // 2
    ref_agg = np.full(len(X_ref), -np.inf)
    for j in range(n):
        ref_agg[:half] = np.maximum(ref_agg[:half],
                                    _fit_score(model, X_ref[half:, j:j+1], X_ref[:half, j:j+1], seed))
        ref_agg[half:] = np.maximum(ref_agg[half:],
                                    _fit_score(model, X_ref[:half, j:j+1], X_ref[half:, j:j+1], seed))
    # rate-based calibration: smallest theta whose reference alarm-sample rate
    # meets the budget (same rule as the RDSG family and all other baselines)
    k = int(np.floor(fa_target * len(ref_agg) * spp / 86400.0))
    srt = np.sort(ref_agg)
    th = float(srt[len(srt) - k - 1]) if k < len(srt) else float(srt[0])
    agg = np.full(len(X_test_inj), -np.inf)
    for j in range(n):
        agg = np.maximum(agg, _fit_score(model, X_ref[:, j:j+1], X_test_inj[:, j:j+1], seed))
    evs = merge_events(np.arange(len(agg)), agg > th, 128)
    return [(int(a), int(b), None) for a, b in evs], time.time() - t0


def point_baseline_scores(X_ref, X_test_inj, model, seed=0):
    """Reference and test point-aggregation score arrays (no thresholding)."""
    n = X_ref.shape[1]
    half = len(X_ref) // 2
    ref_agg = np.full(len(X_ref), -np.inf)
    for j in range(n):
        ref_agg[:half] = np.maximum(ref_agg[:half],
                                    _fit_score(model, X_ref[half:, j:j+1], X_ref[:half, j:j+1], seed))
        ref_agg[half:] = np.maximum(ref_agg[half:],
                                    _fit_score(model, X_ref[:half, j:j+1], X_ref[half:, j:j+1], seed))
    agg = np.full(len(X_test_inj), -np.inf)
    for j in range(n):
        agg = np.maximum(agg, _fit_score(model, X_ref[:, j:j+1], X_test_inj[:, j:j+1], seed))
    return ref_agg, agg


def run_dataset(key, loader, taus=(0, 1, -1)):
    X, spp, meta = loader()
    X_ref, X_test = X[:12000], X[12000:12000 + 18000]
    lengths = LEN_BY_SPP[spp]
    print(f"== {meta['name']} data {X.shape} spp={spp} ref={len(X_ref)} test={len(X_test)}", flush=True)
    # 图权威配置
    planner = build_channel(X_ref, taus, strata=(16, 24))[0]  # 主粒度规划候选图
    force_R, force_E = planner.strong_pairs_, planner.weak_pairs_
    dR0 = planner
    dE0 = build_channel(X_ref, taus, strata=PAIR_STRATA_E, n_subset=6,
                        force_pairs=(force_R, force_E))[0]
    g_spp = spp
    gdn_cache = None
    res = {}
    out_path = os.path.join(OUT, f"e2_{key}.json")
    if os.path.exists(out_path):
        res = json.load(open(out_path, encoding="utf-8"))
    METHODS = ["rdsg", "abl_gamma", "abl_gd", "abl_gj", "b4_if", "b4_lof", "b5a_mi",
               "b5b_pcorr", "b5c_var", "b1_corr", "b2_granger", "b3_gdn"]
    for seed in SEEDS:
        done = all(str(seed) in res.get(m, {}) and "recall" in res.get(m, {}).get(str(seed), {})
                   for m in METHODS if m in res)
        need = [m for m in METHODS if str(seed) not in res.get(m, {})]
        if not need:
            continue
        specs = make_specs(dR0, dE0, X_ref, X_test, seed, lengths)
        X_test_inj = inject(X_test, specs, seed=seed)
        true_events = [(sp["start"], sp["start"] + sp["length"], sp["type"]) for sp in specs]
        seed_res = {}
        # ---- 本文 / 消融 ----
        t0 = time.time()
        pred, rt = ours_pipeline(X_ref, X_test_inj, X_test, taus, spp=spp,
                                 force_R=force_R, force_E=force_E)
        m = event_metrics(true_events, pred, len(X_test), spp)
        m["recall_by_type"] = recall_by_type(true_events, pred)
        m["detail"] = detail_with_type(true_events, pred)
        m["runtime"] = rt
        seed_res["rdsg"] = m
        print(f"[{meta['name']} s{seed}] rdsg R={m['recall']:.2f} ({m['recall_by_type']}) "
              f"fa={m['fa_per_day']:.2f} type={m['type_acc']:.2f} {rt:.0f}s", flush=True)
        # 消融（全种子）
        for name, comps in (("abl_gamma", ("dgamma",)),
                            ("abl_gd", ("dgamma", "ddelta")),
                            ("abl_gj", ("dgamma", "djs"))):
            pr, rt2 = ours_pipeline(X_ref, X_test_inj, X_test, taus, comps=comps, spp=spp,
                                    force_R=force_R, force_E=force_E)
            mm = event_metrics(true_events, pr, len(X_test), spp)
            mm["recall_by_type"] = recall_by_type(true_events, pr)
            mm["runtime"] = rt2
            seed_res[name] = mm
        # ---- 基线 ----
        for name, model in (("b4_if", "if"), ("b4_lof", "lof")):
            pr, rt2 = point_baseline(X_ref, X_test_inj, model, seed=seed, spp=spp)
            seed_res[name] = dict(recall=None, runtime=rt2, events=[[a, b] for a, b, _ in pr])
        pr, _ = MIGraphBaseline(spp=spp, fa_target=FA_TARGET).run(X_ref, X_test_inj)
        seed_res["b5a_mi"] = dict(events=[[a, b] for a, b in pr])
        pr, _ = WindowDivergenceBaseline("pcorr", spp=spp, fa_target=FA_TARGET).run(X_ref, X_test_inj)
        seed_res["b5b_pcorr"] = dict(events=[[a, b] for a, b in pr])
        pr, _ = VARResidualBaseline(spp=spp, fa_target=FA_TARGET).run(X_ref, X_test_inj)
        seed_res["b5c_var"] = dict(events=[[a, b] for a, b in pr])
        pr, _ = WindowDivergenceBaseline("corr", spp=spp, fa_target=FA_TARGET).run(X_ref, X_test_inj)
        seed_res["b1_corr"] = dict(events=[[a, b] for a, b in pr])
        pr, _ = GrangerBaseline(dR0.strong_pairs_[:12], spp=spp, fa_target=FA_TARGET).run(X_ref, X_test_inj)
        seed_res["b2_granger"] = dict(events=[[a, b] for a, b in pr])
        # GDN：每数据集训练一次（留出校准），各 seed 评分
        if gdn_cache is None:
            nref = len(X_ref)
            tr_end = int(nref * 0.8)
            gdn_cache = (train_gdn(X_ref[:tr_end], device="cuda" if __import__("torch").cuda.is_available() else "cpu"),
                         X_ref[tr_end:], tr_end)
        (model, mu, sd, dev), Xcal, tr_end = gdn_cache
        sc_ref = gdn_scores(model, Xcal, mu, sd, dev)
        th = pick_theta_split(sc_ref, FA_TARGET, spp, 128, 128)
        sc_q = gdn_scores(model, X_test_inj, mu, sd, dev)
        evs = merge_events(np.arange(len(sc_q)), sc_q > th, 128)
        # sc_q is scored on the test segment: indices are already test-local
        seed_res["b3_gdn"] = dict(events=[[int(a), int(b)] for a, b in evs],
                                  theta=float(th))
        # 统一评估
        for k, v in seed_res.items():
            if v.get("recall") is not None:
                continue
            pr = [(a, b, None) for a, b in v.get("events", [])]
            mm = event_metrics(true_events, pr, len(X_test), spp)
            mm["recall_by_type"] = recall_by_type(true_events, pr)
            v.update(mm)
        res.setdefault("rdsg", {})[str(seed)] = seed_res["rdsg"]
        for k in ("abl_gamma", "abl_gd", "abl_gj", "b4_if", "b4_lof", "b5a_mi",
                  "b5b_pcorr", "b5c_var", "b1_corr", "b2_granger", "b3_gdn"):
            res.setdefault(k, {})[str(seed)] = seed_res[k]
        json.dump(res, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="metr_la",
                        choices=["metr_la", "pems_bay", "beijing_air"])
    args = parser.parse_args()
    os.makedirs(OUT, exist_ok=True)
    loaders = {"metr_la": load_metr_la, "pems_bay": load_pems_bay,
               "beijing_air": load_beijing_air}
    run_dataset(args.dataset, loaders[args.dataset])


if __name__ == "__main__":
    main()
