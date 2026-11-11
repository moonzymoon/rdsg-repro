# -*- coding: utf-8 -*-
"""E2f 随机宿主审计（循环性扩评）：注入宿主取自候选图**之外**但有参考依赖的对。

预注册口径（运行前固定）：
- 数据：METR-LA（ref 12000 / test 18000，与 E2 完全同切片同协议）。
- 宿主池：非候选边中 |Spearman|>=0.5 的对 → INV/BRK 宿主；
  非候选边中 |Spearman|<0.3 的对 → EMB 宿主。每种子 INV/BRK/EMB 各 10 事件，
  长度池与 E2 相同（30/60/120）。EMB2 不在此审计（其定义绑定子集边结构）。
- 对照方法：RDSG（固定候选图）+ 全对关系基线 corr/pcorr/MI + 点聚合 IF/LOF，
  全部走同一参考段自扫描 FA 匹配协议（0.25/day）。
- 解读规则（两种结果都如实写）：若 RDSG 在非候选宿主上崩塌而全对基线保持，
  如实报告覆盖边界（监控图=方法假设集）；若全对基线也崩，则循环性批评被削弱。

输出：results/paper/e2f_ranhost_metr_la.json
"""
import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, CODE)
sys.path.insert(0, HERE)

from scipy.stats import spearmanr
from rdsg import inject, event_metrics
from rdsg.datasets import load_metr_la
from rdsg.injection import sample_specs
from run_paper_e2 import (build_channel, ours_pipeline, point_baseline,
                          recall_by_type, FA_TARGET, PAIR_STRATA_E)
from baselines.relational import WindowDivergenceBaseline, MIGraphBaseline


def _results_root():
    d = os.path.dirname(os.path.abspath(__file__))
    for base in (d, os.path.dirname(d), os.path.dirname(os.path.dirname(d))):
        if os.path.isdir(os.path.join(base, "results")):
            return os.path.join(base, "results")
    return os.path.normpath(os.path.join(d, "..", "..", "results"))


OUT = os.path.join(_results_root(), "paper", "e2f_ranhost_metr_la.json")
SEEDS = range(1, 11)
N_EACH = 10
STRONG_TH, WEAK_TH = 0.5, 0.3
LEN = (30, 60, 120)


def main():
    X, spp, meta = load_metr_la()
    X_ref, X_test = X[:12000], X[12000:12000 + 18000]
    taus = (0, 1, -1)
    planner = build_channel(X_ref, taus, strata=(16, 24))[0]
    force_R, force_E = planner.strong_pairs_, planner.weak_pairs_
    dE0 = build_channel(X_ref, taus, strata=PAIR_STRATA_E, n_subset=6,
                        force_pairs=(force_R, force_E))[0]
    # 候选边全集（保守：凡出现在任何通道边表中的无序对都算候选）
    cand = set()
    for o in (planner, dE0):
        for (parents, j, _lag) in o.edges_:
            for p in (parents if isinstance(parents, (list, tuple)) else (parents,)):
                cand.add((min(int(p), int(j)), max(int(p), int(j))))
        for pair in getattr(o, "candidate_pairs_", []):
            cand.add((min(pair), max(pair)))
    # 参考尾段 Spearman
    tail = X_ref[-2048:]
    n = X.shape[1]
    rho = {}
    for i in range(n):
        for j in range(i + 1, n):
            rho[(i, j)] = float(spearmanr(tail[:, i], tail[:, j]).statistic)
    strong_pool = [p for p, r in rho.items() if abs(r) >= STRONG_TH and p not in cand]
    weak_pool = [p for p, r in rho.items() if abs(r) < WEAK_TH and p not in cand]
    print(f"候选边 {len(cand)} | 非候选强对池 {len(strong_pool)} | 非候选弱对池 {len(weak_pool)}",
          flush=True)
    assert len(strong_pool) >= 2 * N_EACH and len(weak_pool) >= N_EACH

    res = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
    res.setdefault("protocol", {}).__setitem__(
        "host_pools", dict(n_candidate=len(cand), strong_noncand=len(strong_pool),
                           weak_noncand=len(weak_pool), strong_th=STRONG_TH, weak_th=WEAK_TH))
    for seed in SEEDS:
        sd = str(seed)
        if sd in res.get("seeds", {}):
            continue
        rng = np.random.default_rng(seed * 31 + 7)
        sp_str = [strong_pool[k] for k in rng.permutation(len(strong_pool))]
        sp_wak = [weak_pool[k] for k in rng.permutation(len(weak_pool))]
        specs = sample_specs(None, sp_str, sp_wak, n_each=N_EACH,
                             t_len=len(X_test), margin=480, lengths=LEN, seed=seed)
        for sp in specs:  # INV 的倒相目标与 E2 同规则
            if sp["type"] == "inv":
                r = rho[(min(sp["i"], sp["j"]), max(sp["i"], sp["j"]))]
                sp["rho_target"] = float(np.clip(-r, -1, -0.6)) if r >= 0 else float(np.clip(-r, 0.6, 1))
        for sp in specs:  # 审计完整性：宿主必须全部在候选图外
            assert (min(sp["i"], sp["j"]), max(sp["i"], sp["j"])) not in cand
        X_test_inj = inject(X_test, specs, seed=seed)
        true_events = [(sp["start"], sp["start"] + sp["length"], sp["type"]) for sp in specs]
        sr = {}
        t0 = time.time()
        pred, rt = ours_pipeline(X_ref, X_test_inj, X_test, taus, spp=spp,
                                 force_R=force_R, force_E=force_E)
        m = event_metrics(true_events, pred, len(X_test), spp)
        m["recall_by_type"] = recall_by_type(true_events, pred)
        m["runtime"] = rt
        sr["rdsg"] = m
        print(f"[s{seed}] rdsg R={m['recall']:.3f} ({m['recall_by_type']}) fa={m['fa_per_day']:.2f} {rt:.0f}s",
              flush=True)
        pr, _ = WindowDivergenceBaseline("corr", spp=spp, fa_target=FA_TARGET).run(X_ref, X_test_inj)
        mm = event_metrics(true_events, [(e[0], e[1], None) for e in pr], len(X_test), spp)
        mm["recall_by_type"] = recall_by_type(true_events, [(e[0], e[1], None) for e in pr])
        sr["b1_corr"] = mm
        pr, _ = WindowDivergenceBaseline("pcorr", spp=spp, fa_target=FA_TARGET).run(X_ref, X_test_inj)
        mm = event_metrics(true_events, [(e[0], e[1], None) for e in pr], len(X_test), spp)
        mm["recall_by_type"] = recall_by_type(true_events, [(e[0], e[1], None) for e in pr])
        sr["b5b_pcorr"] = mm
        pr, _ = MIGraphBaseline(spp=spp, fa_target=FA_TARGET).run(X_ref, X_test_inj)
        mm = event_metrics(true_events, [(e[0], e[1], None) for e in pr], len(X_test), spp)
        mm["recall_by_type"] = recall_by_type(true_events, [(e[0], e[1], None) for e in pr])
        sr["b5a_mi"] = mm
        for name, mdl in (("b4_if", "if"), ("b4_lof", "lof")):
            pr, _ = point_baseline(X_ref, X_test_inj, mdl, seed=seed, spp=spp)
            mm = event_metrics(true_events, [(e[0], e[1], None) for e in pr], len(X_test), spp)
            mm["recall_by_type"] = recall_by_type(true_events, [(e[0], e[1], None) for e in pr])
            sr[name] = mm
        res.setdefault("seeds", {})[sd] = sr
        json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1, default=float)
        print(f"[s{seed}] 完成 ({time.time()-t0:.0f}s)", flush=True)
    print("done ->", OUT)


if __name__ == "__main__":
    main()
