# -*- coding: utf-8 -*-
"""S.13 稳健性审计：copula 标准化超参 (alpha, IPF 迭代数) 扫描（合成门，E5 协议）。

预承诺网格（运行前定死，不按结果挑点）：
  alpha ∈ {2, 4, 8, 16, 32} × iters ∈ {1, 2, 4, 8}，全交叉 20 格；
  种子 {201, 202}（与 E5 K/w 扫描同种子、同协议：n_each=8、B=199、单检测器
  K=4、w=128、blk=16、自扫描校准 gap=8*32、FA 目标随 channel_scan 默认）。
主实验冻结值 (alpha=8, iters=2) 不参与任何选择，本表仅报告稳健性；
结果无论好坏全部写入 e5b_alpha_ipf.json 并进 Supplemental S.13。

单因子注入方式：monkeypatch rdsg.stats.ipf_uniform 的默认参——four_tuple 与
std_gamma（图构建排序）均以无参调用进入该函数，因此参考图、边统计、自扫描
校准在每格内整体一致地运行在对应 (alpha, iters) 下。
"""
import os, sys, json, time
import numpy as np

def _results_root():
    d = os.path.dirname(os.path.abspath(__file__))
    for base in (d, os.path.dirname(d), os.path.dirname(os.path.dirname(d))):
        if os.path.isdir(os.path.join(base, "results")):
            return os.path.join(base, "results")
    return os.path.normpath(os.path.join(d, "..", "..", "results"))

ROOT = _results_root()
import os as _os
sys.path.insert(0, _os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rdsg.stats as stats_mod
from rdsg import RDSGDetector, inject, sample_specs, event_metrics
from run_e0_syn import make_synthetic, rank_pairs_by_ref_gamma
from run_e0_syn2 import build_configs, channel_scan, REF_LEN, TEST_LEN

OUT = os.path.join(_results_root(), "paper", "e5b_alpha_ipf.json")
ALPHAS = (2.0, 4.0, 8.0, 16.0, 32.0)
ITERS = (1, 2, 4, 8)
SEEDS = (201, 202)
CUR = [8.0, 2]                       # 当前格的 (alpha, iters)，patch 读取

_orig_ipf = stats_mod.ipf_uniform
def _patched(tab, iters=None, alpha=None):
    return _orig_ipf(tab,
                     iters=iters if iters is not None else CUR[1],
                     alpha=alpha if alpha is not None else CUR[0])
stats_mod.ipf_uniform = _patched


def run_one(Xs, seed):
    X_ref, X_test = Xs[:REF_LEN], Xs[REF_LEN:]
    strong, weak, gam = rank_pairs_by_ref_gamma(X_ref, list(range(Xs.shape[1])), 4, 512)
    vals = np.array(sorted(gam.values()))
    hi, lo = np.quantile(vals, 0.9), np.quantile(vals, 0.1)
    strong = [p for p, v in gam.items() if v >= hi]
    weak = [p for p, v in gam.items() if v <= lo]
    specs = sample_specs(None, strong, weak, n_each=8, t_len=TEST_LEN, margin=600, seed=seed)
    X_inj = inject(X_test, specs, seed=seed)
    true_events = [(sp["start"], sp["start"] + sp["length"], sp["type"]) for sp in specs]
    d = RDSGDetector(K=4, w=128, blk=16, B=199, q=0.05, seed=1)
    d.fit(X_ref, channels=list(range(Xs.shape[1])), taus=list(range(-3, 4)))
    Gcal = {id(d): d.granulate(X_ref)}
    G = {id(d): d.granulate(np.vstack([X_ref, X_inj]))}
    evs, th, _ = channel_scan([d], Gcal, G, "s", cal_end=REF_LEN // 2,
                              scan_start=REF_LEN, scan_end=REF_LEN + TEST_LEN,
                              gap=8 * 32, spp=60.0)
    pred = [(int(a - REF_LEN), min(int(b - REF_LEN), TEST_LEN), None) for a, b in evs]
    m = event_metrics(true_events, pred, TEST_LEN, 60.0)
    return m["recall"], m["fa_per_day"]


def emit_table():
    """把网格写成 paper/gen/s13_table.tex（Supplemental S.13 直接 \input）。"""
    d = json.load(open(OUT, encoding="utf-8"))
    cells = d["cells"]
    rows = []
    for a in ALPHAS:
        vals = [cells[f"a{int(a)}_i{i}"] for i in ITERS]
        rows.append(f"{int(a)} & " + " & ".join(f"{v[0]:.2f}" for v in vals) + " \\\\")
    gen = os.path.join(_results_root(), "..", "paper", "gen", "s13_table.tex")
    with open(gen, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(rows) + "\n\\bottomrule\n")
    print("wrote", gen)


def main():
    rows = {f"a{int(a)}_i{i}": [] for a in ALPHAS for i in ITERS}
    if os.path.exists(OUT):
        old = json.load(open(OUT, encoding="utf-8"))
        rows.update({k: v for k, v in old.items() if isinstance(v, list)})
    for a in ALPHAS:
        for i in ITERS:
            key = f"a{int(a)}_i{i}"
            if len(rows[key]) >= len(SEEDS):
                continue
            CUR[0], CUR[1] = a, i
            t0 = time.time()
            rec, fas = [], []
            for sd in SEEDS:
                Xs = make_synthetic(REF_LEN + TEST_LEN, seed=sd)
                r, f = run_one(Xs, sd)
                rec.append(r); fas.append(f)
            rows[key] = [float(np.mean(rec)), float(np.mean(fas))]
            json.dump({"grid": {"alphas": list(ALPHAS), "iters": list(ITERS),
                                "seeds": list(SEEDS), "protocol": "E5 mirror"},
                       "cells": rows},
                      open(OUT, "w", encoding="utf-8"), indent=1)
            print(f"alpha={int(a):>2} iters={i}: R={np.mean(rec):.3f} "
                  f"fa={np.mean(fas):.2f}  ({time.time()-t0:.0f}s)", flush=True)
    emit_table()


if __name__ == "__main__":
    main()
