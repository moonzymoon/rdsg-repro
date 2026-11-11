# -*- coding: utf-8 -*-
"""S.13 增补：copula 标准化常数在真实网络（METR-LA）上的审计（A4 收尾）。

预承诺网格（运行前定死）：α∈{2,4,8,16,32}×iters=2 五格 + α=8×iters∈{1,4,8} 三格
= 8 格 × 种子 {1,2}。协议与合成门审计（run_e5b）一致：真实宿主（参考段 γ 排名
分位数选取）、n_each=8 注入、单检测器 K=4/w=128/blk=16/B=199、自扫描校准、
monkeypatch 使全管线（图构建+统计+校准）运行在对应 (α, iters) 下。
α=8/iters=2 为主文冻结默认，不参与任何选择；结果无论好坏全部写入
e5c_realnet.json 并进 Supplemental S.13。
"""
import os, sys, json, time
import numpy as np

def _results_root():
    d = os.path.dirname(os.path.abspath(__file__))
    for base in (d, os.path.dirname(d), os.path.dirname(os.path.dirname(d))):
        if os.path.isdir(os.path.join(base, "results")):
            return os.path.join(base, "results")
    return os.path.normpath(os.path.join(d, "..", "..", "results"))

import os as _os
sys.path.insert(0, _os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rdsg.stats as stats_mod
from rdsg import RDSGDetector, inject, sample_specs, event_metrics
from run_e0_syn import rank_pairs_by_ref_gamma
from run_e0_syn2 import channel_scan
from rdsg.datasets import load_metr_la

OUT = os.path.join(_results_root(), "paper", "e5c_realnet.json")
GRID = [(a, 2) for a in (2.0, 4.0, 8.0, 16.0, 32.0)] + [(8.0, i) for i in (1, 4, 8)]
SEEDS = (1, 2)
REF_LEN, TEST_LEN = 12000, 18000
CUR = [8.0, 2]

_orig_ipf = stats_mod.ipf_uniform
def _patched(tab, iters=None, alpha=None):
    return _orig_ipf(tab,
                     iters=iters if iters is not None else CUR[1],
                     alpha=alpha if alpha is not None else CUR[0])
stats_mod.ipf_uniform = _patched


def run_one(X, seed):
    X_ref, X_test = X[:REF_LEN], X[REF_LEN:REF_LEN + TEST_LEN]
    strong, weak, gam = rank_pairs_by_ref_gamma(X_ref, list(range(X.shape[1])), 4, 512)
    vals = np.array(sorted(gam.values()))
    hi, lo = np.quantile(vals, 0.9), np.quantile(vals, 0.1)
    strong = [p for p, v in gam.items() if v >= hi]
    weak = [p for p, v in gam.items() if v <= lo]
    specs = sample_specs(None, strong, weak, n_each=8, t_len=TEST_LEN,
                         margin=600, seed=seed)
    X_inj = inject(X_test, specs, seed=seed)
    true_events = [(sp["start"], sp["start"] + sp["length"], sp["type"]) for sp in specs]
    d = RDSGDetector(K=4, w=128, blk=16, B=199, q=0.05, seed=1)
    d.fit(X_ref, channels=list(range(X.shape[1])), taus=list(range(-3, 4)))
    Gcal = {id(d): d.granulate(X_ref)}
    G = {id(d): d.granulate(np.vstack([X_ref, X_inj]))}
    evs, th, _ = channel_scan([d], Gcal, G, "s", cal_end=REF_LEN // 2,
                              scan_start=REF_LEN, scan_end=REF_LEN + TEST_LEN,
                              gap=8 * 32, spp=300.0)
    pred = [(int(a - REF_LEN), min(int(b - REF_LEN), TEST_LEN), None) for a, b in evs]
    m = event_metrics(true_events, pred, TEST_LEN, 300.0)
    return m["recall"], m["fa_per_day"]


def main():
    X, spp, meta = load_metr_la()
    X = X[:REF_LEN + TEST_LEN]
    rows = {}
    if os.path.exists(OUT):
        rows = json.load(open(OUT, encoding="utf-8")).get("cells", {})
    for a, i in GRID:
        key = f"a{int(a)}_i{i}"
        if key in rows:
            continue
        CUR[0], CUR[1] = a, i
        t0 = time.time()
        rec, fas = [], []
        for sd in SEEDS:
            r, f = run_one(X, sd)
            rec.append(r); fas.append(f)
        rows[key] = [float(np.mean(rec)), float(np.mean(fas))]
        json.dump({"grid": [[a, i] for a, i in GRID], "seeds": list(SEEDS),
                   "protocol": "METR-LA, E5 mirror, real hosts", "cells": rows},
                  open(OUT, "w", encoding="utf-8"), indent=1)
        print(f"alpha={int(a):>2} iters={i}: R={np.mean(rec):.3f} "
              f"fa={np.mean(fas):.2f} ({time.time()-t0:.0f}s)", flush=True)
    r = [v[0] for v in rows.values()]
    print(f"METR-LA audit range: {min(r):.3f}-{max(r):.3f} | default a8_i2 = "
          f"{rows.get('a8_i2')}", flush=True)


if __name__ == "__main__":
    main()
