# -*- coding: utf-8 -*-
"""S.15 规模标度：候选图构建（唯一超线性步）与在线扫描成本 vs 通道数 n（A6 应答）。

设计（预注册）：
- 真实锚点：METR-LA（207 通道）、PEMS-BAY（325）——用真实数据测构图与扫描；
- 合成插值：n∈{500, 1000, 2000} 块相关随机游走（每 ~20 通道一块，块内相关），
  12000 参考点——只测运行时标度，不是检测基准（表注如实声明）；
- 测量：(a) rank_pairs_by_ref_gamma 全对构图时间；(b) RDSGDetector.fit 总时间；
  (c) 固定 116 边配置下单边界族扫描时间（w=128 窗的粒化+建表+四元组+阈值比较）；
  (d) 峰值 RSS。
- 预期（复杂度声明的可检验含义）：(a)(b) 随 n 超线性（≈n²），(c) 与 n 无关
  （E 由通道预算定死），(d) 随 n 线性。

输出：results/paper/scaling.json。
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
from rdsg import RDSGDetector
from run_e0_syn import rank_pairs_by_ref_gamma

OUT = os.path.join(_results_root(), "paper", "scaling.json")
REF_LEN = 12000


def synth_network(n, seed=7, block=20):
    """块相关随机游走：每块内共享一个驱动因子（块内相关≈0.6），块间独立。"""
    rng = np.random.default_rng(seed)
    T = REF_LEN
    n_blocks = int(np.ceil(n / block))
    common = rng.standard_normal((T, n_blocks))
    eps = rng.standard_normal((T, n))
    cols = np.repeat(np.arange(n_blocks), block)[:n]
    return 0.75 * common[:, cols] + 0.65 * eps


def measure(X, label):
    import psutil
    p = psutil.Process(os.getpid())
    n = X.shape[1]
    t0 = time.time()
    strong, weak, gam = rank_pairs_by_ref_gamma(X[:REF_LEN], list(range(n)), 4, 512)
    t_rank = time.time() - t0
    t0 = time.time()
    d = RDSGDetector(K=4, w=128, blk=32, B=99, q=0.05, seed=1,
                     pair_strata=(16, 24), n_subset=6)
    d.fit(X[:REF_LEN], channels=list(range(n)), taus=[0, 1, -1])
    t_fit = time.time() - t0
    rss1 = p.memory_info().rss / 2**20
    # 单边界扫描成本：完整边集 E（含 subset 边）在一个 w=128 窗上的建表+四元组
    from rdsg.tables import block_tables
    from rdsg.stats import four_tuple
    G = d.granulate(X[:REF_LEN])
    E = d.edges_[:116] if len(d.edges_) > 116 else d.edges_
    reps = 5
    t0 = time.time()
    for _ in range(reps):
        tabs = block_tables(G, E, 4, 32, t0=REF_LEN - 128, t1=REF_LEN)
        st = four_tuple(tabs.sum(axis=1), tabs.sum(axis=1), 4)
    t_scan = (time.time() - t0) / reps
    row = dict(n=n, label=label, t_pair_rank_s=round(t_rank, 2),
               t_fit_s=round(t_fit, 2), t_boundary_ms=round(1000 * t_scan, 1),
               rss_peak_mb=round(rss1, 0), E_measured=len(E),
               E_total=len(d.edges_))
    print(json.dumps(row), flush=True)
    return row


def main():
    rows = []
    if os.path.exists(OUT):
        rows = json.load(open(OUT, encoding="utf-8")).get("rows", [])
    done = {r["label"] for r in rows}
    from rdsg.datasets import load_metr_la, load_pems_bay
    real = []
    if "METR-LA(207)" not in done:
        X, _, _ = load_metr_la(); real.append((X, "METR-LA(207)"))
    if "PEMS-BAY(325)" not in done:
        X, _, _ = load_pems_bay(); real.append((X, "PEMS-BAY(325)"))
    for X, lab in real:
        rows.append(measure(X, lab))
        json.dump({"rows": rows}, open(OUT, "w"), indent=1)
    for n in (500, 1000, 2000):
        lab = f"synth({n})"
        if lab in done:
            continue
        X = synth_network(n)
        rows.append(measure(X, lab))
        json.dump({"rows": rows}, open(OUT, "w"), indent=1)
    print("--- summary ---")
    for r in rows:
        print(f"{r['label']:>14}: rank={r['t_pair_rank_s']:>8.1f}s fit={r['t_fit_s']:>7.1f}s "
              f"boundary={r['t_boundary_ms']}ms rss={r['rss_peak_mb']:.0f}MB")


if __name__ == "__main__":
    main()
