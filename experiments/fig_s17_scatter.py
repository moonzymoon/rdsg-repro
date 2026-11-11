# -*- coding: utf-8 -*-
"""S.17 图：气路案例边上的窗口级 gamma-delta 平面视图。

每个 w=128 滚动窗（步长 64）画为 (gamma, delta) 一点：
灰=参考段/正常窗，红=与任一标注攻击重叠的窗，黄=案例攻击(#19)窗。
机制预期（Prop 1(i) 的可视化）：倒相攻击窗在 gamma 近似不变的水平带上
横跨 delta 符号翻转——正是 gamma-only 监控盲区的几何形态。
复用 fig_s10_gas_case 的冻结口径（K=4, blk=32, ipf alpha=8/2 迭代）。
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rdsg.datasets import load_gas_pipeline
from rdsg.granulation import Granulator
from rdsg.tables import block_tables
from rdsg.stats import ipf_uniform, gamma_soft, grain_direction


def _results_root():
    d = os.path.dirname(os.path.abspath(__file__))
    for base in (d, os.path.dirname(d), os.path.dirname(os.path.dirname(d))):
        if os.path.isdir(os.path.join(base, "results")):
            return os.path.join(base, "results")
    return os.path.normpath(os.path.join(d, "..", "..", "results"))


K, BLK, W = 4, 32, 128
EDGE = ((0,), 1, -1)


def main():
    X, spp, meta = load_gas_pipeline()
    ref0, ref1 = meta["ref"]
    t0, t1 = meta["test"]
    events = meta["events"]
    g = Granulator(K=K).fit(X[ref0:ref1])
    G = g.transform(X)
    xs, gs, ds, tag = [], [], [], []
    for a in range(ref1, len(X) - W + 1, 64):
        tab = block_tables(G, [EDGE], K, BLK, t0=a, t1=a + W)[0].sum(axis=0)
        p = ipf_uniform(tab[None])
        gv, dv = float(gamma_soft(p)), float(grain_direction(p, K))
        c = a + W / 2
        inside = any(t0 + e[0] - 64 < c < t0 + e[1] + 64 for e in events)
        case = t0 + events[19][0] - 64 < c < t0 + events[19][1] + 64
        xs.append((gv, dv)); gs.append(gv); ds.append(dv)
        tag.append(2 if case else (1 if inside else 0))
    gs, ds, tag = np.array(gs), np.array(ds), np.array(tag)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(4.4, 3.2), constrained_layout=True)
    m0 = tag == 0
    ax.scatter(gs[m0], ds[m0], s=8, c="0.55", alpha=0.55, lw=0,
               label="normal windows", zorder=2)
    m1 = (tag == 1) & ~m0
    ax.scatter(gs[m1], ds[m1], s=16, c="#c0392b", alpha=0.8, lw=0,
               label="attack windows", zorder=3)
    m2 = tag == 2
    ax.scatter(gs[m2], ds[m2], s=34, c="#f9c74f", edgecolors="#8a5a00",
               lw=0.7, label="case attack", zorder=4)
    ax.axhline(0, color="0.3", lw=0.6, ls=":")
    ax.set_xlabel(r"$\gamma$ (dependency strength)")
    ax.set_ylabel(r"$\delta$ (direction)")
    ax.set_xlim(0.2, 0.9)
    ax.legend(fontsize=7.5, loc="upper right", frameon=False)
    out = os.path.join(_results_root(), "paper", "figs", "s17_gd_plane")
    fig.savefig(out + ".pdf", metadata={"Creator": "", "Producer": "",
                                        "CreationDate": None})
    fig.savefig(out + ".png", dpi=200)
    # 机制量化：攻击窗 vs 正常窗的 gamma/delta 分布对比
    print(f"normal: gamma [{gs[m0].min():.2f},{gs[m0].max():.2f}] "
          f"delta [{ds[m0].min():.2f},{ds[m0].max():.2f}] n={m0.sum()}")
    print(f"attack: gamma [{gs[m1|m2].min():.2f},{gs[m1|m2].max():.2f}] "
          f"delta [{ds[m1|m2].min():.2f},{ds[m1|m2].max():.2f}] n={(m1|m2).sum()}")
    print("case:", [(round(g,2), round(d,2)) for g, d, t in zip(gs, ds, tag) if t == 2])
    print("wrote", out + ".pdf/.png")


if __name__ == "__main__":
    main()
