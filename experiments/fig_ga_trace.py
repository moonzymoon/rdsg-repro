# -*- coding: utf-8 -*-
"""图文摘要底带：案例攻击附近的紧凑 gamma/delta 双轨（S.10 同源冻结口径）。
用于嵌入 graphical_abstract.tex，宽约 60mm，无坐标轴刻度 clutter。"""
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
CASE_EV = 19


def main():
    X, spp, meta = load_gas_pipeline()
    ref0, ref1 = meta["ref"]
    t0, t1 = meta["test"]
    ev = meta["events"][CASE_EV]
    s_abs, e_abs = t0 + ev[0], t0 + ev[1]
    lo = max(ref1, s_abs - 1200)
    g = Granulator(K=K).fit(X[ref0:ref1])
    G = g.transform(X)
    xs, gs, ds = [], [], []
    for a in range(lo, len(X) - W + 1, 64):
        if a > e_abs + 1200:
            break
        tab = block_tables(G, [EDGE], K, BLK, t0=a, t1=a + W)[0].sum(axis=0)
        p = ipf_uniform(tab[None])
        xs.append((a + W / 2 - t0) / spp / 60.0)
        gs.append(float(gamma_soft(p)))
        ds.append(float(grain_direction(p, K)))
    xs, gs, ds = np.array(xs), np.array(gs), np.array(ds)
    a0, a1 = (s_abs - t0) / spp / 60.0, (e_abs - t0) / spp / 60.0

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(2, 1, figsize=(2.5, 1.15), sharex=True,
                             constrained_layout=True)
    for ax, y, c, lab in ((axes[0], gs, "#c0392b", r"$\gamma$ holds"),
                          (axes[1], ds, "#2980b9", r"$\delta$ flips")):
        ax.plot(xs, y, color=c, lw=0.9)
        ax.axvspan(a0, a1, color="#f9c74f", alpha=0.55, lw=0)
        ax.set_ylabel(lab, fontsize=6.5, rotation=0, labelpad=14, va="center")
        ax.set_yticks([])
    axes[1].set_xticks([])
    axes[0].tick_params(left=False, bottom=False)
    axes[1].tick_params(left=False, bottom=False)
    for ax in axes:
        ax.spines["left"].set_visible(False)
    out = os.path.join(_results_root(), "paper", "figs", "ga_trace")
    fig.savefig(out + ".pdf", metadata={"Creator": "", "Producer": "",
                                        "CreationDate": None})
    print("wrote", out + ".pdf | attack band min:", round(a0, 1), round(a1, 1))


if __name__ == "__main__":
    main()
