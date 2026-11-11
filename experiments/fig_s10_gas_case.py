# -*- coding: utf-8 -*-
"""S.10 案例研究图：气路 setpoint->gain 边的 gamma/delta/D_JS 滚动轨迹（补充材料）。

复用 run_gas_case.py 的冻结口径（K=4、blk=32、攻击窗 +-64、参考段全量表、
four_tuple 内部 ipf_uniform(alpha=8, iters=2)），并在锚点窗口对未四舍五入值
做逐位断言（与 results/paper/gas_case_tuples.json 的 edge(0,1) attack#19 行一致），
断言失败即退出非零——正文引用的 0.48/0.50、+0.28/-0.34、0.23 不可复现则本图不生成。
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rdsg.datasets import load_gas_pipeline
from rdsg.granulation import Granulator
from rdsg.tables import block_tables
from rdsg.stats import ipf_uniform, gamma_soft, grain_direction, cond_js


def _results_root():
    d = os.path.dirname(os.path.abspath(__file__))
    for base in (d, os.path.dirname(d), os.path.dirname(os.path.dirname(d))):
        if os.path.isdir(os.path.join(base, "results")):
            return os.path.join(base, "results")
    return os.path.normpath(os.path.join(d, "..", "..", "results"))


OUT = os.path.join(_results_root(), "paper", "figs", "s10_gas_case")
K, BLK, W = 4, 32, 128
EDGE = ((0,), 1, -1)          # setpoint -> gain, lag -1
CASE_EV = 19                  # 正文案例：攻击#19（label 1+2+4+5+7）
# 权威锚点 = 冻结结果文件的全精度值（不手抄，避免截断）
_frozen = json.load(open(os.path.join(_results_root(), "paper",
                                      "gas_case_tuples.json"), encoding="utf-8"))
_row = next(r for r in _frozen if r["edge"] == [0, 1] and r["ev"] == CASE_EV)
ANCHOR = dict(g_ref=_row["g_ref"], g_cur=_row["g_cur"],
              d_ref=_row["d_ref"], d_cur=_row["d_cur"], djs=_row["djs"])


def win_stats(G, tab_ref, w0, w1):
    tab = block_tables(G, [EDGE], K, BLK, t0=w0, t1=w1)[0].sum(axis=0)
    p_ref, p_cur = ipf_uniform(tab_ref[None]), ipf_uniform(tab[None])
    return (float(gamma_soft(p_cur)), float(grain_direction(p_cur, K)),
            float(cond_js(p_ref, p_cur)))


def main():
    X, spp, meta = load_gas_pipeline()
    ref0, ref1 = meta["ref"]
    t0, t1 = meta["test"]
    events = meta["events"]
    g = Granulator(K=K).fit(X[ref0:ref1])
    G = g.transform(X)
    tab_ref = block_tables(G, [EDGE], K, BLK, t0=ref0, t1=ref1)[0].sum(axis=0)
    p_ref = ipf_uniform(tab_ref[None])
    g_ref_line = float(gamma_soft(p_ref))
    d_ref_line = float(grain_direction(p_ref, K))

    # ---- 锚点窗口（与 run_gas_case.py 完全同构的窗口构造）逐位断言 ----
    ev = events[CASE_EV]
    s_abs, e_abs = t0 + ev[0], t0 + ev[1]
    w0 = max(ref1, s_abs - 64)
    w1 = min(len(X), e_abs + 64)
    if w1 - w0 < 128:
        w0 = max(ref1, w1 - 128)
    gc, dc, js = win_stats(G, tab_ref, w0, w1)
    ok = (abs(gc - ANCHOR["g_cur"]) < 1e-9 and abs(dc - ANCHOR["d_cur"]) < 1e-9
          and abs(js - ANCHOR["djs"]) < 1e-9
          and abs(g_ref_line - ANCHOR["g_ref"]) < 1e-9
          and abs(d_ref_line - ANCHOR["d_ref"]) < 1e-9)
    print(f"anchor window [{w0},{w1}): gamma {g_ref_line:.6f}->{gc:.6f} "
          f"delta {d_ref_line:+.6f}->{dc:+.6f} D_JS {js:.6f}  assert={'PASS' if ok else 'FAIL'}")
    if not ok:
        sys.exit("S.10 anchor assertion FAILED - frozen case values not reproduced")

    # ---- 滚动轨迹：w=128 步长 64，覆盖整个测试段 ----
    xs, gs, ds, js_ = [], [], [], []
    for a in range(ref1, len(X) - W + 1, 64):
        gv, dv, jv = win_stats(G, tab_ref, a, a + W)
        xs.append((a + W / 2 - t0) / spp / 60.0)   # 分钟（测试段起点为 0）
        gs.append(gv); ds.append(dv); js_.append(jv)
    att = [(t0 + e[0], t0 + e[1]) for e in events]
    att_min = [(a - t0) / spp / 60.0 for b in att for a in b]

    # ---- 绘图 ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(3, 1, figsize=(6.6, 4.6), sharex=True,
                             constrained_layout=True)
    tracks = [(gs, r"$\gamma$", "#c0392b", (0.0, 1.0)),
              (ds, r"$\delta$", "#2980b9", (-1.0, 1.0)),
              (js_, r"$D_{JS}$", "#16a085", (0.0, 0.6))]
    case_a, case_b = (w0 - t0) / spp / 60.0, (w1 - t0) / spp / 60.0
    for ax, (y, lab, c, ylim) in zip(axes, tracks):
        for (a0, a1) in att:
            ax.axvspan((a0 - t0) / spp / 60.0, (a1 - t0) / spp / 60.0,
                       color="0.82", lw=0, zorder=0)
        ax.axvspan(case_a, case_b, color="#f9c74f", alpha=0.55, lw=0, zorder=1)
        ax.plot(xs, y, color=c, lw=1.0, zorder=3)
        ax.set_ylabel(lab, fontsize=10)
        ax.set_ylim(*ylim)
        ax.tick_params(labelsize=8)
    axes[0].axhline(g_ref_line, color="#c0392b", ls=":", lw=0.8)
    axes[1].axhline(d_ref_line, color="#2980b9", ls=":", lw=0.8)
    axes[0].annotate(f"ref {g_ref_line:.2f}", xy=(xs[0], g_ref_line),
                     xytext=(4, 4), textcoords="offset points", fontsize=7, color="#c0392b")
    axes[1].annotate(f"ref {d_ref_line:+.2f}", xy=(xs[-1], d_ref_line),
                     xytext=(-4, -10), textcoords="offset points", fontsize=7,
                     color="#2980b9", ha="right")
    axes[0].annotate(f"{ANCHOR['g_cur']:.2f}", xy=(case_a, ANCHOR["g_cur"]),
                     xytext=(2, 5), textcoords="offset points", fontsize=7,
                     color="#8a5a00", fontweight="bold")
    axes[1].annotate(f"{ANCHOR['d_cur']:+.2f}", xy=(case_a, ANCHOR["d_cur"]),
                     xytext=(2, -9), textcoords="offset points", fontsize=7,
                     color="#8a5a00", fontweight="bold")
    axes[2].annotate(f"{ANCHOR['djs']:.2f}", xy=(case_b, ANCHOR["djs"]),
                     xytext=(-14, 4), textcoords="offset points", fontsize=7,
                     color="#8a5a00", fontweight="bold")
    axes[2].set_xlabel("Minutes from test-segment start (MSU gas pipeline)")
    axes[2].set_xlim(-2, max(att_min) + 2)
    fig.savefig(OUT + ".pdf", metadata={"Creator": "", "Producer": "",
                                        "CreationDate": None})
    fig.savefig(OUT + ".png", dpi=200)
    print("wrote", OUT + ".pdf/.png")


if __name__ == "__main__":
    main()
