# -*- coding: utf-8 -*-
"""全部图更新为 Nature 封面配色（2026-10-19）。"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
def _results_root():
    """Results directory: the nearest ancestor that ships a results/ folder.

    Local repo: code/experiments/x.py -> <project>/results.
    Reviewer package: <pkg>/experiments/x.py -> <pkg>/results (the shipped
    frozen JSONs), so reruns merge with them instead of writing outside."""
    d = os.path.dirname(os.path.abspath(__file__))
    for base in (d, os.path.dirname(d), os.path.dirname(os.path.dirname(d))):
        if os.path.isdir(os.path.join(base, "results")):
            return os.path.join(base, "results")
    return os.path.normpath(os.path.join(d, "..", "..", "results"))


ROOT = _results_root()
RES = os.path.normpath(os.path.join(_results_root()))
FIG = os.path.join(RES, "paper", "figs")
os.makedirs(FIG, exist_ok=True)

# 统一语义配色（与 TikZ figurestyle 对齐；= figurestyle.mplstyle 色循环）
C_BLUE, C_RED, C_GREEN, C_GRAY, C_ORANGE, C_PURPLE = (
    "#00008C", "#990000", "#2E7D32", "#555555", "#C77C02", "#7B3FA0")
CYCLE = [C_BLUE, C_RED, C_GREEN, C_GRAY, C_ORANGE, C_PURPLE]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "figure.titlesize": 9,
    "lines.linewidth": 1.4, "lines.markersize": 4,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.direction": "in", "ytick.direction": "in",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#555555", "axes.labelcolor": "black",
    "xtick.color": "#555555", "ytick.color": "#555555",
    "legend.frameon": False,
    "grid.linewidth": 0.4, "grid.alpha": 0.3, "grid.linestyle": ":",
    "savefig.format": "pdf", "pdf.fonttype": 42, "figure.dpi": 120,
    "axes.prop_cycle": plt.cycler("color", CYCLE),
})


def fig_e0_mechanism():
    """图2: 能力门控——Nature 配色柱图。"""
    d = json.load(open(os.path.join(RES, "e0", "e0_syn2_summary.json"), encoding="utf-8"))
    s = d["summary"]
    types = ["inv", "brk", "emb"]
    type_labels = ["Inversion", "Breakdown", "Emergence"]
    methods = [("rdsg", "RDSG (detection family)"), ("gamma_only", "γ-only"),
               ("b4_if", "IF-agg"), ("b4_lof", "LOF-agg")]
    colors = {"rdsg": C_BLUE, "gamma_only": C_PURPLE,
              "b4_if": C_ORANGE, "b4_lof": C_GRAY}
    x = np.arange(len(types))
    w = 0.18
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    for i, (mk, mlab) in enumerate(methods):
        vals = [s[mk]["recall_by_type"].get(t, 0) for t in types]
        bars = ax.bar(x + (i - 1.5) * w, vals, w, label=mlab,
                      color=colors[mk], edgecolor="white", linewidth=0.5)
        for b, v in zip(bars, vals):
            if v > 0.02:
                ax.text(b.get_x() + b.get_width()/2, v + 0.015,
                        f"{v:.2f}", ha="center", fontsize=6, color="black")
    ax.set_xticks(x); ax.set_xticklabels(type_labels)
    ax.set_ylabel("Event recall")
    ax.set_ylim(0, 0.65)
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("Capability gate (synthetic, 10 seeds)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "e0_mechanism.pdf"), facecolor="white")
    fig.savefig(os.path.join(FIG, "e0_mechanism.png"), dpi=200, facecolor="white")
    print("e0_mechanism saved")


def fig_e1_curves():
    """图4: Gas pipeline 工作曲线——Nature 配色。"""
    d = json.load(open(os.path.join(RES, "paper", "e1_gas.json"), encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    styles = {"rdsg": (C_BLUE, "o", "RDSG (ours)"),
              "b4_if": (C_ORANGE, "s", "IF-agg"),
              "b4_lof": (C_GRAY, "^", "LOF-agg"),
              "b1_corr": (C_PURPLE, "v", "Corr-matrix"),
              "b5b_pcorr": (C_GREEN, "D", "Partial-corr"),
              "b2_granger": (C_RED, "x", "Granger")}
    for m, (c, mk, lab) in styles.items():
        if m not in d["curves"]: continue
        pts = [p for p in d["curves"][m] if p["fa_per_day"] > 0 or p["recall"] > 0]
        if not pts: continue
        xs = [p["fa_per_day"] for p in pts]
        ys = [p["recall"] for p in pts]
        ax.plot(xs, ys, color=c, marker=mk, markersize=3.5, linewidth=1.5,
                label=lab, alpha=0.9)
    ax.set_xscale("log")
    ax.set_xlabel("False alarms per day")
    ax.set_ylabel("Event recall")
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("Gas pipeline operating curves", fontsize=9)
    ax.grid(axis="y", alpha=0.3, color=C_GRAY)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "e1_curves.pdf"), facecolor="white")
    fig.savefig(os.path.join(FIG, "e1_curves.png"), dpi=200, facecolor="white")
    print("e1_curves saved")


def fig_e6_component():
    """图3: component-firing 矩阵——Nature 配色热图。"""
    d = json.load(open(os.path.join(RES, "paper", "e2d_component.json"), encoding="utf-8"))
    types = ["inv", "brk", "emb", "emb2"]
    cols = ["ddelta", "dgamma", "E"]
    idx = {t: i for i, t in enumerate(types)}
    C = np.zeros((4, 3))
    for seed, v in d["seeds"].items():
        for e in v["events"]:
            if e["pred"] is None: continue
            if e["chan"] == "E":
                col = "E"
            else:
                col = "ddelta" if e["pred"] == "inv" else "dgamma"
            C[idx[e["truth"]], cols.index(col)] += 1
    Cn = C / np.maximum(C.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(3.8, 3.0))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("unified",
        ["#FFFFFF", "#7FA4C8", "#3B5BA5", "#00008C"])
    im = ax.imshow(Cn, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(3)); ax.set_xticklabels(cols, fontsize=8)
    ax.set_yticks(range(4)); ax.set_yticklabels(types, fontsize=8)
    ax.set_xlabel("Fired component / channel", fontsize=9)
    ax.set_ylabel("Injected type", fontsize=9)
    for i in range(4):
        for j in range(3):
            color = "#FFFFFF" if Cn[i,j] > 0.5 else "#3F345B"
            ax.text(j, i, f"{Cn[i,j]:.2f}", ha="center", va="center",
                    fontsize=8, color=color)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Fraction", fontsize=8)
    ax.set_title("Component-firing matrix (METR-LA)", fontsize=9, pad=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "e6_component.pdf"), facecolor="white")
    fig.savefig(os.path.join(FIG, "e6_component.png"), dpi=200, facecolor="white")
    print("e6_component saved")


def fig_runtime():
    """图5: 运行时对比——Nature 配色条图。"""
    d = json.load(open(os.path.join(RES, "paper", "e1_gas.json"), encoding="utf-8"))
    methods = [("rdsg_runtime", "RDSG"), ("b1_corr_runtime", "Corr"),
               ("b2_granger_runtime", "Granger"), ("b4_if_runtime", "IF"),
               ("b5c_var_runtime", "VAR")]
    labels = [m[1] for m in methods]
    vals = [d.get(k, 0) for k, _ in methods]
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    colors = [C_BLUE if l == "RDSG" else C_GRAY for l in labels]
    bars = ax.barh(labels, vals, color=colors, edgecolor="white", height=0.6)
    for b, v in zip(bars, vals):
        ax.text(v + 0.2, b.get_y() + b.get_height()/2,
                f"{v:.1f}s", va="center", fontsize=8, color="black")
    ax.set_xlabel("Runtime (s)")
    ax.set_title("Full-pipeline runtime (gas pipeline, single core)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "runtime.pdf"), facecolor="white")
    fig.savefig(os.path.join(FIG, "runtime.png"), dpi=200, facecolor="white")
    print("runtime saved")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "e0"):
        fig_e0_mechanism()
    if which in ("all", "e1"):
        fig_e1_curves()
    if which in ("all", "e6"):
        fig_e6_component()
    if which in ("all", "rt"):
        fig_runtime()
    print("done")
