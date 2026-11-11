# -*- coding: utf-8 -*-
"""Generate paper enhancement figures: dependency-graph visualisation + runtime comparison."""
import os

def _results_root():
    """Results directory: nearest ancestor shipping a results/ folder.

    Reviewer package: <pkg>/experiments/x.py -> <pkg>/results (the shipped
    frozen JSONs), so reruns merge with them instead of writing outside;
    local repo layout resolves to <project>/results."""
    d = os.path.dirname(os.path.abspath(__file__))
    for base in (d, os.path.dirname(d), os.path.dirname(os.path.dirname(d))):
        if os.path.isdir(os.path.join(base, "results")):
            return os.path.join(base, "results")
    return os.path.normpath(os.path.join(d, "..", "..", "results"))

import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = _results_root()
RES = os.path.normpath(os.path.join(_results_root()))
FIG = os.path.join(RES, "paper", "figs")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 9, "figure.dpi": 150,
                     "axes.spines.top": False, "axes.spines.right": False})


def fig_dependency_graph():
    """Significant dependency-edge subgraph of the synthetic system: normal (reference segment) vs anomalous (after injection)."""
    sys.path.insert(0, os.path.join(ROOT, "code"))
    sys.path.insert(0, os.path.join(ROOT, "code", "experiments"))
    from run_e0_syn import make_synthetic, rank_pairs_by_ref_gamma
    from run_e0_syn2 import REF_LEN, TEST_LEN
    from rdsg import RDSGDetector, Granulator, inject
    from rdsg.tables import edge_code
    from rdsg.stats import gamma_soft

    Xs = make_synthetic(REF_LEN + TEST_LEN, seed=101)
    X_ref = Xs[:REF_LEN]
    K = 4
    gran = Granulator(K).fit(X_ref)
    Gref = gran.transform(X_ref[-2048:])

    # collect gamma for all pairs
    pairs_g = {}
    for i in range(12):
        for j in range(12):
            if i == j:
                continue
            code = edge_code(Gref, (i,), j, 0, K)
            c = code[code >= 0]
            tab = np.bincount(c, minlength=K * K).reshape(K, K).astype(float)
            from rdsg.stats import std_gamma
            pairs_g[(i, j)] = float(std_gamma(tab[None])[0])

    # inject one INV event on (0,1)
    from rdsg import inject
    X_test = Xs[REF_LEN:]
    Xi = inject(X_test, [dict(type="inv", i=0, j=1, start=2000, length=500)], seed=1)
    Xi_full = np.vstack([X_ref, Xi])
    Gfull = gran.transform(Xi_full[-2048:])
    pairs_g_after = {}
    for i in range(12):
        for j in range(12):
            if i == j:
                continue
            code = edge_code(Gfull, (i,), j, 0, K)
            c = code[code >= 0]
            tab = np.bincount(c, minlength=K * K).reshape(K, K).astype(float)
            pairs_g_after[(i, j)] = float(std_gamma(tab[None])[0])
    # delta direction
    from rdsg.stats import grain_direction
    pairs_d_before, pairs_d_after = {}, {}
    for i in range(12):
        for j in range(12):
            if i == j:
                continue
            code = edge_code(Gref, (i,), j, 0, K)
            c = code[code >= 0]
            tab = np.bincount(c, minlength=K * K).reshape(K, K).astype(float)
            pairs_d_before[(i, j)] = float(grain_direction(tab[None], K)[0])
            code2 = edge_code(Gfull, (i,), j, 0, K)
            c2 = code2[code2 >= 0]
            tab2 = np.bincount(c2, minlength=K * K).reshape(K, K).astype(float)
            pairs_d_after[(i, j)] = float(grain_direction(tab2[None], K)[0])

    # draw two subplots: before / after
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    rng = np.random.default_rng(42)
    # circular layout
    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    pos = {i: (np.cos(angles[i]), np.sin(angles[i])) for i in range(12)}
    group_colors = {i: "#3498db" if i < 3 else "#e74c3c" if i < 5 else
                     "#2ecc71" if i < 8 else "#95a5a6" for i in range(12)}

    for ax, pg, pd_, title in [(ax1, pairs_g_before if False else pairs_g, pairs_d_before, "Reference"),
                                (ax2, pairs_g_after, pairs_d_after, "After INV injection")]:
        ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(title, fontsize=10)
        # nodes
        for i in range(12):
            ax.add_patch(plt.Circle(pos[i], 0.08, color=group_colors[i], ec="black", lw=0.5, zorder=3))
            ax.text(pos[i][0], pos[i][1], str(i), ha="center", va="center", fontsize=7, zorder=4)
        # edges: draw only those above the gamma threshold
        for (i, j), g in pg.items():
            if g < 0.30 or i >= j:
                continue
            d = pd_.get((i, j), 0)
            lw = max(0.5, g * 3)
            style = "-" if d >= 0 else "--"
            color = "#e74c3c" if d < 0 else "#3498db"
            dx, dy = pos[j][0] - pos[i][0], pos[j][1] - pos[i][1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < 0.01:
                continue
            ax.annotate("", xy=(pos[j][0] - 0.09 * dx / dist, pos[j][1] - 0.09 * dy / dist),
                        xytext=(pos[i][0] + 0.09 * dx / dist, pos[i][1] + 0.09 * dy / dist),
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                       linestyle=style, alpha=0.7))
    # legend
    for ax in (ax1, ax2):
        for i, gi in group_colors.items():
            pass
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "dep_graph.pdf"))
    print("dep_graph saved")


def fig_runtime():
    """Runtime comparison bar chart (runtime extracted from e1_gas.json)."""
    d = json.load(open(os.path.join(RES, "paper", "e1_gas.json"), encoding="utf-8"))
    methods = [("rdsg_runtime", "RDSG (ours)"), ("b1_corr_runtime", "Corr-matrix"),
               ("b5c_var_runtime", "VAR-resid"), ("b2_granger_runtime", "Granger"),
               ("b4_if_runtime", "IF-agg"), ("b4_lof_runtime", "LOF-agg")]
    labels, times = [], []
    for k, lab in methods:
        if k in d:
            labels.append(lab)
            times.append(d[k])
    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    bars = ax.barh(labels, times, color=["#c0392b" if "ours" in l else "#7f8c8d" for l in labels])
    ax.set_xlabel("Runtime (seconds, full pipeline)")
    ax.set_title("Gas Pipeline (75K points, 9 channels)")
    for bar, t in zip(bars, times):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{t:.1f}s", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "runtime.pdf"))
    print("runtime saved")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "graph"):
        fig_dependency_graph()
    if which in ("all", "runtime"):
        fig_runtime()
