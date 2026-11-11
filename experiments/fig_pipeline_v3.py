# -*- coding: utf-8 -*-
"""Figure 1 v3: professional architecture diagram -- horizontal layered layout, clear spacing, Nature palette."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os

PAL = {
    "pale":   "#DDE6F3",
    "peri":   "#7FA4C8",
    "lilac":  "#B59BC8",
    "orchid": "#D08AB6",
    "mint":   "#8FB79B",
    "violet": "#6A5A8F",
    "indigo": "#3F345B",
    "dark":   "#2C3E50",
    "white":  "#FFFFFF",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
})


def box(ax, x, y, w, h, text, fc, tc="#FFFFFF", fs=8, bold=True, r=0.03, lw=0, ec=None):
    b = FancyBboxPatch((x, y), w, h,
                        boxstyle=f"round,pad={r}",
                        facecolor=fc, edgecolor=ec or fc,
                        linewidth=lw if lw else 0.5, zorder=2)
    ax.add_patch(b)
    ax.text(x+w/2, y+h/2, text, ha="center", va="center",
            fontsize=fs, color=tc, weight="bold" if bold else "normal",
            zorder=3, linespacing=1.4)
    return b


def subbox(ax, x, y, w, h, text, fc, tc="#FFFFFF", fs=6.5):
    return box(ax, x, y, w, h, text, fc, tc, fs, bold=False, r=0.015)


def arrow_h(ax, x1, y, x2, color="#7FA4C8", lw_val=1.5):
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw_val,
                                mutation_scale=14, shrinkA=1, shrinkB=1), zorder=1)


def arrow_diag(ax, x1, y1, x2, y2, color="#7FA4C8", lw=1.5):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=12, shrinkA=1, shrinkB=1), zorder=1)


def main():
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.axis("off")

    # ===== stage labels (top) =====
    stages = [
        (0.00, 0.14, "INPUT"),
        (0.17, 0.10, "GRANULATION"),
        (0.30, 0.10, "TABLES"),
        (0.44, 0.18, "COPULA + STATISTICS"),
        (0.66, 0.10, "GRAPH"),
        (0.79, 0.10, "ALARM"),
        (0.92, 0.08, "OUTPUT"),
    ]
    for sx, sw, st in stages:
        ax.text(sx + sw/2, 0.95, st, ha="center", va="center",
                fontsize=8, color=PAL["indigo"], weight="bold")

    # ===== 1. input =====
    box(ax, 0.00, 0.35, 0.13, 0.40,
        "Multivariate\nTime Series\n$X \\in \\mathbb{R}^{n \\times T}$",
        PAL["pale"], PAL["dark"], fs=9)
    # sub-boxes
    subbox(ax, 0.01, 0.08, 0.05, 0.18, "Reference\n$W_{\\mathrm{ref}}$", PAL["pale"], PAL["dark"])
    subbox(ax, 0.07, 0.08, 0.05, 0.18, "Test\n$W_{\\mathrm{tst}}$", PAL["pale"], PAL["dark"])
    arrow_diag(ax, 0.035, 0.26, 0.035, 0.35, PAL["peri"])
    arrow_diag(ax, 0.095, 0.26, 0.095, 0.35, PAL["peri"])

    # ===== 2. granulation =====
    box(ax, 0.17, 0.30, 0.10, 0.45,
        "Equal-freq\nGranulation\n\n$\\Phi^K$\n\n$K$ levels per\nchannel",
        PAL["peri"], fs=8)
    ax.text(0.22, 0.16, "Frozen on\nreference", ha="center", va="center",
            fontsize=6.5, color=PAL["dark"], style="italic")

    # ===== 3. decision tables =====
    box(ax, 0.31, 0.30, 0.10, 0.45,
        "Granular\nDecision\nTables\n\n$T \\in \\mathbb{N}^{K^{|A|} \\times K}$",
        PAL["lilac"], fs=8)

    # ===== 4. copula + four statistics =====
    # large background box
    bg = FancyBboxPatch((0.44, 0.05), 0.18, 0.85,
                         boxstyle="round,pad=0.01",
                         facecolor="#F5F0FA", edgecolor=PAL["lilac"],
                         linewidth=1.2, linestyle="--", zorder=1, alpha=0.5)
    ax.add_patch(bg)
    # IPF
    box(ax, 0.46, 0.60, 0.06, 0.22,
        "Copula\nIPF\n\n$\\alpha{=}8$\n2 iters",
        PAL["mint"], fs=7)
    # four statistics, 2x2
    subbox(ax, 0.535, 0.60, 0.06, 0.10, "$\\gamma$\n(degree)", PAL["violet"])
    subbox(ax, 0.535, 0.47, 0.06, 0.10, "$\\rho$\n(rule map)", PAL["orchid"])
    subbox(ax, 0.535, 0.34, 0.06, 0.10, "$\\delta$\n(direction)", PAL["peri"])
    subbox(ax, 0.535, 0.21, 0.06, 0.10, "$D_{JS}$\n(diverg.)", PAL["lilac"])
    # detection family annotation
    ax.text(0.53, 0.12, "Detection family: $\\Delta\\gamma + \\Delta\\delta$",
            ha="center", fontsize=6.5, color=PAL["indigo"], weight="bold")
    ax.text(0.53, 0.07, "Attribution: $\\rho$, $D_{JS}$",
            ha="center", fontsize=6, color=PAL["dark"], style="italic")

    # ===== 5. dual-channel graphs =====
    box(ax, 0.66, 0.52, 0.10, 0.28,
        "R Channel\n\nTop-16\nstrong pairs\n(breakdown,\ninversion)",
        PAL["indigo"], fs=7)
    box(ax, 0.66, 0.10, 0.10, 0.28,
        "E Channel\n\nBottom-24\nweak pairs\n(emergence,\nsubset-joint)",
        PAL["orchid"], fs=7)
    ax.text(0.71, 0.02, "Fixed edge set", ha="center",
            fontsize=6.5, color=PAL["dark"], style="italic")

    # ===== 6. calibrated alarm =====
    box(ax, 0.79, 0.25, 0.10, 0.50,
        "Calibrated\nAlarm\n\nSelf-scan\nFA budget\n\nBlock perm.\nattribution",
        PAL["violet"], fs=7)

    # ===== 7. output =====
    box(ax, 0.92, 0.45, 0.08, 0.30,
        "Alarm\nEvents",
        PAL["pale"], PAL["dark"], fs=9)
    subbox(ax, 0.92, 0.08, 0.08, 0.22,
           "Edge\nattribution\n(BH-FDR)", PAL["pale"], PAL["dark"])
    arrow_diag(ax, 0.96, 0.30, 0.96, 0.45, PAL["peri"])

    # ===== arrows =====
    # input -> granulation
    arrow_h(ax, 0.13, 0.50, 0.17, PAL["peri"])
    # granulation -> decision tables
    arrow_h(ax, 0.27, 0.50, 0.31, PAL["peri"])
    # decision tables -> copula (curved upward)
    arrow_diag(ax, 0.41, 0.60, 0.46, 0.68, PAL["lilac"])
    # decision tables -> statistics (direct)
    arrow_diag(ax, 0.41, 0.40, 0.535, 0.35, PAL["lilac"])
    # copula -> statistics
    arrow_h(ax, 0.52, 0.65, 0.535, PAL["mint"], 1.0)
    # statistics -> R channel
    arrow_diag(ax, 0.595, 0.60, 0.66, 0.62, PAL["violet"])
    # statistics -> E channel
    arrow_diag(ax, 0.595, 0.25, 0.66, 0.25, PAL["orchid"])
    # R -> calibration
    arrow_h(ax, 0.76, 0.62, 0.79, PAL["indigo"])
    # E -> calibration
    arrow_diag(ax, 0.76, 0.24, 0.79, 0.35, PAL["orchid"])
    # calibration -> output
    arrow_h(ax, 0.89, 0.55, 0.92, PAL["violet"])

    # ===== bottom annotation =====
    ax.text(0.50, -0.01,
            "Offline: granulation fit + edge selection + calibration ($\\mu$, $\\sigma$, $\\theta$)   |   "
            "Online: per-window table update + statistic + alarm",
            ha="center", fontsize=7, color=PAL["dark"], style="italic")

    plt.tight_layout(pad=0.5)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "..", "paper", "figs", "pipeline.pdf")
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.savefig(out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight",
                facecolor="white")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
