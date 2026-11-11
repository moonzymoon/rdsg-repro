# -*- coding: utf-8 -*-
"""Figure 1 v4: refined architecture diagram -- modelled on TPAMI/TNNLS pipeline-figure style.

Design principles:
- horizontal pipeline + vertical layering (data layer -> method layer -> output layer)
- one rounded rectangle per stage, sub-modules as small colour blocks
- thick gradient-coloured arrows for a strong sense of flow
- Nature 7-colour palette + minimal grey-white background
- no text overlap, uniform font sizes, precisely controlled spacing
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches
import numpy as np
import os

# ===== palette =====
C = {
    "bg":       "#F8F9FB",
    "input":    "#DDE6F3",
    "gran":     "#7FA4C8",
    "table":    "#B59BC8",
    "copula":   "#8FB79B",
    "stat_g":   "#6A5A8F",
    "stat_r":   "#D08AB6",
    "stat_d":   "#7FA4C8",
    "stat_js":  "#B59BC8",
    "r_ch":     "#3F345B",
    "e_ch":     "#D08AB6",
    "alarm":    "#6A5A8F",
    "output":   "#DDE6F3",
    "arrow":    "#7FA4C8",
    "text":     "#2C3E50",
    "text_w":   "#FFFFFF",
    "text_lt":  "#5A7A9A",
    "border":   "#A0B4C8",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
})


def rbox(ax, x, y, w, h, fc, ec=None, r=0.015, alpha=1.0, lw=0, z=2):
    b = FancyBboxPatch((x, y), w, h,
                        boxstyle=f"round,pad={r}",
                        facecolor=fc, edgecolor=ec or fc,
                        linewidth=lw, alpha=alpha, zorder=z)
    ax.add_patch(b)
    return b


def txt(ax, x, y, s, fs=8, c="#2C3E50", weight="normal", ha="center", va="center", z=5, style="normal"):
    ax.text(x, y, s, fontsize=fs, color=c, weight=weight,
            ha=ha, va=va, zorder=z, style=style, linespacing=1.35)


def arr(ax, x1, y1, x2, y2, c=None, lw=2.0, style="-|>", ms=13):
    cc = c or C["arrow"]
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=cc, lw=lw,
                                mutation_scale=ms, shrinkA=0, shrinkB=0,
                                capstyle="round"), zorder=3)


def main():
    fig, ax = plt.subplots(figsize=(14, 5.2))
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # ================= top title =================
    # no title -- provided by the paper caption

    # ================= stage 1: input =================
    # main box
    rbox(ax, 0.00, 0.38, 0.10, 0.35, C["input"], C["border"], r=0.012, lw=0.8)
    txt(ax, 0.05, 0.62, "Multivariate\nTime Series", fs=9, c=C["text"], weight="bold")
    txt(ax, 0.05, 0.49, "$X \\in \\mathbb{R}^{n \\times T}$", fs=7.5, c=C["text_lt"])

    # sub-boxes (reference / test segment)
    rbox(ax, 0.005, 0.10, 0.042, 0.14, "#EBF0F7", "#C8D4E0", r=0.008, lw=0.5)
    txt(ax, 0.026, 0.17, "Reference\n$W_{\\mathrm{ref}}$", fs=6, c=C["text_lt"])
    rbox(ax, 0.053, 0.10, 0.042, 0.14, "#EBF0F7", "#C8D4E0", r=0.008, lw=0.5)
    txt(ax, 0.074, 0.17, "Test\n$W_{\\mathrm{tst}}$", fs=6, c=C["text_lt"])

    # ================= stage 2: granulation =================
    rbox(ax, 0.14, 0.35, 0.10, 0.38, C["gran"], r=0.012)
    txt(ax, 0.19, 0.62, "Granulation", fs=9, c=C["text_w"], weight="bold")
    txt(ax, 0.19, 0.48, "$\\Phi^K$\nEqual-frequency\n$K$ levels", fs=7, c=C["text_w"])
    txt(ax, 0.19, 0.38, "frozen on\nreference", fs=6, c="#D0DCED", style="italic")

    # ================= stage 3: decision tables =================
    rbox(ax, 0.28, 0.35, 0.10, 0.38, C["table"], r=0.012)
    txt(ax, 0.33, 0.62, "Granular\nDecision Tables", fs=9, c=C["text_w"], weight="bold")
    txt(ax, 0.33, 0.48, "$T \\in \\mathbb{N}^{K^{|A|} \\times K}$", fs=7, c=C["text_w"])
    txt(ax, 0.33, 0.38, "counts granule\npairs per block", fs=6, c="#E0D4EC")

    # ================= stage 4: statistic computation =================
    # large background box (dashed border, light fill)
    bg = FancyBboxPatch((0.42, 0.05), 0.22, 0.88,
                         boxstyle="round,pad=0.012",
                         facecolor="#F7F3FA", edgecolor=C["table"],
                         linewidth=1.2, linestyle="--", zorder=1, alpha=0.6)
    ax.add_patch(bg)
    txt(ax, 0.53, 0.90, "Four-tuple Edge Encoding", fs=9, c=C["table"], weight="bold")

    # Copula IPF
    rbox(ax, 0.435, 0.60, 0.09, 0.18, C["copula"], r=0.010)
    txt(ax, 0.48, 0.69, "Copula IPF", fs=8, c=C["text_w"], weight="bold")
    txt(ax, 0.48, 0.64, "$\\alpha{=}8$, 2 iters\nuniform margins", fs=6, c="#D8E8D8")

    # four statistics (stacked vertically, aligned widths)
    stats = [
        ("$\\gamma$", "Soft variable-precision\ndependency degree", 0.48, C["stat_g"]),
        ("$\\rho$", "Modal rule-map\nconsistency", 0.36, C["stat_r"]),
        ("$\\delta$", "Granular direction\n(sign of association)", 0.24, C["stat_d"]),
        ("$D_{JS}$", "Conditional Jensen–\nShannon divergence", 0.12, C["stat_js"]),
    ]
    for sym, desc, sy, sc in stats:
        rbox(ax, 0.445, sy, 0.07, 0.09, sc, r=0.008)
        txt(ax, 0.48, sy + 0.045, sym, fs=10, c=C["text_w"], weight="bold")
        # side text description
        txt(ax, 0.55, sy + 0.045, desc, fs=6, c=C["text"], ha="left")

    # detection family annotation
    txt(ax, 0.53, 0.08, "Detection family: $\\Delta\\gamma + \\Delta\\delta$",
        fs=7, c=C["stat_g"], weight="bold")
    txt(ax, 0.53, 0.11, "Attribution: $\\rho$, $D_{JS}$",
        fs=6, c=C["text_lt"], style="italic")

    # ================= stage 5: dual-channel graphs =================
    # R channel
    rbox(ax, 0.68, 0.52, 0.11, 0.26, C["r_ch"], r=0.012)
    txt(ax, 0.735, 0.65, "R Channel", fs=8, c=C["text_w"], weight="bold")
    txt(ax, 0.735, 0.57, "Top-16 strong pairs\n(breakdown, inversion)", fs=6, c="#C8B8D8")

    # E channel
    rbox(ax, 0.68, 0.12, 0.11, 0.26, C["e_ch"], r=0.012)
    txt(ax, 0.735, 0.25, "E Channel", fs=8, c=C["text_w"], weight="bold")
    txt(ax, 0.735, 0.17, "Bottom-24 weak pairs\n(emergence, subset-joint)", fs=6, c="#F0D8E0")

    # ================= stage 6: calibrated alarm =================
    rbox(ax, 0.83, 0.25, 0.08, 0.45, C["alarm"], r=0.012)
    txt(ax, 0.87, 0.58, "Calibrated\nAlarm", fs=8, c=C["text_w"], weight="bold")
    txt(ax, 0.87, 0.42, "Self-scan\nFA budget\n\nBlock perm.\nattribution", fs=6.5, c="#D0C4DC")

    # ================= stage 7: output =================
    rbox(ax, 0.94, 0.45, 0.055, 0.25, C["output"], C["border"], r=0.010, lw=0.8)
    txt(ax, 0.968, 0.57, "Alarm\nEvents", fs=8, c=C["text"], weight="bold")
    rbox(ax, 0.94, 0.10, 0.055, 0.22, C["output"], C["border"], r=0.010, lw=0.8)
    txt(ax, 0.968, 0.21, "Edge\nattribution\n(BH-FDR)", fs=6, c=C["text_lt"])

    # ================= arrows =================
    Y_MID = 0.53
    # input -> granulation
    arr(ax, 0.10, Y_MID, 0.14, Y_MID)
    # granulation -> tables
    arr(ax, 0.24, Y_MID, 0.28, Y_MID)
    # tables -> copula
    arr(ax, 0.38, 0.58, 0.42, 0.66, c=C["copula"])
    # tables -> statistics (straight down)
    arr(ax, 0.38, 0.42, 0.42, 0.35, c=C["table"])
    # copula -> gamma
    arr(ax, 0.48, 0.60, 0.445, 0.57, c=C["copula"], lw=1.5)
    # statistics -> R channel
    arr(ax, 0.515, 0.62, 0.68, 0.62, c=C["stat_g"])
    # statistics -> E channel
    arr(ax, 0.515, 0.18, 0.68, 0.18, c=C["e_ch"])
    # R -> calibration
    arr(ax, 0.79, 0.62, 0.83, 0.55, c=C["r_ch"])
    # E -> calibration
    arr(ax, 0.79, 0.22, 0.83, 0.35, c=C["e_ch"])
    # calibration -> output
    arr(ax, 0.91, 0.55, 0.94, 0.55, c=C["alarm"])

    # ================= separator dashed lines =================
    ax.axvline(0.42, ymin=0.02, ymax=0.98, color=C["table"], lw=0.8,
              linestyle="--", alpha=0.3, zorder=0)
    ax.axvline(0.655, ymin=0.02, ymax=0.98, color=C["r_ch"], lw=0.8,
              linestyle="--", alpha=0.2, zorder=0)
    ax.axvline(0.815, ymin=0.02, ymax=0.98, color=C["alarm"], lw=0.8,
              linestyle="--", alpha=0.2, zorder=0)

    # ================= bottom annotation =================
    txt(ax, 0.50, 0.00,
        "Offline: granulation fit + edge selection + calibration ($\\mu$, $\\sigma$, $\\theta$)   "
        "|   Online: per-window table update + statistic + alarm",
        fs=6.5, c=C["text_lt"], style="italic")

    plt.tight_layout(pad=0.3)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "..", "paper", "figs", "pipeline.pdf")
    plt.savefig(out, dpi=250, bbox_inches="tight", facecolor="white")
    plt.savefig(out.replace(".pdf", ".png"), dpi=250, bbox_inches="tight",
                facecolor="white")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
