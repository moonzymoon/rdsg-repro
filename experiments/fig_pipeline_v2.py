# -*- coding: utf-8 -*-
"""Redraw Figure 1 (pipeline architecture): Nature cover palette + clean layout + no text overlap.
Design follows the author's doctoral dissertation architecture-diagram style (layered flow + colour blocks + annotations)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np
import os

# Nature cover palette
PAL = {
    "pale_blue": "#DDE6F3",
    "peri_blue": "#7FA4C8",
    "lilac": "#B59BC8",
    "orchid": "#D08AB6",
    "mint": "#8FB79B",
    "violet": "#6A5A8F",
    "indigo": "#3F345B",
    "white": "#FFFFFF",
    "text_dark": "#2C3E50",
    "text_light": "#FFFFFF",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
})

def rounded_box(ax, x, y, w, h, text, facecolor, textcolor="#2C3E50",
                fontsize=8, alpha=1.0, edgecolor=None, lw=0.8, bold=False):
    box = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.02",
                          facecolor=facecolor, edgecolor=edgecolor or facecolor,
                          linewidth=lw, alpha=alpha, zorder=2)
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fontsize, color=textcolor, weight=weight, zorder=3,
            linespacing=1.3)
    return box

def arrow(ax, x1, y1, x2, y2, color="#7FA4C8", lw=1.2, style="-|>"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                shrinkA=2, shrinkB=2),
                zorder=1)

def main():
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.axis("off")

    # ===== Stage 1: input =====
    rounded_box(ax, 0.00, 0.30, 0.13, 0.40,
                "Multivariate\nTime Series\n$x(t)\\in\\mathbb{R}^n$",
                PAL["pale_blue"], fontsize=8, bold=True)
    # input sub-labels
    rounded_box(ax, 0.01, 0.05, 0.05, 0.15, "Reference\n$W_{\\mathrm{ref}}$",
                PAL["pale_blue"], fontsize=6.5, alpha=0.7)
    rounded_box(ax, 0.07, 0.05, 0.05, 0.15, "Test\n$W_{\\mathrm{tst}}$",
                PAL["pale_blue"], fontsize=6.5, alpha=0.7)

    # ===== Stage 2: granulation =====
    rounded_box(ax, 0.17, 0.30, 0.10, 0.40,
                "Granulation\n$\\Phi^K$\n(equal-freq)",
                PAL["peri_blue"], textcolor=PAL["text_light"],
                fontsize=8, bold=True)
    ax.text(0.22, 0.20, "$K$ levels", ha="center", fontsize=6.5,
            color=PAL["text_dark"], style="italic")

    # ===== Stage 3: decision tables =====
    rounded_box(ax, 0.31, 0.30, 0.10, 0.40,
                "Granular\nDecision\nTables",
                PAL["lilac"], textcolor=PAL["text_light"],
                fontsize=8, bold=True)
    ax.text(0.36, 0.20, "$T \\in \\mathbb{N}^{K^{|A|}\\times K}$",
            ha="center", fontsize=6.5, color=PAL["text_dark"])

    # ===== Stage 4: copula standardisation + four statistics =====
    rounded_box(ax, 0.45, 0.55, 0.10, 0.22,
                "Copula IPF\n(standardise)",
                PAL["mint"], textcolor=PAL["text_light"], fontsize=7)
    # four statistics (four small colour blocks)
    stats = [
        (0.45, 0.38, 0.045, 0.12, "$\\gamma$\\n(degree)", PAL["violet"]),
        (0.50, 0.38, 0.045, 0.12, "$\\rho$\\n(rule)", PAL["orchid"]),
        (0.55, 0.38, 0.045, 0.12, "$\\delta$\\n(dir)", PAL["peri_blue"]),
        (0.45, 0.22, 0.045, 0.12, "$D_{JS}$", PAL["mint"]),
    ]
    for sx, sy, sw, sh, stxt, scol in stats:
        rounded_box(ax, sx, sy, sw, sh, stxt, scol,
                    textcolor=PAL["text_light"], fontsize=6.5)
    # detection family annotation
    ax.text(0.525, 0.15, "Detection family: $\\Delta\\gamma + \\Delta\\delta$",
            ha="center", fontsize=7, color=PAL["indigo"], weight="bold")
    ax.text(0.525, 0.08, "Attribution: $\\rho$, $D_{JS}$",
            ha="center", fontsize=6.5, color=PAL["text_dark"], style="italic")

    # ===== Stage 5: dual-channel graphs =====
    rounded_box(ax, 0.63, 0.55, 0.09, 0.22,
                "R channel\n(strong pairs)",
                PAL["indigo"], textcolor=PAL["text_light"], fontsize=7, bold=True)
    rounded_box(ax, 0.63, 0.22, 0.09, 0.22,
                "E channel\n(weak pairs)",
                PAL["orchid"], textcolor=PAL["text_light"], fontsize=7, bold=True)
    ax.text(0.675, 0.12, "Dual-channel\ngraph", ha="center", fontsize=7,
            color=PAL["text_dark"], weight="bold")

    # ===== Stage 6: calibration + alarm =====
    rounded_box(ax, 0.76, 0.30, 0.10, 0.40,
                "Calibrated\nAlarm\n(self-scan FA\nbudget)",
                PAL["violet"], textcolor=PAL["text_light"],
                fontsize=7, bold=True)

    # ===== Stage 7: output =====
    rounded_box(ax, 0.90, 0.38, 0.09, 0.24,
                "Typed\nAlarms",
                PAL["pale_blue"], fontsize=8, bold=True)
    rounded_box(ax, 0.90, 0.10, 0.09, 0.16,
                "Edge attribution\n(BH-FDR)",
                PAL["pale_blue"], fontsize=6.5)

    # ===== arrows =====
    # input -> granulation
    arrow(ax, 0.13, 0.50, 0.17, 0.50)
    # granulation -> decision tables
    arrow(ax, 0.27, 0.50, 0.31, 0.50)
    # decision tables -> copula
    arrow(ax, 0.41, 0.60, 0.45, 0.62)
    # decision tables -> four statistics (direct)
    arrow(ax, 0.41, 0.40, 0.45, 0.35, color=PAL["lilac"])
    # copula -> statistics
    arrow(ax, 0.50, 0.55, 0.50, 0.50, color=PAL["mint"])
    # statistics -> R channel
    arrow(ax, 0.60, 0.55, 0.63, 0.60, color=PAL["violet"])
    # statistics -> E channel
    arrow(ax, 0.60, 0.30, 0.63, 0.30, color=PAL["orchid"])
    # R -> calibration
    arrow(ax, 0.72, 0.62, 0.76, 0.55, color=PAL["indigo"])
    # E -> calibration
    arrow(ax, 0.72, 0.32, 0.76, 0.40, color=PAL["orchid"])
    # calibration -> alarm
    arrow(ax, 0.86, 0.50, 0.90, 0.50)
    # calibration -> edge attribution
    arrow(ax, 0.86, 0.35, 0.90, 0.20, color=PAL["peri_blue"])

    # ===== annotations =====
    # centre labels
    ax.text(0.50, 0.85, "Four-tuple edge encoding", ha="center",
            fontsize=9, color=PAL["indigo"], weight="bold")
    ax.text(0.675, 0.85, "Fixed edge set", ha="center",
            fontsize=8, color=PAL["indigo"])

    plt.tight_layout(pad=0.3)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "..", "paper", "figs", "pipeline.pdf")
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.savefig(out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight",
                facecolor="white")
    print(f"saved {out}")

if __name__ == "__main__":
    main()
