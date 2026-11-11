# -*- coding: utf-8 -*-
"""Figure 1 v5: doctoral-dissertation pipeline style -- black/white outlines + red highlights + maths symbols + sub-labels."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os

# palette: academic black/white + a few accent colours
BLACK = "#1A1A1A"
DARK = "#2C3E50"
GRAY = "#6B7B8D"
LIGHT_FILL = "#F0F4F8"
WHITE = "#FFFFFF"
RED = "#C0392B"
BLUE = "#2980B9"
GREEN = "#27AE60"
ORANGE = "#E67E22"
PURPLE = "#8E44AD"
BORDER = "#333333"
LIGHT_BLUE = "#EBF0F7"
LIGHT_RED = "#FDF0EF"
LIGHT_GREEN = "#EDF7F0"
LIGHT_PURPLE = "#F5EEF8"
LIGHT_ORANGE = "#FDF6EC"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
})


def rbox(ax, x, y, w, h, fc=WHITE, ec=BORDER, lw=0.8, r=0.008, z=2, alpha=1.0):
    b = FancyBboxPatch((x, y), w, h,
                        boxstyle=f"round,pad={r}",
                        facecolor=fc, edgecolor=ec,
                        linewidth=lw, alpha=alpha, zorder=z)
    ax.add_patch(b)
    return b


def txt(ax, x, y, s, fs=7, c=BLACK, weight="normal", ha="center", va="center",
        z=5, style="normal", linespacing=1.3):
    ax.text(x, y, s, fontsize=fs, color=c, weight=weight,
            ha=ha, va=va, zorder=z, style=style, linespacing=linespacing)


def arr(ax, x1, y1, x2, y2, c=BLACK, lw=1.0, style="-|>", ms=10):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=c, lw=lw,
                                mutation_scale=ms, shrinkA=0, shrinkB=0),
                zorder=3)


def main():
    fig, ax = plt.subplots(figsize=(14, 6.5))
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # ============================================================
    # stage labels (a) (b)
    # ============================================================
    txt(ax, 0.50, 0.96, "(a) Offline: reference segment processing", fs=8.5,
        c=BLACK, weight="bold")
    txt(ax, 0.50, 0.50, "(b) Online: sliding-window detection", fs=8.5,
        c=BLACK, weight="bold")

    # ============================================================
    # (a) OFFLINE upper half
    # ============================================================
    # stage separator line
    ax.plot([0.00, 1.00], [0.88, 0.88], color="#CCCCCC", lw=0.5, ls="--", zorder=0)

    # --- 1. input data ---
    rbox(ax, 0.005, 0.68, 0.08, 0.18, LIGHT_BLUE, BORDER, lw=0.8)
    txt(ax, 0.045, 0.79, "Multivariate", fs=7)
    txt(ax, 0.045, 0.755, "Time Series", fs=7, weight="bold")
    txt(ax, 0.045, 0.72, "$X \\in \\mathbb{R}^{n \\times T}$", fs=6.5)

    # --- 2. granulation ---
    rbox(ax, 0.11, 0.68, 0.08, 0.18, LIGHT_BLUE, BORDER, lw=0.8)
    txt(ax, 0.15, 0.81, "Equal-freq", fs=7)
    txt(ax, 0.15, 0.775, "Granulation", fs=7, weight="bold")
    txt(ax, 0.15, 0.735, "$\\Phi^K$\n$K$ levels", fs=6.5)
    txt(ax, 0.15, 0.695, "frozen on ref.", fs=5.5, style="italic", c=GRAY)
    arr(ax, 0.085, 0.77, 0.11, 0.77)

    # --- 3. decision tables ---
    rbox(ax, 0.21, 0.68, 0.08, 0.18, LIGHT_BLUE, BORDER, lw=0.8)
    txt(ax, 0.25, 0.81, "Granular", fs=7)
    txt(ax, 0.25, 0.775, "Decision Tables", fs=7, weight="bold")
    txt(ax, 0.25, 0.73, "$T \\in \\mathbb{N}^{K^{|A|}\\times K}$", fs=6)
    txt(ax, 0.25, 0.695, "granule pair counts", fs=5.5, style="italic", c=GRAY)
    arr(ax, 0.19, 0.77, 0.21, 0.77)

    # --- 4. edge selection ---
    rbox(ax, 0.31, 0.68, 0.08, 0.18, LIGHT_BLUE, BORDER, lw=0.8)
    txt(ax, 0.35, 0.81, "Edge Selection", fs=7, weight="bold")
    txt(ax, 0.35, 0.765, "R: top-16 strong", fs=6)
    txt(ax, 0.35, 0.73, "E: bottom-24 weak", fs=6)
    txt(ax, 0.35, 0.695, "+ subset edges", fs=5.5, style="italic", c=GRAY)
    arr(ax, 0.29, 0.77, 0.31, 0.77)

    # --- 5. calibration ---
    rbox(ax, 0.41, 0.68, 0.08, 0.18, LIGHT_GREEN, BORDER, lw=0.8)
    txt(ax, 0.45, 0.81, "Calibration", fs=7, weight="bold")
    txt(ax, 0.45, 0.765, "$\\mu_e, \\sigma_e$: sliding", fs=6)
    txt(ax, 0.45, 0.73, "splits", fs=6)
    txt(ax, 0.45, 0.695, "$\\theta_c$: self-scan", fs=6)
    txt(ax, 0.45, 0.665, "FA budget", fs=5.5, style="italic", c=GRAY)
    arr(ax, 0.39, 0.77, 0.41, 0.77)

    # --- 6. four statistics ---
    rbox(ax, 0.51, 0.68, 0.08, 0.18, LIGHT_PURPLE, BORDER, lw=0.8)
    txt(ax, 0.55, 0.82, "Four-tuple", fs=7, weight="bold")
    txt(ax, 0.55, 0.785, "per edge:", fs=6)
    # four small labels
    for i, (sym, col) in enumerate([("$\\gamma$", BLUE), ("$\\rho$", PURPLE),
                                     ("$\\delta$", BLUE), ("$D_{JS}$", GREEN)]):
        yy = 0.755 - i * 0.025
        txt(ax, 0.53, yy, sym, fs=6.5, c=col, weight="bold")
        txt(ax, 0.555, yy, ["dep.", "rule", "dir.", "div."][i], fs=5, c=GRAY)
    arr(ax, 0.49, 0.77, 0.51, 0.77)

    # --- 7. Copula ---
    rbox(ax, 0.61, 0.68, 0.08, 0.18, LIGHT_GREEN, BORDER, lw=0.8)
    txt(ax, 0.65, 0.81, "Copula IPF", fs=7, weight="bold")
    txt(ax, 0.65, 0.765, "$\\alpha{=}8$ smooth", fs=6)
    txt(ax, 0.65, 0.73, "2 iters", fs=6)
    txt(ax, 0.65, 0.695, "uniform margins", fs=5.5, style="italic", c=GRAY)
    arr(ax, 0.59, 0.77, 0.61, 0.77)

    # --- 8. deployed artefacts ---
    rbox(ax, 0.71, 0.68, 0.08, 0.18, LIGHT_ORANGE, BORDER, lw=0.8)
    txt(ax, 0.75, 0.81, "Deployed", fs=7, weight="bold")
    txt(ax, 0.75, 0.775, "Artefacts", fs=7)
    txt(ax, 0.75, 0.735, "$\\Phi^K$ bounds", fs=5.5)
    txt(ax, 0.75, 0.71, "edge set", fs=5.5)
    txt(ax, 0.75, 0.685, "$\\mu_e, \\sigma_e, \\theta_c$", fs=5.5)
    arr(ax, 0.69, 0.77, 0.71, 0.77)

    # ============================================================
    # (b) ONLINE lower half
    # ============================================================
    txt(ax, 0.50, 0.62, "(b) Online: sliding-window detection and alarm", fs=8.5,
        c=BLACK, weight="bold")

    # --- online stage separator line ---
    ax.plot([0.00, 1.00], [0.58, 0.58], color="#CCCCCC", lw=0.5, ls="--", zorder=0)

    # --- window input ---
    rbox(ax, 0.005, 0.30, 0.08, 0.20, LIGHT_BLUE, BORDER, lw=0.8)
    txt(ax, 0.045, 0.44, "Sliding", fs=7)
    txt(ax, 0.045, 0.405, "Window", fs=7, weight="bold")
    txt(ax, 0.045, 0.37, "$w{=}128$\n$b{=}32$", fs=6)
    txt(ax, 0.045, 0.33, "new block\narrives", fs=5.5, style="italic", c=GRAY)

    # --- table update ---
    rbox(ax, 0.10, 0.30, 0.08, 0.20, LIGHT_BLUE, BORDER, lw=0.8)
    txt(ax, 0.14, 0.44, "Update", fs=7, weight="bold")
    txt(ax, 0.14, 0.405, "Tables", fs=7)
    txt(ax, 0.14, 0.37, "add new block\nremove old block\n(rolling buffer)", fs=5.5)
    arr(ax, 0.085, 0.40, 0.10, 0.40)

    # --- statistic computation ---
    rbox(ax, 0.20, 0.30, 0.08, 0.20, LIGHT_PURPLE, BORDER, lw=0.8)
    txt(ax, 0.24, 0.44, "Four-tuple", fs=7, weight="bold")
    txt(ax, 0.24, 0.405, "$\\Delta\\gamma$, $\\Delta\\delta$", fs=7)
    txt(ax, 0.24, 0.37, "z-standardise", fs=6)
    txt(ax, 0.24, 0.335, "family max", fs=6)
    arr(ax, 0.18, 0.40, 0.20, 0.40)

    # --- rolling evidence ---
    rbox(ax, 0.30, 0.30, 0.08, 0.20, LIGHT_BLUE, BORDER, lw=0.8)
    txt(ax, 0.34, 0.44, "Rolling", fs=7, weight="bold")
    txt(ax, 0.34, 0.405, "Evidence", fs=7)
    txt(ax, 0.34, 0.37, "$S_e$ = mean of\n$k{=}4$ boundaries", fs=5.5)
    arr(ax, 0.28, 0.40, 0.30, 0.40)

    # --- alarm decision ---
    rbox(ax, 0.40, 0.30, 0.08, 0.20, LIGHT_RED, "#C0392B", lw=1.0)
    txt(ax, 0.44, 0.44, "Alarm", fs=7, weight="bold", c="#C0392B")
    txt(ax, 0.44, 0.405, "if $S_e > \\theta_c$", fs=6.5)
    txt(ax, 0.44, 0.37, "merge events\n(gap $\\leq w$)", fs=5.5)
    arr(ax, 0.38, 0.40, 0.40, 0.40, c="#C0392B")

    # --- attribution ---
    rbox(ax, 0.50, 0.30, 0.08, 0.20, LIGHT_PURPLE, BORDER, lw=0.8)
    txt(ax, 0.54, 0.44, "Attribution", fs=7, weight="bold")
    txt(ax, 0.54, 0.405, "BH-FDR", fs=6.5)
    txt(ax, 0.54, 0.37, "block perm.\nnull", fs=5.5)
    arr(ax, 0.48, 0.40, 0.50, 0.40)

    # --- typing ---
    rbox(ax, 0.60, 0.30, 0.08, 0.20, LIGHT_ORANGE, "#E67E22", lw=0.8)
    txt(ax, 0.64, 0.44, "Type", fs=7, weight="bold", c="#E67E22")
    txt(ax, 0.64, 0.405, "brk / inv / emb", fs=6)
    txt(ax, 0.64, 0.37, "component-\ntriggered rule", fs=5.5)
    arr(ax, 0.58, 0.40, 0.60, 0.40, c="#E67E22")

    # --- edge attribution ---
    rbox(ax, 0.70, 0.30, 0.08, 0.20, LIGHT_BLUE, BORDER, lw=0.8)
    txt(ax, 0.74, 0.44, "Edge", fs=7, weight="bold")
    txt(ax, 0.74, 0.405, "Attribution", fs=7)
    txt(ax, 0.74, 0.37, "BH-FDR on\nblock perm.", fs=5.5)
    arr(ax, 0.68, 0.40, 0.70, 0.40)

    # --- output ---
    rbox(ax, 0.80, 0.30, 0.08, 0.20, LIGHT_GREEN, "#27AE60", lw=1.0)
    txt(ax, 0.84, 0.44, "Typed", fs=7, weight="bold", c="#27AE60")
    txt(ax, 0.84, 0.405, "Alarm", fs=7, weight="bold", c="#27AE60")
    txt(ax, 0.84, 0.37, "+ edge", fs=6)
    txt(ax, 0.84, 0.34, "attribution", fs=6)
    arr(ax, 0.78, 0.40, 0.80, 0.40, c="#27AE60")

    # --- Raspberry Pi annotation ---
    rbox(ax, 0.90, 0.30, 0.08, 0.20, LIGHT_ORANGE, "#E67E22", lw=0.8)
    txt(ax, 0.94, 0.44, "Edge HW", fs=7, weight="bold", c="#E67E22")
    txt(ax, 0.94, 0.405, "Pi 4B:\n459 ms/win\n2.2 win/s", fs=6)
    txt(ax, 0.94, 0.34, "RSS 34 MB", fs=5.5)
    txt(ax, 0.94, 0.315, "10.7 h margin", fs=5.5)

    # ============================================================
    # cross arrows (offline -> online)
    # ============================================================
    # decision tables -> table update (vertical arrow)
    arr(ax, 0.25, 0.68, 0.14, 0.50, GRAY, lw=0.8, style="->")
    txt(ax, 0.25, 0.66, "deploy $\\Phi^K$ bounds,\nedge set", fs=5.5, c=GRAY,
        style="italic", ha="left")

    # calibration -> alarm decision (vertical arrow)
    arr(ax, 0.45, 0.68, 0.44, 0.50, GRAY, lw=0.8, style="->")
    txt(ax, 0.46, 0.65, "deploy $\\mu_e, \\sigma_e, \\theta_c$", fs=5.5, c=GRAY,
        style="italic", ha="left")

    # ============================================================
    # right-side annotation boxes
    # ============================================================
    # detection family annotation box
    rbox(ax, 0.60, 0.02, 0.14, 0.06, LIGHT_PURPLE, PURPLE, lw=0.8)
    txt(ax, 0.67, 0.05, "Detection family: $\\Delta\\gamma + \\Delta\\delta$\n"
        "(D_JS = attribution, rho = not in family)", fs=6,
        c=PURPLE, weight="bold")

    # key result annotation box
    rbox(ax, 0.77, 0.02, 0.14, 0.06, LIGHT_GREEN, "#27AE60", lw=0.8)
    txt(ax, 0.84, 0.05, "Edge margin: 10.7 h arrival\nvs. 459 ms update = 4 orders",
        fs=6, c="#27AE60")

    # ============================================================
    # connector lines (between components)
    # ============================================================
    # window -> table update (already drawn)
    # table update -> statistic computation
    arr(ax, 0.18, 0.40, 0.20, 0.40)
    # statistics -> rolling evidence
    arr(ax, 0.28, 0.40, 0.30, 0.40)
    # rolling evidence -> alarm
    arr(ax, 0.38, 0.40, 0.40, 0.40, c="#C0392B")
    # alarm -> attribution
    arr(ax, 0.48, 0.40, 0.50, 0.40)
    # attribution -> typing
    arr(ax, 0.58, 0.40, 0.60, 0.40, c="#E67E22")
    # typing -> edge attribution
    arr(ax, 0.68, 0.40, 0.70, 0.40)
    # edge attribution -> output
    arr(ax, 0.78, 0.40, 0.80, 0.40, c="#27AE60")

    # ============================================================
    # bottom formula annotation
    # ============================================================
    txt(ax, 0.50, 0.06,
        "$\\gamma(\\pi) = \\sum_g \\pi_g \\max_v \\pi_{v|g}$   "
        "$\\rho = \\sum_g \\pi^{ref}_g \\mathbf{1}[\\phi_r(g){=}\\phi_c(g)]$   "
        "$\\delta = \\mathrm{corr}(g, v; \\pi)$   "
        "$D_{JS} = \\sum_g \\pi^{ref}_g \\mathrm{JSD}(\\pi^{ref}_{\\cdot|g}, \\pi^{cur}_{\\cdot|g})$",
        fs=7, c=DARK, ha="center")

    plt.tight_layout(pad=0.4)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "..", "paper", "figs", "pipeline.pdf")
    plt.savefig(out, dpi=250, bbox_inches="tight", facecolor="white")
    plt.savefig(out.replace(".pdf", ".png"), dpi=250, bbox_inches="tight",
                facecolor="white")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
