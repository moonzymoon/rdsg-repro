# -*- coding: utf-8 -*-
"""Paper figure generation: E1 operating-point curves / E2 recall-by-type bar chart / E6 typing confusion matrix / E0 mechanism comparison."""
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

ROOT = _results_root()
RES = os.path.normpath(os.path.join(_results_root()))
FIG = os.path.join(RES, "paper", "figs")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({"font.size": 10, "figure.dpi": 150,
                     "axes.spines.top": False, "axes.spines.right": False})


def fig_e1_curves():
    d = json.load(open(os.path.join(RES, "paper", "e1_gas.json"), encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    styles = {"rdsg": ("#c0392b", "o", "RDSG (ours)"),
              "b4_if": ("#2c3e50", "s", "IF-agg"),
              "b4_lof": ("#7f8c8d", "^", "LOF-agg"),
              "b1_corr": ("#2980b9", "v", "Corr-matrix"),
              "b5b_pcorr": ("#8e44ad", "D", "Partial-corr"),
              "b2_granger": ("#16a085", "x", "Granger")}
    for m, (c, mk, lab) in styles.items():
        if m not in d["curves"]:
            continue
        pts = [p for p in d["curves"][m] if p["fa_per_day"] > 0 or p["recall"] > 0]
        if not pts:
            continue
        pts = sorted(pts, key=lambda p: p["fa_per_day"])
        ax.plot([p["fa_per_day"] for p in pts], [p["recall"] for p in pts],
                color=c, marker=mk, ms=4, lw=1.4, label=lab)
    ax.set_xlabel("False alarms per day (Gas Pipeline, 53 labelled attacks)")
    ax.set_ylabel("Event recall")
    ax.set_xlim(0, 80)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "e1_curves.pdf"))
    print("e1_curves saved")


def fig_e0_mechanism():
    d = json.load(open(os.path.join(RES, "e0", "e0_syn2_summary.json"), encoding="utf-8"))
    methods = ["rdsg", "gamma_only", "b4_if", "b4_lof", "b5a_mi"]
    labels = ["RDSG", "γ-only", "IF-agg", "LOF-agg", "MI-graph"]
    types = ["inv", "brk", "emb"]
    x = np.arange(len(methods))
    width = 0.25
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for i, t in enumerate(types):
        vals = [d["summary"][m]["recall_by_type"][t] for m in methods]
        ax.bar(x + (i - 1) * width, vals, width, label={"inv": "inversion", "brk": "breakdown",
                                                        "emb": "emergence"}[t])
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Event recall (synthetic gate)")
    ax.axhline(0.8, color="grey", ls=":", lw=0.8)
    ax.legend(fontsize=7.5, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "e0_mechanism.pdf"))
    print("e0_mechanism saved")


def fig_e2_by_type():
    d = json.load(open(os.path.join(RES, "paper", "e2_metr_la.json"), encoding="utf-8"))
    methods = ["rdsg", "gamma_only", "b4_if", "b4_lof", "b5a_mi", "b5b_pcorr",
               "b5c_var", "b1_corr", "b2_granger", "b3_gdn"]
    labels = ["RDSG", "γ-only", "IF", "LOF", "MI", "PCorr", "VAR", "Corr", "Granger", "GDN"]
    types = ["inv", "brk", "emb", "emb2"]
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    x = np.arange(len(methods))
    width = 0.2
    for i, t in enumerate(types):
        vals = []
        for m in methods:
            vals.append(np.mean([v["recall_by_type"].get(t, 0)
                                 for v in d.get(m, {}).values()
                                 if "recall_by_type" in v]) if d.get(m) else 0)
        ax.bar(x + (i - 1.5) * width, vals, width, label=t)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Event recall (METR-LA injection suite)")
    ax.legend(fontsize=8, frameon=False, ncol=4)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "e2_by_type.pdf"))
    print("e2_by_type saved")


def fig_e6_confusion():
    """Component-triggered typing confusion (consistent with the §5.3 body, data: e2d_attr.json).
    Redone 2026-10-11: the old version read event-aligned typing from the e2_metr_la detail,
    and the figure predated the data by one day; a review noted the text-figure mismatch --
    now redrawn from the current data under the component-triggered rule."""
    d = json.load(open(os.path.join(RES, "paper", "e2d_attr.json"), encoding="utf-8"))
    types = ["inv", "brk", "emb", "emb2"]
    preds = ["inv", "brk", "emb"]
    idx = {t: i for i, t in enumerate(types)}
    C = np.zeros((4, 3))
    for seed, v in d["metr_la"]["seeds"].items():
        for e in v["events"]:
            if e.get("pred") is None:
                continue
            C[idx[e["truth"]], preds.index(e["pred"])] += 1
    row = C.sum(axis=1, keepdims=True)
    Cn = np.divide(C, np.maximum(row, 1))
    fig, ax = plt.subplots(figsize=(3.2, 2.8))
    im = ax.imshow(Cn, cmap="Reds", vmin=0, vmax=1)
    ax.set_xticks(range(3)); ax.set_xticklabels(preds, fontsize=8)
    ax.set_yticks(range(4)); ax.set_yticklabels(types, fontsize=8)
    ax.set_xlabel("Predicted type"); ax.set_ylabel("Injected type")
    for i in range(4):
        for j in range(3):
            ax.text(j, i, f"{Cn[i, j]:.2f}", ha="center", va="center",
                    color="white" if Cn[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "e6_confusion.pdf"))
    print("e6_confusion saved (component-triggered)")


def fig_e6_component():
    """F1 (finalised by project lead 2026-10-16): ground-truth x fired-component matrix -- isomorphic to the Prop 1 component-level prediction.
    Rows=injected types (inv/brk/emb/emb2), columns=fired components (ddelta/dgamma) or channel (E).
    Data: e2d_component.json (same 63 events as e2d_attr.json; reconciliation in the audit chain)."""
    d = json.load(open(os.path.join(RES, "paper", "e2d_component.json"), encoding="utf-8"))
    types = ["inv", "brk", "emb", "emb2"]
    cols = ["ddelta", "dgamma", "E"]
    idx = {t: i for i, t in enumerate(types)}
    C = np.zeros((4, 3))
    for seed, v in d["seeds"].items():
        for e in v["events"]:
            if e["pred"] is None:
                continue
            if e["chan"] == "E":
                col = "E"
            else:
                col = "ddelta" if e["pred"] == "inv" else "dgamma"
            C[idx[e["truth"]], cols.index(col)] += 1
    row = C.sum(axis=1, keepdims=True)
    Cn = np.divide(C, np.maximum(row, 1))
    fig, ax = plt.subplots(figsize=(3.2, 2.8))
    im = ax.imshow(Cn, cmap="Reds", vmin=0, vmax=1)
    ax.set_xticks(range(3)); ax.set_xticklabels(cols, fontsize=8)
    ax.set_yticks(range(4)); ax.set_yticklabels(types, fontsize=8)
    ax.set_xlabel("Fired component / channel")
    ax.set_ylabel("Injected type")
    for i in range(4):
        for j in range(3):
            ax.text(j, i, f"{Cn[i, j]:.2f}", ha="center", va="center",
                    color="white" if Cn[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "e6_component.pdf"))
    print("e6_component saved; counts:", C.astype(int).tolist())


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "e1"):
        fig_e1_curves()
    if which in ("all", "e0"):
        fig_e0_mechanism()
    if which in ("all", "e2"):
        fig_e2_by_type()
    if which in ("all", "e6"):
        fig_e6_confusion()
    if which in ("e6c",):
        fig_e6_component()
