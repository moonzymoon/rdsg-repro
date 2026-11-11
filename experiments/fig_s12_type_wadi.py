# -*- coding: utf-8 -*-
"""S.12 图：(a) 三网络×四注入类型池化召回（family vs gamma-only，冻结 pooled_by_type.json
确定性聚合，不产生新数字）；(b) WADI 两点对比（max recall 与 @<=5FA/day 预算点，
从冻结 wadi_curves.json 采样网格读取）。

双重断言（失败即退出非零，图不生成）：
1. METR-LA rdsg 池化分类型值 == 正文冻结宏 \eTwoInv/Brk/Emb/EmbTwo（0.67/0.40/0.42/0.45）；
2. WADI 派生 max/budget 值 == 正文 Table tab:wadi 冻结值。
"""
import os, sys, json
import numpy as np

def _results_root():
    d = os.path.dirname(os.path.abspath(__file__))
    for base in (d, os.path.dirname(d), os.path.dirname(os.path.dirname(d))):
        if os.path.isdir(os.path.join(base, "results")):
            return os.path.join(base, "results")
    return os.path.normpath(os.path.join(d, "..", "..", "results"))

RES = _results_root()
POOL = json.load(open(os.path.join(RES, "paper", "pooled_by_type.json"), encoding="utf-8"))
WADI = json.load(open(os.path.join(RES, "paper", "wadi_curves.json"), encoding="utf-8"))

# ---- 断言 1：与正文冻结宏一致 ----
frozen = {("metr_la", "rdsg", "inv"): 0.67, ("metr_la", "rdsg", "brk"): 0.40,
          ("metr_la", "rdsg", "emb"): 0.42, ("metr_la", "rdsg", "emb2"): 0.45}
for (ds, m, t), v in frozen.items():
    got = round(POOL["e2"][ds][m][t]["pooled_recall"], 2)
    assert abs(got - v) < 5e-3, f"macro mismatch {ds}/{m}/{t}: {got} != {v}"
print("assert1 PASS: METR-LA pooled per-type == frozen macros")

# ---- 断言 2：WADI 派生值 == 正文 tab:wadi ----
FROZEN_WADI = {"rdsg": (0.857, 0.357), "b4_if": (0.857, 0.000), "b2_granger": (0.429, 0.000),
               "b5a_mi": (0.429, None), "b5b_pcorr": (0.286, None), "b5c_var": (0.357, None),
               "b4_lof": (0.000, 0.000)}
NAME = {"rdsg": "RDSG (ours)", "b4_if": "IF-agg", "b4_lof": "LOF-agg", "b5a_mi": "MI-graph",
        "b5b_pcorr": "Partial-corr", "b5c_var": "VAR-resid", "b2_granger": "Granger"}
wadi_rows = {}
for m, (fmax, fbud) in FROZEN_WADI.items():
    pts = WADI[m]
    mx = max(p["recall"] for p in pts)
    bud = [p["recall"] for p in pts if p["fa_per_day"] <= 5.0]
    bud = max(bud) if bud else None
    assert abs(round(mx, 3) - fmax) < 5e-3, f"max mismatch {m}: {mx} != {fmax}"
    if fbud is None:
        assert bud is None or bud <= 0, f"budget unexpectedly evaluable {m}"
        bud = None
    else:
        assert bud is not None and abs(round(bud, 3) - fbud) < 5e-3, f"budget mismatch {m}"
    wadi_rows[m] = (mx, bud)
print("assert2 PASS: WADI max/budget == frozen tab:wadi")

# ---- 追溯数值表（供 Supplemental 直接 \input） ----
gen_dir = os.path.join(RES, "..", "paper", "gen")
os.makedirs(gen_dir, exist_ok=True)
TYPES = ("inv", "brk", "emb", "emb2")
TYPE_NAME = {"inv": "inversion", "brk": "breakdown", "emb": "emergence",
             "emb2": "subset-joint"}
DS_NAME = {"metr_la": "METR-LA", "pems_bay": "PEMS-BAY", "beijing_air": "Beijing-Air"}
lines = []
for ds in ("metr_la", "pems_bay", "beijing_air"):
    for m, tag in (("rdsg", "family"), ("abl_gamma", "$\\gamma$-only")):
        vals = " & ".join(f"{POOL['e2'][ds][m][t]['pooled_recall']:.2f}" for t in TYPES)
        lines.append(f"{DS_NAME[ds]} & {tag} & {vals} \\\\")
with open(os.path.join(gen_dir, "s12_types_table.tex"), "w", encoding="utf-8",
          newline="\n") as f:
    f.write("\n".join(lines) + "\n" + "\\bottomrule\n")
print("wrote gen/s12_types_table.tex")

# ---- 绘图 ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2), width_ratios=[1.6, 1.0],
                               constrained_layout=True)
# (a) 12 槽 = 3 网络 × 4 类型，每槽一对条（family 深色 / gamma-only 浅色）
DS_ORDER = ("metr_la", "pems_bay", "beijing_air")
slots = np.arange(12)
for k, t in enumerate(TYPES):
    for j, (m, fc) in enumerate((("rdsg", "#c0392b"), ("abl_gamma", "#e6b0aa"))):
        vals = [POOL["e2"][ds][m][t]["pooled_recall"] for ds in DS_ORDER]
        pos = [slots[k + 4 * i] + (j - 0.5) * 0.38 for i in range(3)]
        ax1.bar(pos, vals, 0.34, color=fc,
                edgecolor=("#7b241c" if j == 0 else "#c0392b"), lw=0.4, zorder=3,
                label=("family ($\\gamma+\\delta$)" if (k == 0 and j == 0)
                       else ("$\\gamma$-only ablation" if (k == 0 and j == 1) else None)))
ax1.set_xticks(slots)
ax1.set_xticklabels(["inv", "brk", "emb", "sub"] * 3, fontsize=7)
for i, ds in enumerate(DS_ORDER):
    if i:
        ax1.axvline(4 * i - 0.5, color="0.75", lw=0.7, zorder=1)
    ax1.text(slots[4 * i: 4 * i + 4].mean(), 1.24, DS_NAME[ds], ha="center",
             fontsize=8, transform=ax1.get_xaxis_transform())
ax1.set_ylabel("Pooled event recall")
ax1.set_ylim(0, 1.34)
ax1.set_xlim(-0.7, 11.7)
ax1.legend(fontsize=7, loc="upper center", ncol=2, frameon=False, columnspacing=1.2, handlelength=1.4)
ax1.tick_params(labelsize=7)

# (b) WADI 两点对比：圆=max on grid；方=预算点；无网格点<=5FA 的方法只画圆并注明
methods = ["rdsg", "b4_if", "b2_granger", "b5a_mi", "b5c_var", "b5b_pcorr", "b4_lof"]
y = np.arange(len(methods))[::-1]
for i, m in enumerate(methods):
    mx, bud = wadi_rows[m]
    c = "#c0392b" if m == "rdsg" else "#34495e"
    if bud is not None:
        ax2.plot([bud, mx], [y[i], y[i]], color=c, lw=1.2, alpha=0.45, zorder=2)
        ax2.plot(bud, y[i], "s", ms=5, color=c, zorder=3)
    else:
        ax2.annotate("no grid point $\\leq$5 FA/day", xy=(min(mx + 0.03, 0.55), y[i]),
                     fontsize=6.2, color="#7f8c8d", va="center")
    ax2.plot(mx, y[i], "o", ms=5, color=c, zorder=3)
ax2.set_yticks(y)
ax2.set_yticklabels([NAME[m] for m in methods], fontsize=7.5)
ax2.set_xlim(-0.05, 1.02)
ax2.set_xlabel("Event recall (WADI, 14 attacks)")
ax2.plot([], [], "o", ms=5, color="#34495e", label="max on grid")
ax2.plot([], [], "s", ms=5, color="#34495e", label="@ $\\leq 5$ FA/day")
ax2.legend(fontsize=6.5, loc="lower left", frameon=False)
ax2.tick_params(labelsize=7.5)
out = os.path.join(RES, "paper", "figs", "s12_type_wadi")
fig.savefig(out + ".pdf", metadata={"Creator": "", "Producer": "", "CreationDate": None})
fig.savefig(out + ".png", dpi=200)
print("wrote", out + ".pdf/.png")
