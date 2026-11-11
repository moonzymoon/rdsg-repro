# -*- coding: utf-8 -*-
"""S.14（长臂对比）与 S.15（规模标度）表的程序化生成——数字只从冻结/新跑 JSON 来。"""
import os, sys, json
import numpy as np

def _results_root():
    d = os.path.dirname(os.path.abspath(__file__))
    for base in (d, os.path.dirname(d), os.path.dirname(os.path.dirname(d))):
        if os.path.isdir(os.path.join(base, "results")):
            return os.path.join(base, "results")
    return os.path.normpath(os.path.join(d, "..", "..", "results"))

RES = _results_root()
GEN = os.path.join(RES, "..", "paper", "gen")
DS = [("metr_la", "METR-LA"), ("pems_bay", "PEMS-BAY"), ("beijing_air", "Beijing-Air")]

# ---- S.14：compact vs long × AT/TranAD，RDSG 作参照 ----
long_ = json.load(open(os.path.join(RES, "paper", "abl_long.json"), encoding="utf-8"))
comp_at = json.load(open(os.path.join(RES, "paper", "abl_at.json"), encoding="utf-8"))
comp_td = json.load(open(os.path.join(RES, "paper", "abl_tranad.json"), encoding="utf-8"))
rdsg = {ds: np.mean([v["recall"] for v in json.load(
    open(os.path.join(RES, "paper", f"e2_{ds}.json"), encoding="utf-8"))["rdsg"].values()])
    for ds, _ in DS}

BS = chr(92)
rows = []
for ds, name in DS:
    at_c = np.mean([v["recall"] for v in comp_at[ds].values()])
    at_l = np.mean([v["recall"] for v in long_[ds]["at_long"].values()])
    td_c = np.mean([v["recall"] for v in comp_td[ds].values()])
    td_l = np.mean([v["recall"] for v in long_[ds]["tranad_long"].values()])
    rows.append(f"{name} & AT & {at_c:.2f} & {at_l:.2f} & {rdsg[ds]:.2f} {BS}{BS}")
    rows.append(f" & TranAD & {td_c:.2f} & {td_l:.2f} &  {BS}{BS}")
    deltas = f"% {name}: AT {at_c:.3f}->{at_l:.3f}, TranAD {td_c:.3f}->{td_l:.3f}"
    print(deltas)
with open(os.path.join(GEN, "s14_table.tex"), "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(rows) + "\n" + BS + "bottomrule\n")
print("wrote gen/s14_table.tex")

# ---- S.15：规模标度 ----
sc = json.load(open(os.path.join(RES, "paper", "scaling.json"), encoding="utf-8"))
LABEL = {"METR-LA(207)": "METR-LA (real)", "PEMS-BAY(325)": "PEMS-BAY (real)",
         "synth(500)": "synthetic", "synth(1000)": "synthetic", "synth(2000)": "synthetic"}
rows = []
for r in sc["rows"]:
    lab = f"{LABEL[r['label']]}" if "synth" not in r["label"] else "synthetic"
    rows.append(f"{r['n']} & {lab} & {r['t_pair_rank_s']:.1f} & {r['t_fit_s']:.1f} & "
                f"{r['t_boundary_ms']:.1f} & {r['rss_peak_mb']:.0f} {BS}{BS}")
with open(os.path.join(GEN, "s15_table.tex"), "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(rows) + "\n" + BS + "bottomrule\n")
print("wrote gen/s15_table.tex")
