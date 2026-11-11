# -*- coding: utf-8 -*-
"""从实验 JSON 生成论文全部 LaTeX 表格与关键数字宏（数字单一来源）。

输出 paper/gen/*.tex；main.tex \\input 之。改实验 → 重跑本脚本即同步。
"""
import json
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.normpath(os.path.join(HERE, "..", "results"))
GEN = os.path.join(HERE, "gen")
os.makedirs(GEN, exist_ok=True)


def load(path):
    p = os.path.join(RES, path)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def fmt(x, nd=2):
    return f"{x:.{nd}f}"


def w(name, content):
    with open(os.path.join(GEN, name), "w", encoding="utf-8") as f:
        f.write(content)
    print("wrote", name)


def e0_macros():
    d = load("e0/e0_syn2_summary.json")
    if not d:
        return
    s = d["summary"]
    # by-type 数字用池化逐事件口径（pooled_by_type.json），与表注声明一致
    pool = load("paper/pooled_by_type.json")
    pr = pool["e0"] if pool else {}
    def _pooled(m, t, fallback):
        try:
            v = pr[m][t]["pooled_recall"]
            return fmt(v) if v is not None else fmt(fallback)
        except (KeyError, TypeError):
            return fmt(fallback)
    defs = {
        "eZeroRdsRecall": fmt(s["rdsg"]["recall_mean"]),
        "eZeroRdsInv": _pooled("rdsg", "inv", s["rdsg"]["recall_by_type"]["inv"]),
        "eZeroRdsBrk": _pooled("rdsg", "brk", s["rdsg"]["recall_by_type"]["brk"]),
        "eZeroRdsEmb": _pooled("rdsg", "emb", s["rdsg"]["recall_by_type"]["emb"]),
        "eZeroRdsFa": fmt(s["rdsg"]["fa_per_day_mean"]),
        "eZeroGammaRecall": fmt(s["gamma_only"]["recall_mean"]),
        "eZeroGammaInv": _pooled("gamma_only", "inv", s["gamma_only"]["recall_by_type"]["inv"]),
        "eZeroGammaBrk": _pooled("gamma_only", "brk", s["gamma_only"]["recall_by_type"]["brk"]),
        "eZeroGammaEmb": _pooled("gamma_only", "emb", s["gamma_only"]["recall_by_type"]["emb"]),
        "eZeroIfInv": fmt(s["b4_if"]["recall_by_type"].get("inv", 0)),
        "eZeroIfBrk": fmt(s["b4_if"]["recall_by_type"].get("brk", 0)),
        "eZeroIfEmb": fmt(s["b4_if"]["recall_by_type"].get("emb", 0)),
        "eZeroIfRecall": fmt(s["b4_if"]["recall_mean"]),
        "eZeroPval": f"{d['verdict']['wilcoxon_p']:.4g}",
    }
    raw = load("e0/e0_syn2_raw.json")
    if raw and "rdsg" in raw:
        defs["eZeroNSeeds"] = str(len(raw["rdsg"]))
    w("e0_macros.tex", "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in defs.items()))


def e1_macros():
    d = load("paper/e1_gas.json")
    if not d:
        return
    def at_fa(curve, target):
        best = min(curve, key=lambda p: abs(p["fa_per_day"] - target))
        return best
    ours10 = at_fa(d["curves"]["rdsg"], 10)
    if10 = at_fa(d["curves"]["b4_if"], 10)
    lof10 = at_fa(d["curves"]["b4_lof"], 5)
    defs = {
        "eOneEvents": str(d["n_events"]),
        "eOneOursRecallFaTen": fmt(ours10["recall"]),
        "eOneOursFaTen": fmt(ours10["fa_per_day"], 1),
        "eOneIfRecallFaTen": fmt(if10["recall"]),
        "eOneLofRecallLowFa": fmt(lof10["recall"]),
        "eOneOursMaxRecall": fmt(max(p["recall"] for p in d["curves"]["rdsg"])),
    }
    w("e1_macros.tex", "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in defs.items()))


def e2_table():
    """跨数据集 × 方法 × 类型 主表（表体）。"""
    rows = []
    ds = [("metr_la", "METR-LA"), ("beijing_air", "Beijing-Air"), ("pems_bay", "PEMS-BAY")]
    methods = [("rdsg", "RDSG (ours)"), ("abl_gamma", "$\gamma$-only"),
               ("b4_if", "IF-agg"), ("b4_lof", "LOF-agg"), ("b3_gdn", "GDN"),
               ("b1_corr", "Corr-matrix"), ("b2_granger", "Granger"),
               ("b5a_mi", "MI-graph"), ("b5b_pcorr", "Partially-corr"), ("b5c_var", "VAR-resid"),
               ("w6_rough", "Rough-NN"),
               ("abl_at", "Anomaly Trans."), ("abl_tranad", "TranAD")]
    data = {k: load(f"paper/e2_{k}.json") for k, _ in ds}
    w6 = load("paper/w6_rough_baseline.json")
    if w6:
        for ds_key in ("metr_la", "beijing_air", "pems_bay"):
            data[ds_key]["w6_rough"] = dict(w6[ds_key]["seeds"])
    # compact deep scorers: per-dataset {seed: metrics} in their own files
    for src, key in (("paper/abl_at.json", "abl_at"), ("paper/abl_tranad.json", "abl_tranad")):
        d = load(src)
        if d:
            for ds_key in ("metr_la", "beijing_air", "pems_bay"):
                data[ds_key][key] = d.get(ds_key, {})
    body = []
    for mk, mlab in methods:
        cells = []
        recalls_all = []
        for k, _ in ds:
            d = data[k]
            seeds = (d or {}).get(mk) or {}
            if not seeds and mk == "gamma_only":
                seeds = (d or {}).get("abl_gamma") or {}
            r = [v["recall"] for v in seeds.values() if "recall" in v]
            f = [v["fa_per_day"] for v in seeds.values() if v.get("fa_per_day") is not None]
            if not r:
                cells.append("--"); continue
            cells.append(f"{np.mean(r):.2f}±{np.std(r):.2f}")
            if f:
                cells.append(f"{np.mean(f):.2f}")
        if all(c == "--" for c in cells):
            continue
        body.append(f"{mlab} & " + " & ".join(cells) + r" \\")
    body.append("\hline")
    w("tab_e2_body.tex", "\n".join(body))


def e2_ours_types():
    pool = load("paper/pooled_by_type.json")
    if not pool:
        return
    src = pool["e2"]["metr_la"]["rdsg"]
    tag = {"inv": "Inv", "brk": "Brk", "emb": "Emb", "emb2": "EmbTwo"}
    defs = {}
    for t, tg in tag.items():
        v = src.get(t, {}).get("pooled_recall")
        if v is not None:
            defs[f"eTwo{tg}"] = fmt(v)
    NL = chr(10)
    BS = chr(92)
    body = NL.join(f"{BS}newcommand{{{BS}{k}}}{{{v}}}" for k, v in defs.items())
    w("e2_macros.tex", body)


def e3_macros():
    d = load("paper/e2_metr_la.json")
    if not d:
        return
    def mean_of(m):
        r = [v["recall"] for v in d.get(m, {}).values() if "recall" in v]
        return fmt(np.mean(r)) if r else "--"
    defs = {
        "eThreeFull": mean_of("rdsg"),
        "eThreeGamma": mean_of("abl_gamma"),
        "eThreeGammaDelta": mean_of("abl_gd"),
        "eThreeGammaJs": mean_of("abl_gj"),
    }
    w("e3_macros.tex", "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in defs.items()))


def runtime_macros():
    d = load("paper/e1_gas.json")
    if not d:
        return
    keys = {"rdsg_runtime": "eSevenOurs", "b1_corr_runtime": "eSevenCorr",
            "b5c_var_runtime": "eSevenVar", "b2_granger_runtime": "eSevenGranger",
            "b4_if_runtime": "eSevenIf"}
    defs = {}
    for k, v in keys.items():
        if k in d:
            defs[v] = fmt(d[k], 1)
    w("e7_macros.tex", "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in defs.items()))


def e2_nseeds():
    """表 caption 用的种子数宏（从 JSON 长度读取，杜绝手写种子数）。"""
    d = load("paper/e2_metr_la.json")
    if not d:
        return
    n = len(d.get("rdsg", {}))
    w("e2n_macros.tex", "\\newcommand{\\eTwoNSeeds}{%d}" % n)


def e2_signif_macros():
    """E2 显著性检验 p 值宏（来自 e2_significance.json）。"""
    d = load("paper/e2_significance.json")
    if not d:
        return
    def pf(x):
        return "--" if x is None else (f"{x:.3f}" if x >= 0.001 else f"{x:.1e}")
    key = {"metr_la": "Metr", "beijing_air": "Bj", "pems_bay": "Bay"}
    defs = {}
    for ds, tag in key.items():
        r = d.get(ds)
        if not r:
            continue
        defs[f"eTwoPgOnly{tag}"] = pf(r.get("p_vs_gamma_only"))
        defs[f"eTwoPStrong{tag}"] = pf(r.get("p_vs_strongest"))
    w("e2s_macros.tex", "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in defs.items()))


def e0b_macros():
    """E0b reshape 实验宏（full vs γ+δ）。"""
    d = load("e0/e0b_reshape_summary.json")
    if not d:
        return
    s = d["summary"]
    p = s.get("wilcoxon_p_greater")
    defs = {
        "eZeroBFull": fmt(s["full_recall"]),
        "eZeroBGd": fmt(s["gamma_delta_recall"]),
        "eZeroBP": "--" if p is None else (f"{p:.3f}" if p >= 0.001 else f"{p:.1e}"),
        "eZeroBN": str(s.get("n_seeds", "")),
    }
    w("e0b_macros.tex", "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in defs.items()))


def e1c_macros():
    """E1c 点足迹分解宏。"""
    d = load("paper/e1c_gas_subset.json")
    if not d:
        return
    defs = {
        "eOneCRel": str(d["n_relational_only"]),
        "eOneCVis": str(d["n_point_visible"]),
        "eOneCTotal": str(d["n_events"]),
    }
    w("e1c_macros.tex", "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in defs.items()))


def e2c_table():
    """E2c 宿主线性度分组表体（池化逐事件口径，来自 e2c_linearity.json）。"""
    d = load("paper/e2c_linearity.json")
    if not d:
        return
    ms = [("rdsg", "RDSG (ours)"), ("b5b_pcorr", "Partially-corr"),
          ("b1_corr", "Corr-matrix"), ("b4_lof", "LOF-agg")]
    ds = [("metr_la", "METR-LA"), ("beijing_air", "Beijing-Air"), ("pems_bay", "PEMS-BAY")]
    groups = [("linear", "high-assoc. host"), ("nonlinear", "lower-assoc. host"), ("subset", "subset host")]
    body = []
    for k, nice in ds:
        first = True
        for g, glab in groups:
            cells = []
            for mk, _ in ms:
                cell = (d[k]["by_group"].get(mk, {}).get(g) or {})
                r = cell.get("recall")
                cells.append("--" if r is None else f"{r:.2f}")
            row = (nice if first else "") + " & " + glab + " & " + " & ".join(cells) + r" \\"
            body.append(row)
            first = False
    body.append(r"\hline")
    w("tab_e2c_body.tex", "\n".join(body))


def e2c_macros():
    """E2c 解读段用的关键数字宏。"""
    d = load("paper/e2c_linearity.json")
    if not d:
        return
    def g(k, m, grp):
        c = d[k]["by_group"].get(m, {}).get(grp) or {}
        r = c.get("recall")
        return "--" if r is None else f"{r:.2f}"
    defs = {
        "eTwoCcBjLinRdsg": g("beijing_air", "rdsg", "linear"),
        "eTwoCcBjNonRdsg": g("beijing_air", "rdsg", "nonlinear"),
        "eTwoCcBjNonPcorr": g("beijing_air", "b5b_pcorr", "nonlinear"),
        "eTwoCcBjSubRdsg": g("beijing_air", "rdsg", "subset"),
        "eTwoCcMetrLinRdsg": g("metr_la", "rdsg", "linear"),
        "eTwoCcMetrNonRdsg": g("metr_la", "rdsg", "nonlinear"),
        "eTwoCcMetrNonPcorr": g("metr_la", "b5b_pcorr", "nonlinear"),
        "eTwoCcBayLinLof": g("pems_bay", "b4_lof", "linear"),
        "eTwoCcBayNonLof": g("pems_bay", "b4_lof", "nonlinear"),
    }
    w("e2c_macros.tex", "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in defs.items()))


def e2len_macros():
    """RDSG 按事件长度的池化 recall 宏（来自 detail，逐事件口径）。"""
    d = load("paper/pooled_by_type.json")
    det_src = os.path.normpath(os.path.join(RES, "paper", "e2_metr_la.json"))
    if not os.path.exists(det_src):
        return
    data = json.load(open(det_src, encoding="utf-8"))
    agg = {}
    for s, v in data["rdsg"].items():
        for st, L, t, dl, hit in v.get("detail", []):
            agg.setdefault(int(L), [0, 0])
            agg[int(L)][1] += 1
            agg[int(L)][0] += 1 if hit else 0
    defs = {}
    lens = sorted(agg)
    for tag, L in zip("ABC", lens):
        h, n = agg[L]
        defs[f"eTwoLen{tag}Val"] = f"{h/n:.2f}"
        defs[f"eTwoLen{tag}Pts"] = str(L)
    w("e2len_macros.tex", "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in defs.items()))


def e5_macros():
    """E5 敏感性 5-seed 口径宏。"""
    d = load("paper/e5_sensitivity5_summary.json")
    if not d:
        return
    recs = {k: v["recall"] for k, v in d.items()}
    defs = {
        "eFiveLo": f"{min(recs.values()):.2f}",
        "eFiveHi": f"{max(recs.values()):.2f}",
        "eFiveDefault": f"{recs['4_128']:.2f}",
        "eFiveGap": f"{max(recs.values()) - recs['4_128']:.2f}",
        "eFiveN": str(d["4_128"]["n"]),
    }
    w("e5_macros.tex", "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in defs.items()))


def e2d_macros():
    """W2+W4 分量归因与分量触发判型宏。"""
    d = load("paper/e2d_attr.json")
    if not d:
        return
    m = d["metr_la"]; b = d["beijing_air"]
    defs = {
        "eTwoDMetrAcc": f"{m['type_acc_component']:.2f}",
        "eTwoDMetrInv": f"{m['by_truth']['inv']['acc']:.2f}",
        "eTwoDMetrInvN": str(m['by_truth']['inv']['n']),
        "eTwoDBjAcc": f"{b['type_acc_component']:.2f}",
        "eTwoDBjEmbTwo": f"{b['by_truth']['emb2']['acc']:.2f}",
    }
    w("e2d_macros.tex", "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in defs.items()))



def w3w5_macros():
    """W3a 效应量功效 / W3b 并集部署 / W5 滚动原型 宏。"""
    defs = {}
    w3b = load("paper/w3b_union.json")
    if w3b:
        defs["eWThreeBUnion"] = f"{w3b['unions']['rdsg+b5b_pcorr+b4_lof']['recall']:.2f}"
        defs["eWThreeBLof"] = f"{w3b['singles']['b4_lof']['recall']:.2f}"
    w5 = load("paper/e1d_rolling.json")
    if w5:
        s = w5["summary"]
        defs["eWFiveStatic"] = f"{s['static']['recall_at_fa5']:.2f}"
        defs["eWFiveRoll"] = f"{s['rolling']['recall_at_fa5']:.2f}"
    w3a = load("e0/w3a_power_curve.json")
    if w3a and "summary" in w3a:
        c = w3a["summary"]
        defs["eWThreeAHi"] = f"{c['p0.65_L120']['recall']:.2f}"
        defs["eWThreeAHiB"] = f"{c['p0.8_L120']['recall']:.2f}"
        defs["eWThreeALo"] = f"{c['p0.2_L30']['recall']:.2f}"
    NL = chr(10)
    BSC = chr(92)
    body = NL.join(f"{BSC}newcommand{{{BSC}{k}}}{{{v}}}" for k, v in defs.items())
    w("w3w5_macros.tex", body)


def e2_length_table():
    """三网事件长度-recall 对照表体（池化，rdsg detail 单一来源）。"""
    import collections
    rows_out = []
    for key, nice in (("metr_la", "METR-LA"), ("beijing_air", "Beijing-Air"),
                      ("pems_bay", "PEMS-BAY")):
        d = load(f"paper/e2_{key}.json")
        if not d:
            continue
        agg = collections.defaultdict(lambda: [0, 0])
        for s, v in d["rdsg"].items():
            for st, L, t, dl, hit in v.get("detail", []):
                agg[int(L)][1] += 1
                agg[int(L)][0] += 1 if hit else 0
        lens = sorted(agg)
        ls = "/".join(str(L) for L in lens)
        rs = "/".join(f"{agg[L][0]/agg[L][1]:.2f}" for L in lens)
        rows_out.append(nice + " & " + ls + " & " + rs + r" \\")
    rows_out.append(r"\hline")
    NLc = chr(10)
    w("tab_len_body.tex", NLc.join(rows_out))


if __name__ == "__main__":
    e0_macros()
    e1_macros()
    e2_table()
    e2_ours_types()
    e3_macros()
    runtime_macros()
    e2_nseeds()
    e2_signif_macros()
    e0b_macros()
    e1c_macros()
    e2c_table()
    e2c_macros()
    e2len_macros()
    e5_macros()
    w3w5_macros()
    e2d_macros()
    e2_length_table()
