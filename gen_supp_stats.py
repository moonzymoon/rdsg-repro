# gen_supp_stats.py --- generate Supplemental statistics tables from
# frozen result JSONs (no new experiments; derived summaries only).
# Outputs: gen/supp_seed_table.tex, gen/supp_ci_table.tex
import json
import os
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
E2 = {
    "METR-LA": os.path.join(_results_root(), "paper", "e2_metr_la.json"),
    "Beijing-Air": os.path.join(_results_root(), "paper", "e2_beijing_air.json"),
    "PEMS-BAY": os.path.join(_results_root(), "paper", "e2_pems_bay.json"),
}
EXT = os.path.join(_results_root(), "paper", "ext_metrics.json")
_here = os.path.dirname(os.path.abspath(__file__))
OUTDIR = (os.path.join(_here, "gen") if os.path.isdir(os.path.join(_here, "gen"))
          else os.path.join(_here, "paper", "gen"))


def per_seed(d, method):
    seeds = d[method]
    rec, fa = [], []
    for s in sorted(seeds, key=int):
        v = seeds[s]
        rec.append(v.get("recall", v.get("event_recall")))
        fa.append(v.get("fa_per_day", v.get("fa", v.get("false_alarms_per_day"))))
    return rec, fa


def signed_rank_biserial(x, y):
    """Matched-pairs rank-biserial correlation for Wilcoxon signed-rank:
    r = sum(sign(d_i) * rank(|d_i|)) / (n(n+1)/2), zeros dropped."""
    diffs = [(a - b) for a, b in zip(x, y) if (a - b) != 0]
    n = len(diffs)
    if n == 0:
        return None
    order = sorted(range(n), key=lambda i: abs(diffs[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(diffs[order[j + 1]]) == abs(diffs[order[i]]):
            j += 1
        avg = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    num = sum((1 if d > 0 else -1) * r for d, r in zip(diffs, ranks))
    return num / (n * (n + 1) / 2.0)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    # --- per-seed table (RDSG vs gamma-only) + effect sizes
    rows = []
    for ds, path in E2.items():
        d = json.load(open(path, encoding="utf-8"))
        rec, fa = per_seed(d, "rdsg")
        grec, gfa = per_seed(d, "abl_gamma")
        r = signed_rank_biserial(rec, grec)
        for i in range(len(rec)):
            rows.append(
                f"{ds if i == 0 else ''} & {i+1} & {rec[i]:.2f} & {fa[i]:.2f} & "
                f"{grec[i]:.2f} & {gfa[i]:.2f} \\\\"
            )
        rows.append(
            f" & mean & {sum(rec)/len(rec):.2f} & {sum(fa)/len(fa):.2f} & "
            f"{sum(grec)/len(grec):.2f} & {sum(gfa)/len(gfa):.2f} \\\\"
        )
        if r is None:
            eff = "effect size unavailable"
        else:
            eff = f"{r:.2f}"
        rows.append(
            f" & \\multicolumn{{5}}{{l}}{{rank-biserial $r$ (RDSG vs $\\gamma$-only): "
            f"{eff}}} \\\\"
        )
        rows.append("\\hline")
    open(os.path.join(OUTDIR, "supp_seed_table.tex"), "w", encoding="utf-8").write(
        "\n".join(rows)
    )

    # --- CI table from ext_metrics (bootstrap recall CIs, FA means)
    ext = json.load(open(EXT, encoding="utf-8"))
    label = {
        "rdsg": "RDSG", "abl_gamma": "$\\gamma$-only", "b4_if": "IF-agg",
        "b4_lof": "LOF-agg", "b5b_pcorr": "partial corr", "b1_corr": "corr-matrix",
    }
    rows = []
    for ds, methods in ext.items():
        if not isinstance(methods, dict):
            continue
        name = {"metr_la": "METR-LA", "beijing_air": "Beijing-Air",
                "pems_bay": "PEMS-BAY"}.get(ds, ds)
        first = True
        for m, v in methods.items():
            if m not in label or not isinstance(v, dict):
                continue
            ci = v["recall_ci"]
            if isinstance(ci, str):
                ci = json.loads(ci)
            lo, hi = ci
            rows.append(
                f"{name if first else ''} & {label[m]} & {v['recall_mean']:.2f} & "
                f"[{lo:.2f}, {hi:.2f}] & {v['fa_mean']:.2f} \\\\"
            )
            first = False
        rows.append("\\hline")
    open(os.path.join(OUTDIR, "supp_ci_table.tex"), "w", encoding="utf-8").write(
        "\n".join(rows)
    )
    print("wrote supp_seed_table.tex and supp_ci_table.tex")




def paired_ci():
    """Paired per-seed recall differences (RDSG - strongest external
    baseline) with bootstrap CIs; writes supp_paired_table.tex.

    The strongest external baseline is derived from the data (mean per-seed
    recall over the external baselines), consistent with
    experiments/make_derived_stats.py and e2_significance.json.
    """
    import random
    EXTERNAL = ("b4_if", "b4_lof", "b5a_mi", "b5b_pcorr", "b5c_var",
                "b1_corr", "b2_granger", "b3_gdn", "abl_at", "abl_tranad")
    LABEL = {"b4_if": "IF-agg", "b4_lof": "LOF-agg", "b5a_mi": "MI-graph",
             "b5b_pcorr": "partial corr", "b5c_var": "VAR-resid",
             "b1_corr": "corr-matrix", "b2_granger": "Granger", "b3_gdn": "GDN",
             "abl_at": "Anomaly Trans.", "abl_tranad": "TranAD"}
    rows = []
    for ds, path in E2.items():
        d = json.load(open(path, encoding="utf-8"))
        cfg = {"METR-LA": "metr_la", "Beijing-Air": "beijing_air",
               "PEMS-BAY": "pems_bay"}[ds]
        for dk, dfn in (("abl_at", "abl_at.json"), ("abl_tranad", "abl_tranad.json")):
            dp = os.path.join(_results_root(), "paper", dfn)
            if os.path.exists(dp):
                dd = json.load(open(dp, encoding="utf-8")).get(cfg)
                if dd:
                    d.setdefault(dk, {}).update(dd)
        means = {}
        for m in EXTERNAL:
            if m not in d:
                continue
            r, _ = per_seed(d, m)
            if r:
                means[m] = sum(r) / len(r)
        strongest = max(means, key=means.get)
        rec, _ = per_seed(d, "rdsg")
        brec, _ = per_seed(d, strongest)
        diffs = [a - b for a, b in zip(rec, brec)]
        rng = random.Random(0)
        n = len(diffs)
        boots = []
        for _ in range(10000):
            samp = [diffs[rng.randrange(n)] for _ in range(n)]
            boots.append(sum(samp) / n)
        boots.sort()
        lo, hi = boots[249], boots[9749]
        flag = "" if lo > 0 or hi < 0 else " (CI crosses zero)"
        rows.append(f"{ds} & {LABEL.get(strongest, strongest)} & "
                    f"{sum(diffs)/n:+.3f} & [{lo:+.3f}, {hi:+.3f}]{flag} \\\\")
    rows.append("\\hline")
    open(os.path.join(OUTDIR, "supp_paired_table.tex"), "w", encoding="utf-8").write(
        "\n".join(rows))
    print("wrote supp_paired_table.tex")

if __name__ == "__main__":
    main()
    paired_ci()
