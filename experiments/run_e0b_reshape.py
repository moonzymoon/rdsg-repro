# -*- coding: utf-8 -*-
"""E0b：reshape 注入实验（Prop 1(ii) 的经验落点，审稿加固 2026-10-06）。

注入构造（Prop 1(ii)）：窗口内仅对伙伴粒 i-grain==g* 的时刻重排 j 值，
使 g* 的条件分布（j 粒构成）偏移——模态粒保持（γ 变化 ≤ π_g*·O(ε)）、
窗口 j 多重集保持（边缘逐点不变）。预期：full 四元组（含 D_JS）可检，
γ+δ 消融近乎盲。

预注册分支（加固方案 2.3）：
(a) full recall 显著 > γ+δ（Wilcoxon p<0.05）→ Prop 1(ii) 有经验落点，入正文；
(b) 否则 → D_JS 必要性按理论-only 表述，诚实降格。

与 E0 同一系统/协议/FA 预算；只注入 reshape 事件（强对宿主）。
"""
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

import os, sys, json, time
import numpy as np

ROOT = _results_root()
import os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rdsg import RDSGDetector, event_metrics
from rdsg.metrics import match_events
from run_e0_syn import make_synthetic, recall_by_type, _to_short
from run_e0_syn2 import (REF_LEN, TEST_LEN, KS, WS, BLK, B, PAIR_STRATA,                         FA_TARGET, CAL_END, SPP, build_configs, channel_scan,
                         _merge_alarms)

OUT = os.path.normpath(os.path.join(_results_root(), "e0"))
N_EACH, MARGIN = 10, 600
KK = 4  # reshape 构造用的粒数（与主配置一致）


def reshape_inject(X_test, X_ref, specs, seed):
    """对 g*（参考段伙伴粒中占比最大者）内的时刻重排 j 值：
    g* 的条件分布（j 粒构成）偏移而模态组保持（γ 变化 ≤ π_g*·O(ε)），
    j 的窗口多重集不变（仅 g* 内部两组值交换）→ 边缘逐点保持。"""
    rng = np.random.default_rng(seed + 777)
    X = np.array(X_test, copy=True)
    for sp in specs:
        s, L, i, j = sp["start"], sp["length"], sp["i"], sp["j"]
        sl = slice(s, s + L)
        edges = np.quantile(X_ref[:, i], np.linspace(0, 1, KK + 1)[1:-1])
        gi = np.digitize(X[sl, i], edges)                 # 伙伴粒
        counts = [(gi == g).sum() for g in range(KK)]
        gstar = int(np.argmax(counts))
        idx = np.where(gi == gstar)[0]
        if len(idx) < 6:
            continue
        ej = np.quantile(X_ref[:, j], np.linspace(0, 1, KK + 1)[1:-1])
        grp = np.digitize(X[sl, j][idx], ej)              # g* 内 j 粒
        modal = np.bincount(grp, minlength=KK).argmax()
        others = [g for g in range(KK) if g != modal]
        if len(others) == 2:  # K=4 主情形：两非模态组的值互换 → 条件分布重分布、模态质量不动
            a = idx[grp == others[0]]
            b = idx[grp == others[1]]
            if len(a) and len(b):
                va = X[sl, j][a].copy()
                vb = X[sl, j][b].copy()
                rng.shuffle(va); rng.shuffle(vb)
                m = min(len(va), len(vb))
                X[sl, j][a[:m]] = vb[:m]
                X[sl, j][b[:m]] = va[:m]
    return X


def sample_specs_rs(strong_pairs, n_each, t_len, margin, seed, lengths=(60, 120)):
    rng = np.random.default_rng(seed)
    specs = []
    occupied = np.zeros(t_len, bool)
    def _place(L):
        for _ in range(200):
            s = rng.integers(margin, max(margin + 1, t_len - margin - L))
            if not occupied[s - margin:s + L + margin].any():
                occupied[s:s + L] = True
                return int(s)
        return None
    for (i, j) in strong_pairs[:n_each]:
        L = int(rng.choice(lengths))
        s = _place(L)
        if s is not None:
            specs.append(dict(type="rsp", i=int(i), j=int(j), start=s, length=L))
    return specs


def main():
    os.makedirs(OUT, exist_ok=True)
    taus = list(range(-3, 4))
    raw_path = os.path.join(OUT, "e0b_reshape_raw.json")
    all_res = {"full": {}, "gamma_delta": {}}
    if os.path.exists(raw_path):
        all_res.update(json.load(open(raw_path, encoding="utf-8")))
    for seed in range(1, 11):
        if seed in all_res["full"]:
            continue
        t0 = time.time()
        Xs = make_synthetic(REF_LEN + TEST_LEN, seed=100 + seed)
        X_ref, X_test = Xs[:REF_LEN], Xs[REF_LEN:]
        dets = build_configs(X_ref, taus)
        gd = build_configs(X_ref, taus, components=("dgamma", "ddelta"))
        dets_R = build_configs(X_ref, taus, strata=(PAIR_STRATA[0], 0))
        dets_E = build_configs(X_ref, taus, strata=(0, PAIR_STRATA[1]))
        gd_R = build_configs(X_ref, taus, components=("dgamma", "ddelta"), strata=(PAIR_STRATA[0], 0))
        gd_E = build_configs(X_ref, taus, components=("dgamma", "ddelta"), strata=(0, PAIR_STRATA[1]))
        specs = sample_specs_rs(dets_R[0].strong_pairs_, N_EACH, TEST_LEN, MARGIN, seed)
        true_events = [(sp["start"], sp["start"] + sp["length"], "rsp") for sp in specs]
        X_inj = reshape_inject(X_test, X_ref, specs, seed)
        print(f"seed {seed}: {len(specs)} reshape events ({time.time()-t0:.0f}s)", flush=True)
        for label, dd, dd_R, dd_E in (("full", dets, dets_R, dets_E),
                                      ("gamma_delta", gd, gd_R, gd_E)):
            Gfull = np.vstack([X_ref, X_inj])
            Gs = {id(d): d.granulate(Gfull) for d in dd + dd_R + dd_E}
            Gcal = {id(d): d.granulate(X_ref) for d in dd + dd_R + dd_E}
            evs_R, thR, _ = channel_scan(dd_R, Gcal, Gs, "R")
            evs_E, thE, _ = channel_scan(dd_E, Gcal, Gs, "E")
            evs = [(a - REF_LEN, b - REF_LEN, None) for a, b in evs_R] + \
                  [(a - REF_LEN, b - REF_LEN, None) for a, b in evs_E]
            m = event_metrics(true_events, evs, TEST_LEN, SPP)
            m["recall_by_type"] = recall_by_type(true_events, evs)
            all_res[label][seed] = m
            print(f"  [{label}] R={m['recall']:.2f} fa/d={m['fa_per_day']:.2f} n_pred={m['n_pred']}",
                  flush=True)
        json.dump(all_res, open(raw_path, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1, default=float)

    from scipy.stats import wilcoxon
    fr = [v["recall"] for v in all_res["full"].values()]
    gr = [v["recall"] for v in all_res["gamma_delta"].values()]
    if len(fr) >= 5 and not np.allclose(fr, gr):
        p = float(wilcoxon(fr, gr, alternative="greater").pvalue)
    elif len(fr) >= 5:
        p = 1.0  # identical per-seed recalls
    else:
        p = None
    out = {"summary": {"full_recall": float(np.mean(fr)), "gamma_delta_recall": float(np.mean(gr)),
                       "full_fa": float(np.mean([v["fa_per_day"] for v in all_res["full"].values()])),
                       "gd_fa": float(np.mean([v["fa_per_day"] for v in all_res["gamma_delta"].values()])),
                       "wilcoxon_p_greater": p,
                       "n_seeds": len(fr)},
          "config": dict(N_EACH=N_EACH, lengths=(60, 120), FA_TARGET=FA_TARGET)}
    json.dump(out, open(os.path.join(OUT, "e0b_reshape_summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=float)
    print(json.dumps(out["summary"], ensure_ascii=False))
    branch = "a" if (p is not None and p < 0.05) else "b"
    print(f"预注册分支: ({branch})")


if __name__ == "__main__":
    main()
