# -*- coding: utf-8 -*-
"""SMD 探针：论文协议（FA 速率校准 + 事件级评估）在 SMD 真实标签上的可行性测试。

决策门（跑前声明）：这是"标准 benchmark 门票"探针。结果合理 → 进 Supplemental；
结果不佳 → 不引用（§5.1 的排除理由是协议性的：点标签使事件级匹配近似、
参考段含异常风险，与结果无关，不构成选择性报告）。

协议对齐：
- 参考段 = train split（SMD 官方划分），测试段 = test split（真实点标签）；
- 事件形成 = 与气路完全相同的规则（60 样本窗攻击密度 >=0.3，gap<=2 窗合并）——
  复用 load_gas_pipeline 的事件派生逻辑口径；
- spp=60（1 分钟采样）；FA 目标 0.25/天；R/E 双通道 (16,24)+subset6；K=4,w=128;
- 方法：RDSG family + gamma-only 消融 + LOF-agg + pcorr（同一速率校准）。
"""
import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rdsg import RDSGDetector, event_metrics
from rdsg.metrics import match_events
from run_e0_syn2 import channel_scan
from run_paper_e2 import build_channel, scan_channel, point_baseline, PAIR_STRATA_E, lengths_gap

SMD_BASE = r"D:\0科研\数据\TSAD_数据集\SMD"
OUT = os.path.join(HERE, "..", "..", "results", "paper", "smd_probe.json")
SPP = 60.0
FA_TARGET = 0.25
MACHINES = ["machine-1-1", "machine-1-2", "machine-2-1"]


def load_smd(machine):
    tr = np.loadtxt(os.path.join(SMD_BASE, "train", machine + ".txt"), delimiter=",")
    te = np.loadtxt(os.path.join(SMD_BASE, "test", machine + ".txt"), delimiter=",")
    lab = np.loadtxt(os.path.join(SMD_BASE, "test_label", machine + ".txt"), delimiter=",")
    return tr, te, lab


def events_from_labels(lab, win=60, dens=0.3, gap_wins=2):
    """与气路同口径：60 样本窗标签密度>=dens 合并（gap<=gap_wins 窗）。"""
    n = len(lab) // win
    d = lab[:n * win].reshape(n, win).mean(axis=1)
    ab = d >= dens
    events, i = [], 0
    while i < n:
        if ab[i]:
            j, gap = i, 0
            while j < n and gap <= gap_wins:
                gap = 0 if ab[j] else gap + 1
                j += 1
            s, e = i * win, (j - gap) * win
            if e > s:
                events.append((s, e))
            i = j
        else:
            i += 1
    return events


def run_machine(machine, res):
    tr, te, lab = load_smd(machine)
    tr = (tr - tr.mean(0)) / (tr.std(0) + 1e-9)
    te = (te - tr.mean(0)) / (tr.std(0) + 1e-9)   # 用参考段统计标准化（无泄漏）
    true_events = [(s, e, "real") for s, e in events_from_labels(lab)]
    n_true = len(true_events)
    X_full = np.vstack([tr, te])
    taus = [0, 1, -1]
    planner = build_channel(tr, taus, strata=(16, 24))[0]
    force_R, force_E = planner.strong_pairs_, planner.weak_pairs_
    out = {"n_channels": tr.shape[1], "n_ref": len(tr), "n_test": len(te),
           "n_true_events": n_true, "anom_rate": float(lab.mean())}
    # ---- ours (family) 与 gamma-only ----
    for name, comps in (("rdsg", None), ("gamma_only", ("dgamma",))):
        t0 = time.time()
        from run_paper_e2 import build_channel as bc
        dR = bc(tr, taus, comps=comps, strata=(16, 0), force_pairs=(force_R, force_E))
        dE = bc(tr, taus, comps=comps, strata=(0, 24), n_subset=6,
                force_pairs=(force_R, force_E))
        evs_R, _, _ = scan_channel(dR, tr, X_full, SPP)
        evs_E, _, _ = scan_channel(dE, tr, X_full, SPP)
        evs = [(a - len(tr), b - len(tr)) for a, b in evs_R + evs_E]
        evs = [(max(a, 0), b) for a, b in evs if b > len(tr)]
        pred = [(int(a), int(min(b, len(te))), None) for a, b in evs]
        m = event_metrics(true_events, pred, len(te), SPP)
        out[name] = dict(recall=m["recall"], fa_per_day=m["fa_per_day"])
        print(f"[{machine}] {name}: R={m['recall']:.3f} fa={m['fa_per_day']:.2f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    # ---- 简单基线：LOF-agg / pcorr ----
    from baselines.relational import WindowDivergenceBaseline
    pr, _ = WindowDivergenceBaseline("pcorr", spp=SPP, fa_target=FA_TARGET).run(tr, te)
    m = event_metrics(true_events, [(a, b, None) for a, b in pr], len(te), SPP)
    out["pcorr"] = dict(recall=m["recall"], fa_per_day=m["fa_per_day"])
    print(f"[{machine}] pcorr: R={m['recall']:.3f} fa={m['fa_per_day']:.2f}", flush=True)
    pl, _ = point_baseline(tr, te, "lof", fa_target=FA_TARGET, seed=0, spp=SPP)
    m = event_metrics(true_events, [(e[0], e[1], None) for e in pl], len(te), SPP)
    out["lof_agg"] = dict(recall=m["recall"], fa_per_day=m["fa_per_day"])
    print(f"[{machine}] lof: R={m['recall']:.3f} fa={m['fa_per_day']:.2f}", flush=True)
    res[machine] = out
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1)


def main():
    res = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
    for mch in MACHINES:
        if mch not in res:
            run_machine(mch, res)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if isinstance(vv, dict)}
                      for k, v in res.items()}, indent=1))


if __name__ == "__main__":
    main()
