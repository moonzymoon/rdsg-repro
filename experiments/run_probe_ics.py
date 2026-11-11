# -*- coding: utf-8 -*-
"""双探针：BATADAL（水网 ICS，WADI 姊妹，可能进文作第三份攻击日志）与
PSM（标准基准家族，SMD 同模式）。

预声明决策门（跑前固定）：
- BATADAL 若 RDSG 预算召回合理（显著非零且不低于简单基线）→ 作为 Supplemental
  第三份 ICS 攻击日志入文；否则 → 不引用，入弹药库（排除理由：协议性）。
- PSM 与 SMD 同模式：预期点型异常为主，值级基线强、关系族弱 → 弹药库。
协议：与论文完全一致——参考段自扫描 FA 速率校准 0.25/天、事件级指标、
R/E 双通道 (16,24)+subset6、K=4/w=128；事件形成按采样率自适应密度规则。
"""
import os, sys, json, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rdsg import event_metrics
from run_paper_e2 import build_channel, scan_channel, point_baseline
from baselines.relational import WindowDivergenceBaseline

DATA = r"D:\0科研\数据\TSAD_数据集"
OUT = os.path.join(HERE, "..", "..", "results", "paper", "probe_ics.json")


def events_from_labels(lab, win, dens=0.3, gap_wins=2):
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


def load_batadal():
    tr = pd.read_csv(os.path.join(DATA, "BATADAL", "BATADAL_training1.csv"))
    te = pd.read_csv(os.path.join(DATA, "BATADAL", "BATADAL_training2.csv"))
    tr.columns = [c.strip() for c in tr.columns]
    te.columns = [c.strip() for c in te.columns]
    cols = [c for c in tr.columns if c not in ("DATETIME", "ATT_FLAG")]
    Xr = tr[cols].ffill().bfill().values.astype(float)
    Xe = te[cols].ffill().bfill().values.astype(float)
    lab = (te["ATT_FLAG"].values == 1).astype(float)
    return Xr, Xe, lab, 3600.0, 24


def load_psm():
    Xr = np.load(os.path.join(DATA, "PSM", "train.npy"))
    Xe = np.load(os.path.join(DATA, "PSM", "test.npy"))
    lab = np.load(os.path.join(DATA, "PSM", "test_label.npy")).astype(float).ravel()
    Xr = np.nan_to_num(Xr, nan=np.nanmean(Xr, 0))
    Xe = np.nan_to_num(Xe, nan=np.nanmean(Xr, 0))
    return Xr, Xe, lab, 60.0, 60


def run_probe(name, res):
    if name == "batadal":
        Xr, Xe, lab, spp, win = load_batadal()
    else:
        Xr, Xe, lab, spp, win = load_psm()
    mu, sd = Xr.mean(0), Xr.std(0) + 1e-9
    Xr, Xe = (Xr - mu) / sd, (Xe - mu) / sd
    true_events = [(s, e, "real") for s, e in events_from_labels(lab, win)]
    out = dict(n_channels=Xr.shape[1], n_ref=len(Xr), n_test=len(Xe),
               n_true=len(true_events), anom_rate=float(lab.mean()),
               event_lens=[e - s for s, e, _ in true_events])
    X_full = np.vstack([Xr, Xe])
    taus = [0, 1, -1]
    planner = build_channel(Xr, taus, strata=(16, 24))[0]
    fR, fE = planner.strong_pairs_, planner.weak_pairs_
    for nm, comps, strata in (("rdsg", None, (16, 0)), ("gamma_only", ("dgamma",), (16, 0))):
        t0 = time.time()
        dR = build_channel(Xr, taus, comps=comps, strata=strata, force_pairs=(fR, fE))
        dE = build_channel(Xr, taus, comps=comps, strata=(0, 24), n_subset=6,
                           force_pairs=(fR, fE))
        evs = []
        for d in (dR, dE):
            ee, _, _ = scan_channel(d, Xr, X_full, spp)
            evs += ee
        evs = [(a - len(Xr), b - len(Xr)) for a, b in evs]
        pred = [(max(int(a), 0), int(min(b, len(Xe))), None) for a, b in evs if b > len(Xr)]
        m = event_metrics(true_events, pred, len(Xe), spp)
        out[nm] = dict(recall=m["recall"], fa=m["fa_per_day"])
        print(f"[{name}] {nm}: R={m['recall']:.3f} fa={m['fa_per_day']:.2f} ({time.time()-t0:.0f}s)", flush=True)
    pr, _ = WindowDivergenceBaseline("pcorr", spp=spp, fa_target=0.25).run(Xr, Xe)
    m = event_metrics(true_events, [(a, b, None) for a, b in pr], len(Xe), spp)
    out["pcorr"] = dict(recall=m["recall"], fa=m["fa_per_day"])
    print(f"[{name}] pcorr: R={m['recall']:.3f} fa={m['fa_per_day']:.2f}", flush=True)
    pl, _ = point_baseline(Xr, Xe, "lof", fa_target=0.25, seed=0, spp=spp)
    m = event_metrics(true_events, pl, len(Xe), spp)
    out["lof_agg"] = dict(recall=m["recall"], fa=m["fa_per_day"])
    print(f"[{name}] lof: R={m['recall']:.3f} fa={m['fa_per_day']:.2f}", flush=True)
    res[name] = out
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1)


def main():
    res = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
    for name in ("batadal", "psm"):
        if name not in res:
            run_probe(name, res)
    print(json.dumps(res, indent=1)[:1500])


if __name__ == "__main__":
    main()
