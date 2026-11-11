# -*- coding: utf-8 -*-
"""E0 capability gate experiment v2 (synthetic VAR capability gate) -- re-test after the statistic-aggregation restructuring.

v2 design changes (see results/e0/E0_gate_report_20260926.md sec. 5 + method draft v1.1 appendix):
1. Unit validity mask: (edge, component) units with sigma < 5% of the component median are dropped from the family;
   sigma floor = 0.2 x component median -- removes heavy-tail units polluting the family maximum
   (the main cause of the v1 thresholds 7.2-8.9);
2. Candidate-pair stratification: top-m strong pairs + bottom-m' weak pairs (reference-segment information only), family size controlled;
3. Reference-segment self-scan sets theta: mu/sigma use the first half of the reference segment [0, REF/2), and the
   full alarm pipeline (multi-config union + event merging) runs on the second half for empirical calibration,
   directly controlling the FA alarm-window rate <= target (0.5/day) -- replaces alpha/boundary bookkeeping and fixes
   the unmanaged FA of the multi-config union;
4. Baseline FA matching: IF/LOF thresholds are scanned on the reference segment to FA <= 0.5/day; the MI graph uses
   the same self-scan criterion (fixes the v1 zero-alarm issue).
"""
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

import os, sys, json, time
import numpy as np

ROOT = _results_root()
import os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rdsg import RDSGDetector, inject, sample_specs, event_metrics
from rdsg.tables import block_tables, window_sum
from rdsg.metrics import match_events
from rdsg.detector import build_pair_edges
from run_e0_syn import (make_synthetic, rank_pairs_by_ref_gamma, recall_by_type,
                        merge_typed, per_event_detail, _to_short)

OUT = os.path.normpath(os.path.join(_results_root(), "e0"))
SPP = 60.0
REF_LEN, TEST_LEN = 12000, 18000
KS, WS = (4, 2), (128, 64, 32)  # lever 2: w=32 raises the contamination share of L=30 events to ~0.94
BLK, B = 64, 499
N_EACH, MARGIN = 10, 600
PAIR_STRATA = (16, 24)  # true structure of the 12-channel system: 7 genuinely strong unordered pairs + 12 weak ordered pairs in group D
FA_TARGET = 0.25         # events/day (self-scan -> test has ~1.6x transfer inflation; tightened to keep the gate FA at 1.0)
CAL_END = REF_LEN // 2   # mu/sigma segment: [0, 6000); self-scan segment: [6000, 12000)


def _merge_alarms(times, flags, gap):
    """Merge alarms in time into event intervals [(t0, t1)]."""
    on = times[flags]
    if len(on) == 0:
        return []
    events, t0, prev = [], on[0], on[0]
    for t in on[1:]:
        if t - prev > gap:
            events.append((t0, prev + BLK))
            t0 = t
        prev = t
    events.append((t0, prev + BLK))
    return events


def pick_theta(u_ref, t_ref, fa_target, spp, gap):
    """Rate-based self-scan calibration: the smallest theta whose reference-segment
    alarm-window rate meets the budget (fa_target read as alarm windows per day over
    the self-scan span, already carrying the per-component split of the family
    budget). On spans whose budget admits zero alarm windows this reduces to the
    reference maximum; event merging at the evaluation gap can only lower the event
    count below this rate, so the budget holds in both windows and merged events."""
    span_days = (t_ref[-1] - t_ref[0]) * spp / 86400.0
    n = len(u_ref)
    k = int(np.floor(fa_target * span_days))           # allowed alarm windows
    srt = np.sort(u_ref)
    if k >= n:
        return float(srt[0]), n
    th = float(srt[n - k - 1])                          # smallest theta with count(u>th) <= k
    return th, int((u_ref > th).sum())


# Detection family components: rho is a weighted sum of 0/1 equality tests (a discrete jump statistic whose
# z is quantised into steps of ~2.5, so its family maximum is naturally 5-15, pushing theta into the signal
# region) -- rho is excluded from the detection family, which uses only the continuous components;
# rho is kept for the typing rule (detector.type_events uses the raw four_tuple and is unaffected).
DETECT_COMPS = ("dgamma", "ddelta", "djs")


def build_configs(X_ref, taus, components=None, strata=None):
    dets = []
    for K in KS:
        for w in WS:
            comps = components or DETECT_COMPS
            if K == 2:
                comps = tuple(c for c in comps if c != "djs")  # conditional distribution too coarse at 2 granules
            d = RDSGDetector(K=K, w=w, blk=min(BLK, w // 2), B=B, q=0.05, seed=1,
                             components=comps,
                             pair_strata=strata if strata is not None else PAIR_STRATA)
            d.fit(X_ref, channels=list(range(X_ref.shape[1])), taus=taus)
            dets.append(d)
    return dets


def channel_scan(dd, Gcal, Gs, label, prune_cap=5.5, fa_target=None,
                 cal_end=None, scan_start=None, scan_end=None, gap=None, spp=None,
                 selfscan_start=None):
    """Single channel (one group of detectors): calibration -> self-scan pruning -> per-component family theta -> test scan -> alarms/events.

    Per-component family split (v5): each component c sets its own theta_c (FA target split evenly, union
    bound controls); alarm = any component exceeding its theta_c. Removes the threshold tax of
    max-over-components.
    cal_end/scan_start/scan_end are parameterisable (e.g. non-standard segment lengths in E1).
    Returns (events, thetas, (grid, Tq)).
    """
    fa_target = fa_target or FA_TARGET
    cal_end = CAL_END if cal_end is None else cal_end
    scan_start = REF_LEN if scan_start is None else scan_start
    scan_end = REF_LEN + TEST_LEN if scan_end is None else scan_end
    gap = 8 * BLK if gap is None else gap
    spp = SPP if spp is None else spp
    selfscan_start = cal_end if selfscan_start is None else selfscan_start
    for d in dd:
        d.calibrate(Gcal[id(d)], cal_end, cal_t0=0)
        d.prune_units_by_scan(Gcal[id(d)], selfscan_start, scan_start, z_cap=5.5)
    comps = sorted(set(dd[0].components).intersection(*[set(d.components) for d in dd]))
    fa_c = fa_target / len(comps)
    grid_r, Zfull_r = _unit_stack(dd, Gcal, selfscan_start, scan_start)
    thetas = {}
    for ci, c in enumerate(comps):
        Tc = Zfull_r[:, :, ci].max(axis=1)
        thetas[c], _ = pick_theta(Tc, grid_r, fa_c, spp, gap=gap)
    grid_q, Zfull_q = _unit_stack(dd, Gs, scan_start, scan_end)
    alarm = np.zeros(len(grid_q), bool)
    for ci, c in enumerate(comps):
        Tc_q = Zfull_q[:, :, ci].max(axis=1)
        alarm |= Tc_q > thetas[c]
    evs = _merge_alarms(grid_q, alarm, gap=gap)
    return evs, thetas, (grid_q, Zfull_q.max(axis=(1, 2)))


def _unit_stack(dd, Gs, t_start, t_end):
    """Stack unit-level z across configs: aligned by edge identity, unit = (edge, component), taking the max over configs."""
    comps = sorted(set(dd[0].components).intersection(*[set(d.components) for d in dd]))
    common = set(dd[0].edges_)
    for d in dd:
        common &= set(d.edges_)
    common = sorted(common)
    cpos = {e: k for k, e in enumerate(common)}
    per = []
    for d in dd:
        t, Z = d.zfull_series(Gs[id(d)], t_start, t_end)  # (nb, E, Cd)
        cd_map = [(k, comps.index(c)) for k, c in enumerate(d.components) if c in comps]
        per.append((t, Z, list(d.edges_)))
    grid = sorted(set().union(*[set(t.tolist()) for t, _, _ in per]))
    gpos = {t: i for i, t in enumerate(grid)}
    Zu = np.zeros((len(grid), len(common), len(comps)))
    for t, Z, es in per:
        rows = np.array([gpos[ti] for ti in t if ti in gpos])
        for k, e in enumerate(es):
            if e not in cpos:
                continue
            kc = cpos[e]
            tgt = Zu[rows, kc, :]
            src = Z[: len(rows), k, :]
            for k_in, ci_out in cd_map:
                tgt[:, ci_out] = np.maximum(tgt[:, ci_out], src[:, k_in])
            Zu[rows, kc, :] = tgt
    return np.asarray(grid), Zu


def run_ours(Xs, seed, taus, dets, ab_dets, dets_R, dets_E, ab_R, ab_E):
    X_ref, X_test = Xs[:REF_LEN], Xs[REF_LEN:]
    # injection hosts are taken from each detector's own candidate graph (ranking of the main config K=4, w=128) --
    # protocol correction: the experiment breaks/grows exactly the edges the detector monitors (hosts are always in the family)
    strong = dets_R[0].strong_pairs_
    weak = dets_E[0].weak_pairs_
    specs = sample_specs(None, strong, weak, n_each=N_EACH,
                         lengths=(30, 60, 120), t_len=TEST_LEN, margin=MARGIN, seed=seed)
    # lever 3: INV sign-flip protocol -- target = sign-flipped reference-segment rank correlation (|rho*| >= 0.6)
    from scipy.stats import spearmanr as _sr
    for sp in specs:
        if sp["type"] == "inv":
            r_ref = float(_sr(X_ref[-1024:, sp["i"]], X_ref[-1024:, sp["j"]]).statistic)
            sp["rho_target"] = float(np.clip(-r_ref, -1, -0.6)) if r_ref >= 0                 else float(np.clip(-r_ref, 0.6, 1))
    X_test_inj = inject(X_test, specs, seed=seed)
    true_events = [(sp["start"], sp["start"] + sp["length"], sp["type"]) for sp in specs]

    out = {}
    for label, dd, dd_R, dd_E in (("rdsg", dets, dets_R, dets_E),
                                  ("gamma_only", ab_dets, ab_R, ab_E)):
        Gfull = np.vstack([X_ref, X_test_inj])
        Gs = {id(d): d.granulate(Gfull) for d in dd + dd_R + dd_E}
        Gcal = {id(d): d.granulate(X_ref) for d in dd + dd_R + dd_E}
        # independent calibration and alarm per channel (R = strong graph for breakdown/inversion; E = weak pairs for emergence)
        evs_R, theta_R, (tqR, TqR) = channel_scan(dd_R, Gcal, Gs, "R")
        evs_E, theta_E, (tqE, TqE) = channel_scan(dd_E, Gcal, Gs, "E")
        print(f"  [{label}] θ_R={theta_R} (n_ev_R={len(evs_R)}) "
              f"θ_E={theta_E} (n_ev_E={len(evs_E)})", flush=True)
        theta = dict(theta_R)
        theta.update({f"E:{k}": v for k, v in theta_E.items()})
        n_ev = len(evs_R) + len(evs_E)
        evs = [(a, b, "R") for a, b in evs_R] + [(a, b, "E") for a, b in evs_E]
        t_q, T_q = tqR, TqR  # main-channel series for localisation
        # event localisation + typing (using the strongest config at the event peak)
        typed = []
        for (t0, t1, ch) in evs:
            dd_ch = dd_R if ch == "R" else dd_E
            tq_ch, Tq_ch = (tqR, TqR) if ch == "R" else (tqE, TqE)
            member_ts = tq_ch[(tq_ch >= t0) & (tq_ch < t1)]
            member_us = Tq_ch[(tq_ch >= t0) & (tq_ch < t1)]
            peak_t = int(member_ts[np.argmax(member_us)])
            # for the edge of the peak unit, find the config with the largest z -> use that config for localisation and typing
            from rdsg.tables import window_sum as _ws
            bi, bz = 0, -1.0
            for ci, d in enumerate(dd_ch):
                t_lo = max(0, peak_t - BLK)
                t_hi = min(peak_t + BLK + d.w, REF_LEN + TEST_LEN)
                if t_hi - t_lo < d.w + BLK:
                    continue
                tt, ZZ = d.z_series(Gs[id(d)], t_start=t_lo, t_end=t_hi)
                if len(tt) and ZZ.max() > bz:
                    bz, bi = float(ZZ.max()), ci
            d = dd_ch[bi]
            d.z_series(Gs[id(d)], REF_LEN, REF_LEN + TEST_LEN)  # restore the full-span cached block tables
            ev = dict(t0=t0, t1=t1, scale=d.w,
                      members=[dict(t=int(mt), zmax=float(mu), sig_idx=None, n_sig=0)
                               for mt, mu in zip(member_ts, member_us)])
            ev = d.localize(Gs[id(d)], [ev])[0]
            ev = d.type_events(Gs[id(d)], [ev])[0]
            typed.append(ev)
        pred = [(e["t0"] - REF_LEN, min(e["t1"] - REF_LEN, TEST_LEN), _to_short(e.get("type", "?")))
                for e in typed]
        m = event_metrics(true_events, pred, TEST_LEN, SPP)
        m["recall_by_type"] = recall_by_type(true_events, pred)
        m["per_event"] = per_event_detail(true_events, pred)
        m["theta"], m["selfscan_events"] = theta, n_ev
        out[label] = m
        print(f"[{label} seed {seed}] θ={ {k: round(v,2) for k,v in theta.items()} } R={m['recall']:.2f} "
              f"({m['recall_by_type']}) fa/d={m['fa_per_day']:.2f} type={m['type_acc']:.2f} "
              f"delay={m['delay_med']}", flush=True)
    return out, true_events, X_test_inj


def run_baselines(Xs, seed, taus, dets, X_test_inj, true_events):
    X_ref = Xs[:REF_LEN]
    res = {}
    # ---- IF / LOF: reference-segment FA matching ----
    from baselines.point_agg import _fit_score
    n = X_ref.shape[1]
    half = X_ref.shape[0] // 2
    ref_agg = np.full(X_ref.shape[0], -np.inf)
    for j in range(n):
        ref_agg[:half] = np.maximum(ref_agg[:half],
                                    _fit_score("if", X_ref[half:, j:j+1], X_ref[:half, j:j+1], seed))
        ref_agg[half:] = np.maximum(ref_agg[half:],
                                    _fit_score("if", X_ref[:half, j:j+1], X_ref[half:, j:j+1], seed))
    days_ref = X_ref.shape[0] * SPP / 86400.0
    allow = max(1, int(np.ceil(FA_TARGET * days_ref)))
    for model in ("if", "lof"):
        ra = np.full(X_ref.shape[0], -np.inf)
        for j in range(n):
            ra[:half] = np.maximum(ra[:half],
                                   _fit_score(model, X_ref[half:, j:j+1], X_ref[:half, j:j+1], seed))
            ra[half:] = np.maximum(ra[half:],
                                   _fit_score(model, X_ref[:half, j:j+1], X_ref[half:, j:j+1], seed))
        # rate-based calibration, identical semantics to the family pick_theta
        k = int(np.floor(FA_TARGET * len(ra) * SPP / 86400.0))
        srt = np.sort(ra)
        th = float(srt[len(srt) - k - 1]) if k < len(srt) else float(srt[0])
        agg = np.full(TEST_LEN, -np.inf)
        for j in range(n):
            agg = np.maximum(agg, _fit_score(model, X_ref[:, j:j+1],
                                             X_test_inj[:, j:j+1], seed))
        evs = _merge_alarms(np.arange(TEST_LEN), agg > th, gap=2 * BLK)
        pred = [(a, b, None) for a, b in evs]
        m = event_metrics(true_events, pred, TEST_LEN, SPP)
        m["recall_by_type"] = recall_by_type(true_events, pred)
        res[f"b4_{model}"] = m
        print(f"[b4_{model} seed {seed}] R={m['recall']:.2f} ({m['recall_by_type']}) "
              f"fa/d={m['fa_per_day']:.2f}", flush=True)
    # ---- MI graph: same self-scan criterion ----
    from rdsg.stats import four_tuple  # noqa
    d0 = dets[0]  # (K=4, w=512)
    G = d0.granulate(np.vstack([X_ref, X_test_inj]))
    edges = d0.edges_
    K, blk, half = d0.K, d0.blk, d0.w // d0.blk

    def mi_series(t_start, t_end):
        span0 = max(0, t_start - d0.w)
        tab = block_tables(G, edges, K, blk, t0=span0, t1=t_end)
        nb = tab.shape[1]
        k0 = (t_start - span0 + blk - 1) // blk
        ts, ss = [], []
        for k in range(k0, nb - half + 1):
            T1 = window_sum(tab, np.arange(k - half, k))
            T2 = window_sum(tab, np.arange(k, k + half))
            P1 = T1 / np.maximum(T1.sum(axis=(-2, -1), keepdims=True), 1e-12)
            P2 = T2 / np.maximum(T2.sum(axis=(-2, -1), keepdims=True), 1e-12)
            m1 = P1.sum(-1, keepdims=True) * P1.sum(-2, keepdims=True)
            m2 = P2.sum(-1, keepdims=True) * P2.sum(-2, keepdims=True)
            mi1 = np.where(P1 > 0, P1 * np.log(np.maximum(P1 / np.maximum(m1, 1e-12), 1e-12)), 0).sum((-2, -1))
            mi2 = np.where(P2 > 0, P2 * np.log(np.maximum(P2 / np.maximum(m2, 1e-12), 1e-12)), 0).sum((-2, -1))
            ts.append(span0 + k * blk)
            ss.append(float(np.abs(mi2 - mi1).sum()))
        return np.asarray(ts), np.asarray(ss)

    t_r, s_r = mi_series(CAL_END, REF_LEN)
    theta, _ = pick_theta(s_r, t_r, FA_TARGET, SPP, gap=8 * BLK)
    t_q, s_q = mi_series(REF_LEN, REF_LEN + TEST_LEN)
    evs = _merge_alarms(t_q, s_q > theta, gap=8 * BLK)
    pred = [(a - REF_LEN, b - REF_LEN, None) for a, b in evs]
    m = event_metrics(true_events, pred, TEST_LEN, SPP)
    m["recall_by_type"] = recall_by_type(true_events, pred)
    m["theta"] = theta
    res["b5a_mi"] = m
    print(f"[b5a_mi seed {seed}] θ={theta:.1f} R={m['recall']:.2f} ({m['recall_by_type']}) "
          f"fa/d={m['fa_per_day']:.2f}", flush=True)
    return res


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10,
                    help="run only the first N seeds (smoke test)")
    n_seeds = ap.parse_args().seeds
    os.makedirs(OUT, exist_ok=True)
    taus = list(range(-3, 4))
    raw_path = os.path.join(OUT, "e0_syn2_raw.json")
    all_res = {m: {} for m in ("rdsg", "gamma_only", "b4_if", "b4_lof", "b5a_mi")}
    if os.path.exists(raw_path):
        all_res.update(json.load(open(raw_path, encoding="utf-8")))
    for seed in range(1, 1 + n_seeds):
        if seed in all_res["rdsg"]:
            continue
        t0 = time.time()
        Xs = make_synthetic(REF_LEN + TEST_LEN, seed=100 + seed)
        dets = build_configs(Xs[:REF_LEN], taus)
        ab = build_configs(Xs[:REF_LEN], taus, components=("dgamma",))
        dets_R = build_configs(Xs[:REF_LEN], taus, strata=(PAIR_STRATA[0], 0))
        dets_E = build_configs(Xs[:REF_LEN], taus, strata=(0, PAIR_STRATA[1]))
        ab_R = build_configs(Xs[:REF_LEN], taus, components=("dgamma",), strata=(PAIR_STRATA[0], 0))
        ab_E = build_configs(Xs[:REF_LEN], taus, components=("dgamma",), strata=(0, PAIR_STRATA[1]))
        print(f"seed {seed}: configs built ({time.time()-t0:.0f}s)", flush=True)
        out, true_events, X_test_inj = run_ours(Xs, seed, taus, dets, ab, dets_R, dets_E, ab_R, ab_E)
        for k, v in out.items():
            all_res[k][seed] = v
        json.dump(all_res, open(raw_path, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1, default=float)
        bres = run_baselines(Xs, seed, taus, dets, X_test_inj, true_events)
        for k, v in bres.items():
            all_res[k][seed] = v
        json.dump(all_res, open(raw_path, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1, default=float)

    summary = {}
    for m, per in all_res.items():
        if not per:
            continue
        recalls = [v["recall"] for v in per.values()]
        fas = [v["fa_per_day"] for v in per.values()]
        rt = {t: float(np.mean([v["recall_by_type"][t] for v in per.values()]))
              for t in ("inv", "brk", "emb")}
        summary[m] = dict(recall_mean=float(np.mean(recalls)), fa_per_day_mean=float(np.mean(fas)),
                          recall_by_type=rt,
                          delay_med=float(np.nanmedian([v["delay_med"] for v in per.values()])),
                          type_acc=float(np.mean([v["type_acc"] for v in per.values()])),
                          pa_f1=float(np.mean([v["pa_f1"] for v in per.values()])))
    from scipy.stats import wilcoxon
    ours_r = [v["recall"] for v in all_res["rdsg"].values()]
    if_r = [v["recall"] for v in all_res["b4_if"].values()]
    p = 1.0 if np.allclose(ours_r, if_r) else float(wilcoxon(ours_r, if_r).pvalue)
    rt = summary["rdsg"]["recall_by_type"]
    verdict = dict(gate_recall_by_type=bool(all(v >= 0.8 for v in rt.values())),
                   gate_fa=bool(summary["rdsg"]["fa_per_day_mean"] <= 1.0),
                   gate_wilcoxon=bool(p < 0.05), wilcoxon_p=p,
                   PASS=bool(all(v >= 0.8 for v in rt.values())
                             and summary["rdsg"]["fa_per_day_mean"] <= 1.0 and p < 0.05))
    out = dict(summary=summary, verdict=verdict,
               config=dict(REF_LEN=REF_LEN, TEST_LEN=TEST_LEN, KS=KS, WS=WS, BLK=BLK,
                           B=B, N_EACH=N_EACH, MARGIN=MARGIN, PAIR_STRATA=PAIR_STRATA,
                           FA_TARGET=FA_TARGET, CAL_END=CAL_END, SPP=SPP))
    json.dump(out, open(os.path.join(OUT, "e0_syn2_summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=float)
    print(json.dumps(verdict, ensure_ascii=False))
    for m, s in summary.items():
        print(m, json.dumps(s, ensure_ascii=False))


if __name__ == "__main__":
    main()
