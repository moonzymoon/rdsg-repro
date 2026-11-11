# -*- coding: utf-8 -*-
"""E0 capability gate experiment (stop-condition criterion, reviewer revision 5).

Data: METR-LA (REF=12000 / TEST=22000, chronological split, no overlap).
Injection: margin-preserving suite INV/BRK/EMB, 10 events/seed x 5 seeds (margin=700 > merge radius 512).
Methods: our RDSG vs gamma-only ablation vs B4 (IF/LOF aggregation) vs B5a (MI graph).
Gate (values suggested by the registered card): recall >= 0.8 for all three event types, false alarms <= 1/day,
and Wilcoxon p < 0.05 vs B4. If not met -> project stops and failure modes are reported.

Calibration sharing: the reference segment is invariant across seeds -> ours/gamma-only are each
calibrated once and reused for the 5 seeds.
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
from rdsg import RDSGDetector, Granulator, inject, sample_specs, event_metrics
from rdsg.data import load_metr_like_h5
from rdsg.tables import edge_code
from rdsg.stats import gamma_soft
from rdsg.metrics import match_events
from rdsg.detector import build_pair_edges
from baselines.point_agg import point_agg_scores, alarms_to_events
from baselines.mi_graph import MIGraphDetector

DATA = os.path.join(ROOT, "datasets", "traffic", "metr-la.h5")
OUT = os.path.join(_results_root(), "e0")
SPP = 300.0  # seconds per point (METR-LA 5-minute sampling)

N_CHANNELS = 40
REF_LEN, TEST_LEN = 12000, 22000
K, B = 4, 499
W = 512  # main window used to stratify injection pairs
WS = (512, 128)  # dual-scale detection windows: coarse scale for stable localisation, fine scale for resolution on short events
BLK = 64
CAL_SPAN = 8000  # tail of the reference segment used for mu/sigma calibration (adapts to slow drift)
N_EACH, MARGIN = 10, 640
TYPE_MAP = {"inversion": "inv", "breakdown": "brk", "emergence": "emb"}


def load_data():
    df = load_metr_like_h5(DATA)
    X = df.values.astype(float)
    seg = X[: REF_LEN + TEST_LEN]
    ok = ~np.isnan(seg).any(axis=0)
    chans = [i for i, k in enumerate(ok) if k][:N_CHANNELS]
    Xs = seg[:, chans]
    print(f"data: {Xs.shape}, channels {len(chans)}, NaN-free slice ok", flush=True)
    return Xs


def rank_pairs_by_ref_gamma(X_ref, channels, K, w):
    """Full pairwise gamma_soft (lag=0) on the tail window of the reference segment -> quantile-stratified strong/weak pairs (injection pairs chosen using reference-segment information only)."""
    gran = Granulator(K).fit(X_ref)
    Gref = gran.transform(X_ref[-2 * w:])
    gam = {}
    for i in channels:
        for j in channels:
            if i == j:
                continue
            code = edge_code(Gref, (i,), j, 0, K)
            c = code[code >= 0]
            tab = np.bincount(c, minlength=K * K).reshape(K, K).astype(float)
            gam[(i, j)] = float(gamma_soft(tab[None])[0])
    vals = np.array(sorted(gam.values()))
    hi, lo = np.quantile(vals, 0.9), np.quantile(vals, 0.1)
    strong = sorted([p for p, v in gam.items() if v >= hi], key=gam.get, reverse=True)
    weak = sorted([p for p, v in gam.items() if v <= lo], key=gam.get)
    return strong, weak, gam


def recall_by_type(true_events, pred_events):
    out = {}
    for typ in ("inv", "brk", "emb"):
        te = [e for e in true_events if e[2] == typ]
        if not te:
            out[typ] = None
            continue
        matches, _, _ = match_events(te, [(p[0], p[1]) for p in pred_events])
        out[typ] = sum(1 for _, p in matches if p is not None) / len(te)
    return out


def _to_short(t):
    return TYPE_MAP.get(t, t)


def merge_typed(events):
    """Cross-scale merge: time-overlapping events take the union interval; typing prefers the fine scale (higher contamination share, more reliable typing)."""
    if not events:
        return []
    evs = sorted(events, key=lambda e: e["t0"])
    out = []
    for e in evs:
        if out and e["t0"] <= out[-1]["t1"]:
            prev = out[-1]
            prev["t1"] = max(prev["t1"], e["t1"])
            fine = e if e.get("scale", 1e9) < prev.get("scale", 1e9) else prev
            coarse = prev if fine is e else e
            prev["scale"] = fine.get("scale")
            prev["type"] = fine.get("type", "?") if fine.get("type") not in (None, "untyped")                 else (coarse.get("type") or prev.get("type", "?"))
        else:
            out.append(e)
    return out


def per_event_detail(true_events, pred_events):
    """Per-ground-truth-event hit detail (for failure-mode analysis): (start, length, type, hit)."""
    from rdsg.metrics import match_events as _mm
    matches, _, _ = _mm(true_events, [(p[0], p[1]) for p in pred_events])
    out = []
    for (t0, t1, typ), p in matches:
        out.append([int(t0), int(t1 - t0), typ, bool(p is not None)])
    return out


def run_seed(Xs, seed, taus, det_full, det_ab):
    t0 = time.time()
    X_ref, X_test = Xs[:REF_LEN], Xs[REF_LEN:]
    strong, weak, gam = rank_pairs_by_ref_gamma(X_ref, list(range(Xs.shape[1])), K, W)
    assert len(strong) >= 2 * N_EACH, f"insufficient strong pairs: {len(strong)}"
    assert len(weak) >= N_EACH, f"insufficient weak pairs: {len(weak)}"
    specs = sample_specs(None, strong, weak, n_each=N_EACH,
                         lengths=(30, 60, 120), t_len=TEST_LEN, margin=MARGIN, seed=seed)
    X_test_inj = inject(X_test, specs, seed=seed)
    true_events = [(sp["start"], sp["start"] + sp["length"], sp["type"]) for sp in specs]
    n_by_type = {t: sum(1 for e in true_events if e[2] == t) for t in ("inv", "brk", "emb")}
    print(f"[seed {seed}] events placed: {n_by_type}", flush=True)

    res = {}
    # ---- our RDSG (full four-tuple; dual-scale; calibration shared across seeds) ----
    G = det_full[0].granulate(np.vstack([X_ref, X_test_inj]))
    typed_all = []
    for det in det_full:  # per scale
        events, _ = det.detect(G, REF_LEN)
        events = det.localize(G, events)
        events = det.type_events(G, events)
        for e in events:
            e["scale"] = det.w
        typed_all.extend(events)
    events = merge_typed(typed_all)
    pred = [(e["t0"] - REF_LEN, min(e["t1"] - REF_LEN, TEST_LEN), _to_short(e.get("type", "?"))) for e in events]
    res["rdsg"] = event_metrics(true_events, pred, TEST_LEN, SPP)
    res["rdsg"]["recall_by_type"] = recall_by_type(true_events, pred)
    res["rdsg"]["per_event"] = per_event_detail(true_events, pred)

    # ---- ablation: gamma-only (same dual scale) ----
    G2 = det_ab[0].granulate(np.vstack([X_ref, X_test_inj]))
    typed_ab = []
    for det in det_ab:
        ev_ab, _ = det.detect(G2, REF_LEN)
        ev_ab = det.localize(G2, ev_ab)
        ev_ab = det.type_events(G2, ev_ab)
        for e in ev_ab:
            e["scale"] = det.w
        typed_ab.extend(ev_ab)
    ev_ab = merge_typed(typed_ab)
    pred_ab = [(e["t0"] - REF_LEN, min(e["t1"] - REF_LEN, TEST_LEN), _to_short(e.get("type", "?"))) for e in ev_ab]
    res["gamma_only"] = event_metrics(true_events, pred_ab, TEST_LEN, SPP)
    res["gamma_only"]["recall_by_type"] = recall_by_type(true_events, pred_ab)
    res["gamma_only"]["per_event"] = per_event_detail(true_events, pred_ab)

    # ---- B4: IF / LOF aggregation ----
    for model in ("if", "lof"):
        alarms, _ = point_agg_scores(X_ref, X_test_inj, model=model, seed=seed)
        evs = alarms_to_events(alarms, gap=2 * BLK)
        pred_b = [(int(a), int(b), None) for a, b in evs]
        res[f"b4_{model}"] = event_metrics(true_events, pred_b, TEST_LEN, SPP)
        res[f"b4_{model}"]["recall_by_type"] = recall_by_type(true_events, pred_b)

    # ---- B5a: MI graph ----
    mig = MIGraphDetector(K=K, w=W, blk=BLK)
    edges = build_pair_edges(det_full[0].channels_, taus)
    mig.fit(X_ref, edges)
    evs_mi, _ = mig.detect(G, REF_LEN)
    pred_mi = [(int(a - REF_LEN), int(b - REF_LEN), None) for a, b in evs_mi]
    res["b5a_mi"] = event_metrics(true_events, pred_mi, TEST_LEN, SPP)
    res["b5a_mi"]["recall_by_type"] = recall_by_type(true_events, pred_mi)

    print(f"[seed {seed}] ours R={res['rdsg']['recall']:.2f} "
          f"({res['rdsg']['recall_by_type']}) fa/d={res['rdsg']['fa_per_day']:.2f} "
          f"type={res['rdsg']['type_acc']:.2f} | γonly R={res['gamma_only']['recall']:.2f} "
          f"({res['gamma_only']['recall_by_type']}) | IF R={res['b4_if']['recall']:.2f} "
          f"| LOF R={res['b4_lof']['recall']:.2f} | MI R={res['b5a_mi']['recall']:.2f} "
          f"| {time.time()-t0:.0f}s", flush=True)
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    Xs = load_data()
    taus = list(range(-3, 4))
    # calibrate once (reference segment invariant across seeds)
    t0 = time.time()
    det_full, det_ab = [], []
    for w in WS:
        d = RDSGDetector(K=K, w=w, blk=min(BLK, w // 2), B=B, q=0.05, seed=1)
        d.fit(Xs[:REF_LEN], channels=list(range(Xs.shape[1])), taus=taus)
        det_full.append(d)
        d2 = RDSGDetector(K=K, w=w, blk=min(BLK, w // 2), B=B, q=0.05, seed=1,
                          components=("dgamma",))
        d2.fit(Xs[:REF_LEN], channels=list(range(Xs.shape[1])), taus=taus)
        det_ab.append(d2)
    G_cal = det_full[0].granulate(Xs[:REF_LEN])
    for i, (d, d2) in enumerate(zip(det_full, det_ab)):
        print(f"calibrating ours w={d.w} ...", flush=True)
        d.calibrate(G_cal, REF_LEN, verbose=True, cal_span=CAL_SPAN)
        print(f"calibrating gamma-only w={d2.w} ...", flush=True)
        d2.calibrate(G_cal, REF_LEN, verbose=True, cal_span=CAL_SPAN)
    print(f"calibration done in {time.time()-t0:.0f}s", flush=True)

    all_res = {}
    for seed in range(1, 6):
        all_res[seed] = run_seed(Xs, seed, taus, det_full, det_ab)
        with open(os.path.join(OUT, "e0_raw.json"), "w", encoding="utf-8") as f:
            json.dump(all_res, f, ensure_ascii=False, indent=1, default=float)

    methods = ["rdsg", "gamma_only", "b4_if", "b4_lof", "b5a_mi"]
    summary = {}
    for m in methods:
        recalls = np.array([all_res[s][m]["recall"] for s in all_res])
        fas = np.array([all_res[s][m]["fa_per_day"] for s in all_res])
        rt = {t: float(np.mean([all_res[s][m]["recall_by_type"][t] for s in all_res]))
              for t in ("inv", "brk", "emb")}
        summary[m] = dict(recall_mean=float(recalls.mean()), recall_std=float(recalls.std()),
                          recall_by_type=rt,
                          fa_per_day_mean=float(fas.mean()),
                          delay_med=float(np.nanmedian([all_res[s][m]["delay_med"] for s in all_res])),
                          type_acc=float(np.mean([all_res[s][m]["type_acc"] for s in all_res])),
                          pa_f1=float(np.mean([all_res[s][m]["pa_f1"] for s in all_res])))

    from scipy.stats import wilcoxon
    ours_r = [all_res[s]["rdsg"]["recall"] for s in all_res]
    if_r = [all_res[s]["b4_if"]["recall"] for s in all_res]
    p = 1.0 if np.allclose(ours_r, if_r) else float(wilcoxon(ours_r, if_r).pvalue)

    rt = summary["rdsg"]["recall_by_type"]
    gate_recall = all(v >= 0.8 for v in rt.values())
    gate_fa = summary["rdsg"]["fa_per_day_mean"] <= 1.0
    gate_p = p < 0.05
    verdict = dict(gate_recall_by_type=bool(gate_recall), gate_fa=bool(gate_fa),
                   gate_wilcoxon=bool(gate_p), PASS=bool(gate_recall and gate_fa and gate_p),
                   wilcoxon_p_vs_b4if=p)
    out = dict(summary=summary, verdict=verdict, seeds=list(all_res.keys()),
               config=dict(N_CHANNELS=N_CHANNELS, REF_LEN=REF_LEN, TEST_LEN=TEST_LEN,
                           K=K, WS=WS, BLK=BLK, B=B, N_EACH=N_EACH, MARGIN=MARGIN, CAL_SPAN=CAL_SPAN))
    with open(os.path.join(OUT, "e0_summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)
    print(json.dumps(verdict, ensure_ascii=False))
    for m in methods:
        print(m, json.dumps(summary[m], ensure_ascii=False))


if __name__ == "__main__":
    main()
