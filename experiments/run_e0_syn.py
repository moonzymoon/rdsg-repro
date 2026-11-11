# -*- coding: utf-8 -*-
"""E0 capability gate experiment (synthetic VAR version) -- capability gate for the stop-condition criterion.

Finding (2026-09-26, see results/e0/e0_metr_la_*.json and the failure-mode analysis):
all METR-LA channel pairs are bound by a common daily cycle (gamma_soft in [0.53, 0.62]); no genuinely
weak pairs exist, so EMB (emergence) injections cannot form genuine emergence events; moreover, with
K=4 granules the zero-noise spread of the modal mapping is sigma_rho ~ 0.25, and the reachable
in-window signal z peaks at 2.5-5 < the full-edge-family maxT threshold. -> METR-LA can only serve as
a real-data stress test.

This script: a synthetic VAR(1) system (designed strong/moderate/independent channel groups, so weak
pairs genuinely exist) + the same injection protocol + the same gate criterion, serving as the
capability gate for the method. Multiple granulation scales K in {4, 2}.
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
from rdsg.tables import edge_code
from rdsg.stats import gamma_soft, std_gamma
from rdsg.metrics import match_events
from baselines.point_agg import point_agg_scores, alarms_to_events
from baselines.mi_graph import MIGraphDetector
from rdsg.detector import build_pair_edges

OUT = os.path.normpath(os.path.join(_results_root(), "e0"))
SPP = 60.0            # synthetic 1-minute sampling
N_CHANNELS = 12
REF_LEN, TEST_LEN = 12000, 8000
BLK, B = 64, 499
KS = (4, 2)           # multiple granulation scales
WS = (512, 128)       # dual detection windows
N_EACH, MARGIN = 10, 700
CAL_SPAN = 8000
TYPE_MAP = {"inversion": "inv", "breakdown": "brk", "emergence": "emb"}


def make_synthetic(T, seed=0):
    """Factor-structure synthetic system (v2, redesigned 2026-09-27):

    The old VAR put all dependence in the lag terms, leaving tau=0 channel pairs nearly independent
    (strong pairs gamma ~ 0.30 vs weak pairs 0.27 = noise-level ordering); BRK had nothing to break
    and the gamma blind spot could not be established. v2 uses within-group common factors to create
    genuine tau=0 structure:
    - Group A ch0-2: 0.8*F1 + 1.0 eps -> within-group correlation ~ +0.86 (breakdown/inversion hosts)
    - Group B ch3-4: +/-0.75*F2 + 1.0 eps -> within-group negative correlation ~ -0.85 (negative-phase strong pairs)
    - Group C ch5-7: 0.6*F3 + 1.0 eps -> within-group correlation ~ +0.68 (moderate)
    - Group D ch8-11: pure eps -> independent (emergence-injection hosts, gamma ~ floor 0.27)
    Factor AR(1) with phi=0.95 keeps the autocorrelation realistic.
    """
    rng = np.random.default_rng(seed)
    n = N_CHANNELS
    F = np.zeros((T, 4))
    for t in range(1, T):
        F[t] = 0.95 * F[t - 1] + rng.standard_normal(4)
    eps = rng.standard_normal((T, n))
    load = np.zeros(n)
    fac = np.zeros(n, dtype=int)
    load[0:3], fac[0:3] = 0.8, 0
    load[3], fac[3] = 0.75, 1
    load[4], fac[4] = -0.75, 1        # opposite sign -> within-group negative correlation
    load[5:8], fac[5:8] = 0.6, 2
    load[8:12], fac[8:12] = 0.0, 3
    noise = np.array([1.0]*3 + [1.0]*2 + [1.0]*3 + [1.0]*4)  # noise scaled up to match the factor-AR amplification
    X = F[:, fac] * load[None, :] + eps * noise[None, :]
    return X


def rank_pairs_by_ref_gamma(X_ref, channels, K, w):
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
            gam[(i, j)] = float(std_gamma(tab[None])[0])
    vals = np.array(sorted(gam.values()))
    hi, lo = np.quantile(vals, 0.75), np.quantile(vals, 0.25)
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
            prev["type"] = fine.get("type", "?") if fine.get("type") not in (None, "untyped") \
                else (coarse.get("type") or prev.get("type", "?"))
        else:
            out.append(e)
    return out


def per_event_detail(true_events, pred_events):
    matches, _, _ = match_events(true_events, [(p[0], p[1]) for p in pred_events])
    return [[int(t0), int(t1 - t0), typ, bool(p is not None)]
            for (t0, t1, typ), p in matches]


def build_detectors(X_ref, taus, components=None):
    dets = []
    for K in KS:
        for w in WS:
            d = RDSGDetector(K=K, w=w, blk=min(BLK, w // 2), B=B, q=0.05, seed=1,
                             components=components)
            d.fit(X_ref, channels=list(range(X_ref.shape[1])), taus=taus)
            dets.append(d)
    return dets


def run_config(Xs, dets, seed, label, results):
    t0 = time.time()
    X_ref, X_test = Xs[:REF_LEN], Xs[REF_LEN:]
    # injection-pair stratification uses the main granule and main window (K=4, w=512)
    strong, weak, gam = rank_pairs_by_ref_gamma(X_ref, list(range(Xs.shape[1])), 4, 512)
    assert len(strong) >= 2 * N_EACH, f"insufficient strong pairs: {len(strong)}"
    assert len(weak) >= N_EACH, f"insufficient weak pairs: {len(weak)}"
    specs = sample_specs(None, strong, weak, n_each=N_EACH,
                         lengths=(30, 60, 120), t_len=TEST_LEN, margin=MARGIN, seed=seed)
    X_test_inj = inject(X_test, specs, seed=seed)
    true_events = [(sp["start"], sp["start"] + sp["length"], sp["type"]) for sp in specs]
    Xfull = np.vstack([X_ref, X_test_inj])
    typed_all = []
    for det in dets:
        G = det.granulate(Xfull)  # each detector's own K
        events, _ = det.detect(G, REF_LEN)
        events = det.localize(G, events)
        events = det.type_events(G, events)
        for e in events:
            e["scale"] = det.w
        typed_all.extend(events)
    events = merge_typed(typed_all)
    pred = [(e["t0"] - REF_LEN, min(e["t1"] - REF_LEN, TEST_LEN), _to_short(e.get("type", "?")))
            for e in events]
    m = event_metrics(true_events, pred, TEST_LEN, SPP)
    m["recall_by_type"] = recall_by_type(true_events, pred)
    m["per_event"] = per_event_detail(true_events, pred)
    results[seed] = m
    print(f"[{label} seed {seed}] R={m['recall']:.2f} ({m['recall_by_type']}) "
          f"fa/d={m['fa_per_day']:.2f} type={m['type_acc']:.2f} delay={m['delay_med']:.0f} "
          f"| {time.time()-t0:.0f}s", flush=True)
    return results


def run_seed_full(Xs, seed, taus, det_full, det_ab):
    results = {}
    run_config(Xs, det_full, seed, "ours", results.setdefault("rdsg", {}))
    run_config(Xs, det_ab, seed, "γonly", results.setdefault("gamma_only", {}))
    # B4
    X_ref, X_test = Xs[:REF_LEN], Xs[REF_LEN:]
    strong, weak, _ = rank_pairs_by_ref_gamma(X_ref, list(range(Xs.shape[1])), 4, 512)
    specs = sample_specs(None, strong, weak, n_each=N_EACH,
                         lengths=(30, 60, 120), t_len=TEST_LEN, margin=MARGIN, seed=seed)
    X_test_inj = inject(X_test, specs, seed=seed)
    true_events = [(sp["start"], sp["start"] + sp["length"], sp["type"]) for sp in specs]
    for model in ("if", "lof"):
        alarms, _ = point_agg_scores(X_ref, X_test_inj, model=model, seed=seed)
        evs = alarms_to_events(alarms, gap=2 * BLK)
        pred_b = [(int(a), int(b), None) for a, b in evs]
        m = event_metrics(true_events, pred_b, TEST_LEN, SPP)
        m["recall_by_type"] = recall_by_type(true_events, pred_b)
        results[f"b4_{model}"] = {seed: m}
        print(f"[b4_{model} seed {seed}] R={m['recall']:.2f} ({m['recall_by_type']}) "
              f"fa/d={m['fa_per_day']:.2f}", flush=True)
    # B5a MI
    mig = MIGraphDetector(K=4, w=512, blk=BLK)
    mig.fit(X_ref, build_pair_edges(det_full[0].channels_, taus))
    G = det_full[0].granulate(np.vstack([X_ref, X_test_inj]))
    evs_mi, _ = mig.detect(G, REF_LEN)
    pred_mi = [(int(a - REF_LEN), int(b - REF_LEN), None) for a, b in evs_mi]
    m = event_metrics(true_events, pred_mi, TEST_LEN, SPP)
    m["recall_by_type"] = recall_by_type(true_events, pred_mi)
    results["b5a_mi"] = {seed: m}
    print(f"[b5a_mi seed {seed}] R={m['recall']:.2f} ({m['recall_by_type']}) "
          f"fa/d={m['fa_per_day']:.2f}", flush=True)
    return results


def main():
    os.makedirs(OUT, exist_ok=True)
    taus = list(range(-3, 4))
    raw_path = os.path.join(OUT, "e0_syn_raw.json")
    all_res = {m: {} for m in ("rdsg", "gamma_only", "b4_if", "b4_lof", "b5a_mi")}
    if os.path.exists(raw_path):
        all_res.update(json.load(open(raw_path, encoding="utf-8")))
    for seed in range(1, 6):
        if seed in all_res["rdsg"]:
            continue
        Xs = make_synthetic(REF_LEN + TEST_LEN, seed=100 + seed)
        t0 = time.time()
        det_full = build_detectors(Xs[:REF_LEN], taus)
        det_ab = build_detectors(Xs[:REF_LEN], taus, components=("dgamma",))
        for d in det_full + det_ab:
            Gd = d.granulate(Xs[:REF_LEN])  # granulate with each detector's own K
            d.calibrate(Gd, REF_LEN, verbose=False, cal_span=CAL_SPAN)
        print(f"seed {seed}: detectors calibrated ({time.time()-t0:.0f}s), "
              f"thr={[(d.K, d.w, round(d.alarm_thr_,2)) for d in det_full]}", flush=True)
        r = run_seed_full(Xs, seed, taus, det_full, det_ab)
        for mname, per_seed in r.items():
            all_res[mname].update(per_seed)
        json.dump(all_res, open(raw_path, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1, default=float)

    # summary + gate
    summary = {}
    for m, per_seed in all_res.items():
        if not per_seed:
            continue
        recalls = [v["recall"] for v in per_seed.values()]
        fas = [v["fa_per_day"] for v in per_seed.values()]
        rt = {t: float(np.mean([v["recall_by_type"][t] for v in per_seed.values()]))
              for t in ("inv", "brk", "emb")}
        summary[m] = dict(recall_mean=float(np.mean(recalls)), fa_per_day_mean=float(np.mean(fas)),
                          recall_by_type=rt,
                          delay_med=float(np.nanmedian([v["delay_med"] for v in per_seed.values()])),
                          type_acc=float(np.mean([v["type_acc"] for v in per_seed.values()])),
                          pa_f1=float(np.mean([v["pa_f1"] for v in per_seed.values()])))
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
               config=dict(N_CHANNELS=N_CHANNELS, REF_LEN=REF_LEN, TEST_LEN=TEST_LEN,
                           KS=KS, WS=WS, BLK=BLK, B=B, N_EACH=N_EACH, MARGIN=MARGIN,
                           CAL_SPAN=CAL_SPAN, SPP=SPP))
    json.dump(out, open(os.path.join(OUT, "e0_syn_summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=float)
    print(json.dumps(verdict, ensure_ascii=False))
    for m, s in summary.items():
        print(m, json.dumps(s, ensure_ascii=False))


if __name__ == "__main__":
    main()
