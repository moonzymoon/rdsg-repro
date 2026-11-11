# -*- coding: utf-8 -*-
"""W2+W4: component-level alarm attribution + component-triggered event typing
(residual attack-surface cleanup plan).

Re-run the official detection pipeline on Beijing-Air and METR-LA with 10 seeds
each (same configuration and protocol as e2), instrumented to record the
triggering component of every alarm boundary (which family, dgamma/ddelta,
crosses the threshold). The event-level triggering component is the mode of the
components over the boundaries inside the event window.

W4 typing rule (pre-registered):
  R-channel events: dgamma dominant -> breakdown; ddelta dominant -> inversion
  E-channel events: emergence (emb/emb2 not distinguished)
Types compared against ground truth -> component-triggered type accuracy.
Branches: (a) gated type_acc >= 0.6 -> typing upgraded to functional auxiliary;
      (b) otherwise -> keep the auxiliary framing and report this experiment
      honestly.
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
from rdsg import inject
from rdsg.datasets import load_metr_la, load_beijing_air
from rdsg.metrics import match_events
from run_e0_syn2 import _unit_stack, pick_theta, BLK
from run_paper_e2 import build_channel, make_specs, PAIR_STRATA_E, LEN_BY_SPP
from run_e0_syn2 import channel_scan

OUT = os.path.normpath(os.path.join(_results_root(), "paper"))
LOADERS = {"beijing_air": load_beijing_air, "metr_la": load_metr_la}


def instrumented_scan(dd, Gcal, Gs, scan_start, scan_end, selfscan_start, spp, gap):
    """Instrumented version of channel_scan: additionally returns
    (per-boundary triggering components, thetas)."""
    for d in dd:
        d.calibrate(Gcal[id(d)], 0 if False else None, cal_t0=0) if False else None
    # Reuse channel_scan's calibration/pruning/thresholding: copy its
    # implementation directly and add per-component tracking
    for d in dd:
        d.calibrate(Gcal[id(d)], scan_start, cal_t0=0)
        d.prune_units_by_scan(Gcal[id(d)], selfscan_start, scan_start, z_cap=5.5)
    comps = sorted(set(dd[0].components).intersection(*[set(d.components) for d in dd]))
    fa_c = 0.25 / len(comps)
    grid_r, Zr = _unit_stack(dd, Gcal, selfscan_start, scan_start)
    thetas = {}
    for c in comps:
        Tc = Zr[:, :, comps.index(c)].max(axis=1)
        thetas[c], _ = pick_theta(Tc, grid_r, fa_c, spp, gap=gap)
    grid_q, Zq = _unit_stack(dd, Gs, scan_start, scan_end)
    trig = np.zeros((len(grid_q), len(comps)), bool)
    alarm = np.zeros(len(grid_q), bool)
    for ci, c in enumerate(comps):
        Tc_q = Zq[:, :, comps.index(c)].max(axis=1)
        hit = Tc_q > thetas[c]
        trig[:, ci] = hit
        alarm |= hit
    evs = _merge_alarms_local(grid_q, alarm, gap)
    return evs, thetas, comps, trig, grid_q


def _merge_alarms(times, flags, gap):
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


_merge_alarms_local = _merge_alarms


def main():
    taus = (0, 1, -1)
    out_path = os.path.join(OUT, "e2d_attr.json")
    res = json.load(open(out_path, encoding="utf-8")) if os.path.exists(out_path) else {}
    for key, loader in LOADERS.items():
        X, spp, meta = loader()
        X_ref, X_test = X[:12000], X[12000:12000 + 18000]
        lengths = LEN_BY_SPP[spp]
        planner = build_channel(X_ref, taus, strata=(16, 24))[0]
        force_R, force_E = planner.strong_pairs_, planner.weak_pairs_
        dE0 = build_channel(X_ref, taus, strata=PAIR_STRATA_E, n_subset=6,
                            force_pairs=(force_R, force_E))[0]
        res.setdefault(key, {"seeds": {}})
        n_correct = n_tot = 0
        by_rule = {}
        for seed in range(1, 11):
            if str(seed) in res[key]["seeds"]:
                continue
            t0 = time.time()
            specs = make_specs(planner, dE0, X_ref, X_test, seed, lengths)
            X_inj = inject(X_test, specs, seed=seed)
            true_events = [(sp["start"], sp["start"] + sp["length"], sp["type"]) for sp in specs]
            # Dual-channel instrumentation
            dR = build_channel(X_ref, taus, strata=(16, 0))
            dE = build_channel(X_ref, taus, strata=(0, 24), n_subset=6,
                               force_pairs=(force_R, force_E))
            Gcal_R = {id(d): d.granulate(X_ref) for d in dR}
            Gcal_E = {id(d): d.granulate(X_ref) for d in dE}
            Gs_R = {id(d): d.granulate(np.vstack([X_ref, X_inj])) for d in dR}
            Gs_E = {id(d): d.granulate(np.vstack([X_ref, X_inj])) for d in dE}
            evs_R, thR, compsR, trigR, gridR = instrumented_scan(
                dR, Gcal_R, Gs_R, len(X_ref), len(X_ref) + len(X_test), 0, spp, 4 * 64)
            evs_E, thE, compsE, trigE, gridE = instrumented_scan(
                dE, Gcal_E, Gs_E, len(X_ref), len(X_ref) + len(X_test), 0, spp, 4 * 64)
            # Event-level triggering component
            def ev_components(evs, trig, grid, comps):
                out = []
                for (a, b) in evs:
                    m = (grid >= a) & (grid < b)
                    counts = trig[m].sum(axis=0)
                    out.append({c: int(counts[i]) for i, c in enumerate(comps)})
                return out
            compR = ev_components(evs_R, trigR, gridR, compsR)
            compE = ev_components(evs_E, trigE, gridE, compsE)
            # Type assignment
            true3 = [(a, b, None) for a, b, _ in true_events]
            tidx = {(int(a), int(b)): k for k, (a, b, _t) in enumerate(true_events)}
            preds = []
            for (a, b), cc in zip([(x, y) for x, y in evs_R], compR):
                dom = max(cc, key=cc.get) if cc else "dgamma"
                preds.append((int(a) - len(X_ref), int(b) - len(X_ref),
                              "brk" if dom == "dgamma" else "inv"))
            for (a, b), cc in zip([(x, y) for x, y in evs_E], compE):
                preds.append((int(a) - len(X_ref), int(b) - len(X_ref), "emb"))
            matches, _, _ = match_events(true3, [(p[0], p[1], None) for p in preds])
            ptype = {}
            for pi, (a, b, t) in enumerate(preds):
                ptype[(a, b)] = t
            seed_rows = []
            for tr, p in matches:
                k = tidx[(int(tr[0]), int(tr[1]))]
                truth = true_events[k][2]
                if p is None:
                    seed_rows.append(dict(idx=k, truth=truth, pred=None))
                else:
                    pt = ptype[(int(p[0]), int(p[1]))]
                    ok = (pt == truth) or (truth == "emb2" and pt == "emb")
                    seed_rows.append(dict(idx=k, truth=truth, pred=pt, ok=bool(ok)))
                    n_tot += 1
                    n_correct += int(ok)
                    by_rule.setdefault(truth, [0, 0])
                    by_rule[truth][1] += 1
                    by_rule[truth][0] += int(ok)
            res[key]["seeds"][str(seed)] = dict(events=seed_rows,
                                                thetas=dict(thR, **{"E:" + k: v for k, v in thE.items()}))
            print(f"[{key} s{seed}] events={len(seed_rows)} typed_hit={sum(1 for r in seed_rows if r.get('ok'))} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            json.dump(res, open(out_path, "w", encoding="utf-8"), indent=1, default=float)
        acc = n_correct / max(n_tot, 1)
        res[key]["type_acc_component"] = round(acc, 3)
        res[key]["by_truth"] = {k: dict(acc=round(v[0] / max(v[1], 1), 3), n=v[1])
                                for k, v in by_rule.items()}
        print(f"== {key}: component-triggered type_acc={acc:.2f} (n={n_tot}) {res[key]['by_truth']}")
    json.dump(res, open(out_path, "w", encoding="utf-8"), indent=1, default=float)
    print("saved e2d_attr.json")


if __name__ == "__main__":
    main()
