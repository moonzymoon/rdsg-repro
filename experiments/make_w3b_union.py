# -*- coding: utf-8 -*-
"""Regenerate w3b_union.json from e1c per-attack matched sets at <=5 FA/day."""
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

import os, json

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.normpath(os.path.join(HERE, ".."))
RP = os.path.join(_results_root(), "paper")
e1c = json.load(open(os.path.join(RP, "e1c_gas_subset.json"), encoding="utf-8"))
n_ev = e1c["n_events"]
picked = {}
for m, pts in e1c["curves"].items():
    ok = [p for p in pts if p["fa_per_day"] <= 5.0]
    best = max(ok, key=lambda p: p["recall"]) if ok else None
    picked[m] = set(best["matched"]) if best else set()
out = {"singles": {m: {"recall": round(len(h) / n_ev, 3), "n": len(h)}
                   for m, h in picked.items()},
       "unions": {}}
for name, ms in [("rdsg+b5b_pcorr", ["rdsg", "b5b_pcorr"]),
                 ("rdsg+b4_lof", ["rdsg", "b4_lof"]),
                 ("rdsg+b4_lof+b4_if", ["rdsg", "b4_lof", "b4_if"]),
                 ("rdsg+b5b_pcorr+b4_lof", ["rdsg", "b5b_pcorr", "b4_lof"])]:
    u = set().union(*(picked[m] for m in ms))
    out["unions"][name] = {"recall": round(len(u) / n_ev, 3), "n": len(u)}
json.dump(out, open(os.path.join(RP, "w3b_union.json"), "w"), indent=1)
print("singles:", {m: v["recall"] for m, v in out["singles"].items()})
print("unions:", {k: v["recall"] for k, v in out["unions"].items()})
