# -*- coding: utf-8 -*-
"""Final check: macro values in the manuscript vs experiment JSONs."""
import json
import re
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.normpath(os.path.join(HERE, "..", "results"))
BS = chr(92)
pat = re.compile(re.escape(BS + "newcommand{" + BS) + r"(\w+)" + re.escape("}{") + r"(.+?)" + re.escape("}"))


def load(p):
    return json.load(open(os.path.join(RES, p), encoding="utf-8"))


macros = {}
for f in ("e0_macros", "e1_macros", "e2_macros", "e3_macros", "e7_macros"):
    for line in open(os.path.join(HERE, "gen", f + ".tex"), encoding="utf-8"):
        m = pat.match(line.strip())
        if m:
            macros[m.group(1)] = m.group(2)

e0 = load("e0/e0_syn2_summary.json")
e1 = load("paper/e1_gas.json")
e2m = load("paper/e2_metr_la.json")
e2b = load("paper/e2_beijing_air.json")
e2p = load("paper/e2_pems_bay.json")
pool = load("paper/pooled_by_type.json")

def seed_mean(d, method, sub):
    seeds = (d or {}).get(method) or {}
    vals = [v["recall_by_type"][sub] for v in seeds.values()
            if "recall_by_type" in v and v["recall_by_type"].get(sub) is not None]
    return round(float(np.mean(vals)), 2) if vals else None

def pooled_e0(m, sub):
    v = pool["e0"][m][sub]["pooled_recall"]
    return round(v, 2) if v is not None else None

def pooled_e2(ds, m, sub):
    v = pool["e2"][ds][m][sub]["pooled_recall"]
    return round(v, 2) if v is not None else None

checks = [
    ("eZeroRdsRecall", macros.get("eZeroRdsRecall"), round(e0["summary"]["rdsg"]["recall_mean"], 2)),
    ("eZeroRdsInv", macros.get("eZeroRdsInv"), pooled_e0("rdsg", "inv")),
    ("eZeroRdsFa", macros.get("eZeroRdsFa"), round(e0["summary"]["rdsg"]["fa_per_day_mean"], 2)),
    ("eZeroPval", macros.get("eZeroPval"), f"{e0['verdict']['wilcoxon_p']:.4g}"),
    ("eOneEvents", macros.get("eOneEvents"), str(e1["n_events"])),
    ("eTwoInv", macros.get("eTwoInv"), pooled_e2("metr_la", "rdsg", "inv")),
    ("eTwoBrk", macros.get("eTwoBrk"), pooled_e2("metr_la", "rdsg", "brk")),
]
bad = 0
for name, a, b in checks:
    try:
        ok = abs(float(a) - float(b)) < 5e-3 if ('.' in str(a) or '.' in str(b)) else str(a) == str(b)
    except (TypeError, ValueError):
        ok = str(a) == str(b)
    bad += (not ok)
    print(("OK " if ok else "BAD"), name, "tex=", a, "json=", b)

import numpy as np
for k, d in (("METR-LA", e2m), ("Beijing-Air", e2b), ("PEMS-BAY", e2p)):
    r = [v["recall"] for v in d["rdsg"].values() if "recall" in v]
    fa = [v["fa_per_day"] for v in d["rdsg"].values() if "fa_per_day" in v]
    print(f"{k}: ours R={np.mean(r):.2f} fa={np.mean(fa):.2f} (n={len(r)})")

print("\nRESULT:", "ALL CONSISTENT" if bad == 0 else f"{bad} MISMATCH")
