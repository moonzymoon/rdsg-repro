# -*- coding: utf-8 -*-
"""The four primary event-level metrics (Reviewer Comment 3) plus the secondary
point-adjusted F1."""
import numpy as np


def _overlap(a0, a1, b0, b1):
    return max(a0, b0) < min(a1, b1)


def match_events(true_events, pred_events):
    """Greedy matching (sorted by start point). true/pred: [(t0,t1,type?)]. Returns matches, misses, falses, delays."""
    matches, delays = [], []
    used = np.zeros(len(pred_events), bool)
    for tr in sorted(true_events, key=lambda e: e[0]):
        t0, t1 = tr[0], tr[1]
        cand = [p for p in range(len(pred_events))
                if not used[p] and _overlap(pred_events[p][0], pred_events[p][1], t0, t1)]
        if cand:
            p = min(cand, key=lambda p: pred_events[p][0])
            used[p] = True
            matches.append((tr, pred_events[p]))
            delays.append(max(0, pred_events[p][0] - t0))
        else:
            matches.append((tr, None))
    falses = [pred_events[p] for p in range(len(pred_events)) if not used[p]]
    return matches, delays, falses


def event_metrics(true_events, pred_events, test_len_points, seconds_per_point):
    """Returns a dict: event detection rate (recall), false alarms per day, median detection
    delay, type accuracy, and pa_f1."""
    matches, delays, falses = match_events(true_events, pred_events)
    n_true = len(true_events)
    recall = sum(1 for _, p in matches if p is not None) / max(n_true, 1)
    days = test_len_points * seconds_per_point / 86400.0
    fa_per_day = len(falses) / max(days, 1e-9)
    delay_med = float(np.median(delays)) if delays else np.nan
    typed_ok = 0
    typed_tot = 0
    for tr, p in matches:
        if p is None or len(tr) < 3 or len(p) < 3:
            continue
        typed_tot += 1
        if tr[2] == p[2]:
            typed_ok += 1
    type_acc = typed_ok / max(typed_tot, 1)
    # point-adjusted F1 (secondary)
    pa_f1 = _point_adjusted_f1(true_events, pred_events, test_len_points)
    return dict(recall=recall, n_true=n_true, n_pred=len(pred_events),
                fa_per_day=fa_per_day, delay_med=delay_med,
                type_acc=type_acc, pa_f1=pa_f1)


def _point_adjusted_f1(true_events, pred_events, T):
    y = np.zeros(T, np.int8)
    for t0, t1, *rest in true_events:
        y[t0:t1] = 1
    if not pred_events:
        return 0.0
    yhat = np.zeros(T, np.int8)
    for t0, t1, *rest in pred_events:
        yhat[t0:t1] = 1
    # point adjustment
    for t0, t1, *rest in true_events:
        seg = yhat[t0:t1]
        if seg.any():
            yhat[t0:t1] = 1
    tp = int(((y == 1) & (yhat == 1)).sum())
    fp = int(((y == 0) & (yhat == 1)).sum())
    fn = int(((y == 1) & (yhat == 0)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return 2 * prec * rec / max(prec + rec, 1e-9)
