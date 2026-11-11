# -*- coding: utf-8 -*-
"""B4: point-detector aggregation baseline (per-channel IF / LOF + max
aggregation).

Theoretical prediction for the mechanism-specific experiments (Proposition 2):
under margin-preserving injections its recall is approximately its false-alarm
rate. The reference-segment threshold uses half-split cross-scoring (guards
against the optimistic bias of scoring the training segment by itself).
"""
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


def _fit_score(model, x_train, x_score, seed):
    if model == "if":
        m = IsolationForest(n_estimators=200, random_state=seed, contamination="auto")
    elif model == "lof":
        m = LocalOutlierFactor(n_neighbors=20, novelty=True)
    else:
        raise ValueError(model)
    m.fit(x_train)
    return -m.score_samples(x_score)


def point_agg_scores(X_ref, X_test, model="if", qthr=0.995, seed=0):
    """Train a point detector per channel -> score test points -> max-aggregate
    across channels -> set the threshold by a reference-segment quantile."""
    n = X_ref.shape[1]
    T_ref = X_ref.shape[0]
    agg = np.full(X_test.shape[0], -np.inf)
    for j in range(n):
        agg = np.maximum(agg, _fit_score(model, X_ref[:, j:j + 1], X_test[:, j:j + 1], seed))

    half = T_ref // 2
    ref_agg = np.full(T_ref, -np.inf)
    for j in range(n):
        # second half trains, scores first half; first half trains, scores second half
        ref_agg[:half] = np.maximum(ref_agg[:half],
                                    _fit_score(model, X_ref[half:, j:j + 1], X_ref[:half, j:j + 1], seed))
        ref_agg[half:] = np.maximum(ref_agg[half:],
                                    _fit_score(model, X_ref[:half, j:j + 1], X_ref[half:, j:j + 1], seed))
    thr = np.quantile(ref_agg, qthr)
    alarms = agg > thr
    return alarms, agg


def alarms_to_events(alarms, gap=64):
    idx = np.where(alarms)[0]
    if len(idx) == 0:
        return []
    events = []
    t0 = prev = idx[0]
    for t in idx[1:]:
        if t - prev > gap:
            events.append((t0, prev + 1))
            t0 = t
        prev = t
    events.append((t0, prev + 1))
    return events
