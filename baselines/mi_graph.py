# -*- coding: utf-8 -*-
"""B5a: mutual-information graph change baseline (added during review).

Sliding-window granular-domain mutual information matrices -> change statistic
= sum_e |Delta MI_e|; the threshold is the quantile of the same statistic over
adjacent non-overlapping window pairs inside the reference segment (consistent
calibration, never touches the test set).
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rdsg.granulation import Granulator
from rdsg.tables import edge_code, block_tables

_EPS = 1e-12


def _mi_from_table(tab: np.ndarray) -> np.ndarray:
    """tab (..., KA, K) counts -> MI (nats)."""
    P = tab / np.maximum(tab.sum(axis=(-2, -1), keepdims=True), _EPS)
    pg = P.sum(axis=-1, keepdims=True)
    pv = P.sum(axis=-2, keepdims=True)
    ratio = P / np.maximum(pg * pv, _EPS)
    return np.where(P > 0, P * np.log(ratio), 0.0).sum(axis=(-2, -1))


class MIGraphDetector:
    def __init__(self, K=4, w=512, blk=None, q=0.95, gap_blocks=2, step_blocks=1):
        self.K, self.w, self.q, self.gap_blocks, self.step_blocks = K, w, q, gap_blocks, step_blocks
        self.blk = blk if blk is not None else max(8, w // 8)

    def fit(self, X_ref, edges, taus=None):
        self.gran_ = Granulator(self.K).fit(X_ref)
        self.edges_ = edges
        return self

    def _tables(self, G, t0, t1):
        return block_tables(G, self.edges_, self.K, self.blk, t0=t0, t1=t1)

    def detect(self, G, t_ref_end):
        K, blk, half = self.K, self.blk, self.w // self.blk
        span0 = max(0, t_ref_end - self.w)
        tab = self._tables(G, span0, G.shape[0])
        nb = tab.shape[1]
        # Calibration within the reference segment: sum of |Delta MI| over
        # adjacent window pairs
        calib = []
        t = max(0, t_ref_end - 4 * self.w)
        while t + 2 * self.w <= t_ref_end:
            b0 = (t - span0) // blk
            T1 = tab[:, b0:b0 + half].sum(axis=1)
            T2 = tab[:, b0 + half:b0 + 2 * half].sum(axis=1)
            calib.append(np.abs(_mi_from_table(T2) - _mi_from_table(T1)).sum())
            t += self.w
        thr = np.quantile(calib, self.q) if calib else np.inf
        # Sliding-window detection
        k = (t_ref_end - span0) // blk
        alarms = []
        while k + half <= nb:
            T1 = tab[:, k - half:k].sum(axis=1)
            T2 = tab[:, k:k + half].sum(axis=1)
            stat = np.abs(_mi_from_table(T2) - _mi_from_table(T1)).sum()
            if stat > thr:
                alarms.append(span0 + k * blk)
            k += self.step_blocks
        events = []
        for t in alarms:
            if events and t - events[-1][1] <= self.gap_blocks * blk:
                events[-1] = (events[-1][0], t + blk)
            else:
                events.append((t, t + blk))
        return events, dict(threshold=float(thr))
