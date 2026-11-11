# -*- coding: utf-8 -*-
"""Block-level contingency tables: the unified foundation for tuple edge statistics and
the block permutation test.

Core efficiency design: each edge's K^|A| × K contingency table on each time block is
computed only once; the statistic of any window = the sum of the block tables it
contains; a permutation = a reordering and summation of block tables.
"""
import numpy as np
from typing import List, Tuple

Edge = Tuple[Tuple[int, ...], int, int]  # (A: tuple of parent channels, d: target channel, tau: lag)


def edge_code(G: np.ndarray, A, d: int, tau: int, K: int) -> np.ndarray:
    """Returns a length-T code sequence (invalid positions are -1):
    code = grain_A(t-tau) * K + grain_d(t). When |A|=1, grain_A is a single-channel
    granule; when |A|=2, grain_A = g1*K + g2 (table shape K^2 × K).
    """
    T = G.shape[0]
    code = np.full(T, -1, dtype=np.int64)
    lo = max(0, tau)
    hi = min(T, T + tau)
    if hi <= lo:
        return code
    gA = G[lo - tau: hi - tau, A[0]].astype(np.int64)
    for a in A[1:]:
        gA = gA * K + G[lo - tau: hi - tau, a].astype(np.int64)
    code[lo:hi] = gA * K + G[lo:hi, d].astype(np.int64)
    return code


def block_tables(G: np.ndarray, edges: List[Edge], K: int, blk: int,
                 t0: int = 0, t1: int = None) -> np.ndarray:
    """Computes block contingency tables for the edge set over [t0, t1) on a fixed block
    grid (start aligned to t0).

    Returns (E, nb, KA, K) counts with KA = K^|A| (all edges share the table shape of the
    largest |A|; edges with smaller |A| use only the first K^|A| rows).
    For uniform memory use and vectorization, the caller should keep |A| consistent
    across edges (call in batches).
    """
    if t1 is None:
        t1 = G.shape[0]
    T = t1 - t0
    nb = (T + blk - 1) // blk  # ceiling division: the trailing incomplete block is kept (slightly smaller table mass, harmless)
    amax = max(len(e[0]) for e in edges)
    KA = K ** amax
    tab = np.zeros((len(edges), nb, KA, K), dtype=np.float32)
    blocks = np.arange(T) // blk
    for ei, (A, d, tau) in enumerate(edges):
        code = edge_code(G[t0:t1], A, d, tau, K)
        valid = code >= 0
        flat = blocks[valid] * (KA * K) + code[valid]
        cnt = np.bincount(flat, minlength=nb * KA * K)
        tab[ei] = cnt.reshape(nb, KA, K)
    return tab


def window_sum(tab: np.ndarray, idx) -> np.ndarray:
    """Sums the block tables over idx (an array of block indices) into window tables.
    tab: (E, nb, KA, K) -> (E, KA, K)."""
    return tab[:, idx, :, :].sum(axis=1)
