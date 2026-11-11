# -*- coding: utf-8 -*-
"""WADI A2 loader for RDSG pipeline.

Uses the pre-processed npy from WADI A2 (Nov 2019 version).
93 channels, 784k reference points (14 days), 173k test points (2 days).
Binary attack labels. Sampling 1s.

The reference is subsampled to match our pipeline's expected scale
(~12k points) by taking the last contiguous block.
"""
import os
import numpy as np


def load_wadi(ref_points=12000, test_points=172801,
              max_channels=40, npy_dir=None):
    """Load WADI A2 from pre-processed npy files.

    Returns (X, spp, meta): X (T, n) with reference + test concatenated,
    spp seconds per point, meta with events list.
    """
    if npy_dir is None:
        npy_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "..", "..", "..", "..", "数据", "TSAD_数据集")
        # Optional pre-processed copy (ref.npy/test.npy/test_label.npy derived
        # from the official WADI A2 release): set $WADI_NPY_DIR to use it.
        alt = os.environ.get("WADI_NPY_DIR")
        if alt and os.path.isdir(alt):
            npy_dir = alt

    ref = np.load(os.path.join(npy_dir, "ref.npy"), allow_pickle=True)
    test = np.load(os.path.join(npy_dir, "test.npy"), allow_pickle=True)
    label = np.load(os.path.join(npy_dir, "test_label.npy"), allow_pickle=True).ravel()

    # Channel selection: top-N by reference variance (consistent with MI baseline)
    variances = np.nanvar(ref, axis=0)
    # Remove near-constant channels
    ok = variances > np.percentile(variances, 10)
    ok_idx = np.where(ok)[0]
    # Rank by variance, take top max_channels
    ranked = ok_idx[np.argsort(-variances[ok_idx])][:max_channels]
    ranked = np.sort(ranked)

    ref_sel = ref[:, ranked]
    test_sel = test[:, ranked]

    # Handle NaN
    from scipy.interpolate import interp1d
    for arr in (ref_sel, test_sel):
        for j in range(arr.shape[1]):
            col = arr[:, j]
            bad = np.isnan(col)
            if bad.any():
                good = ~bad
                if good.sum() > 2:
                    f = interp1d(np.where(good)[0], col[good],
                                 bounds_error=False, fill_value="extrapolate")
                    col[bad] = f(np.where(bad)[0])
                else:
                    col[bad] = 0

    # Take last ref_points of reference (most recent, closest to test conditions)
    if ref_sel.shape[0] > ref_points:
        ref_sel = ref_sel[-ref_points:]
    X = np.vstack([ref_sel, test_sel])

    # Extract attack events from binary label
    events = []
    in_attack = False
    start = 0
    for i in range(len(label)):
        if label[i] == 1 and not in_attack:
            in_attack = True
            start = i
        elif label[i] == 0 and in_attack:
            in_attack = False
            events.append((start, i, "attack"))
    if in_attack:
        events.append((start, len(label), "attack"))

    meta = {"name": "WADI", "channels": len(ranked),
            "rows": X.shape[0], "n_ref": ref_sel.shape[0],
            "n_test": test_sel.shape[0], "events": events,
            "n_events": len(events), "source": "WADI.A2"}

    return X, 1.0, meta


if __name__ == "__main__":
    X, spp, meta = load_wadi()
    print(f"X: {X.shape}, spp={spp}")
    print(f"meta: {meta['name']}, {meta['channels']} ch, "
          f"ref={meta['n_ref']}, test={meta['n_test']}, "
          f"events={meta['n_events']}")
    for e in meta["events"][:10]:
        print(f"  [{e[0]:>7}, {e[1]:>7}) len={e[1]-e[0]:>5} type={e[2]}")
