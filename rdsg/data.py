# -*- coding: utf-8 -*-
"""Data loading utilities (direct h5py reads, bypassing pandas3/pytables compatibility issues with legacy h5 files)."""
import numpy as np
import h5py
import pandas as pd


def load_metr_like_h5(path: str) -> pd.DataFrame:
    """Read a DCRNN-format h5 file (df/axis1, df/block0_values, df/block0_items)."""
    with h5py.File(path, "r") as f:
        g = f["df"]
        idx = g["axis1"][:]
        vals = g["block0_values"][:]
        cols = [c.decode() if isinstance(c, bytes) else str(c) for c in g["block0_items"][:]]
    return pd.DataFrame(vals, index=pd.to_datetime(idx, unit="s"), columns=cols)
