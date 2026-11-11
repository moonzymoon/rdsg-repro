# -*- coding: utf-8 -*-
"""Equal-frequency granulation: quantiles are estimated only from the reference segment and
frozen for use in all windows (leakage-prevention discipline)."""
import numpy as np


class Granulator:
    """K-granule equal-frequency granulation. fit runs on the reference segment; transform
    uses the frozen boundaries."""

    def __init__(self, K: int = 4):
        assert K >= 2
        self.K = K

    def fit(self, X_ref: np.ndarray):
        # X_ref: (T, n) float
        qs = np.linspace(0.0, 1.0, self.K + 1)[1:-1]  # K-1 internal boundaries
        self.bounds_ = np.quantile(np.asarray(X_ref, float), qs, axis=0)  # (K-1, n)
        # Bounds must be strictly increasing (constant-value channels collapse into few
        # granules; reported as-is)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, float)
        K = self.K
        G = np.empty(X.shape, dtype=np.int64)
        for j in range(X.shape[1]):
            G[:, j] = np.searchsorted(self.bounds_[:, j], X[:, j], side="right")
        return np.clip(G, 0, K - 1)

    def fit_transform(self, X_ref):
        return self.fit(X_ref).transform(X_ref)
