# -*- coding: utf-8 -*-
"""关系型基线组（B1/B2/B5a/b/c）：统一"参考段自扫描 FA 匹配"协议。

所有基线输出边界统计序列 S(b)（窗口对 [b-w,b) vs [b,b+w) 的结构差异）。
参考零分布 = 同一构造在参考段上的原生分辨率自扫描（同窗长 w、同步长 blk、同相邻窗对），与测试流及主管线 pick_theta 的校准语义完全一致。
与本文方法走同一校准/告警管线：参考段滑窗对定 θ（FA 目标），测试扫描合并事件。
这构成论文的 FA 匹配比较协议。
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rdsg.granulation import Granulator
from rdsg.tables import block_tables, window_sum

_EPS = 1e-9


def _series_from_stat(X, w, blk, stat_fn, t_start, t_end):
    """通用滑窗统计：stat_fn(X_ref_win, X_cur_win) -> float。"""
    span0 = max(0, t_start - w)
    nb = (t_end - span0) // blk
    half = w // blk
    out_t, out_s = [], []
    k = (t_start - span0 + blk - 1) // blk
    while k + half <= nb:
        a0, a1 = span0 + (k - half) * blk, span0 + k * blk
        b1 = min(span0 + (k + half) * blk, t_end)
        if a1 <= a0 or b1 <= a1:
            k += 1
            continue
        S = stat_fn(X[a0:a1], X[a1:b1])
        out_t.append(span0 + k * blk)
        out_s.append(S)
        k += 1
    return np.asarray(out_t), np.asarray(out_s)



def _pair_split_series(X, stat_fn, t_start, t_end, step_pairs=400):
    """Legacy scan retained for the frozen gas-pipeline (E1) reproduction:
    long-window (w = span/6) non-overlapping pair series. The E2/WADI
    baselines use the native-resolution self-scan (_series_from_stat) instead;
    E1 sweeps quantile operating curves, so no budget calibration is involved.
    """
    w = (t_end - t_start) // 6
    ts, ss = [], []
    step = max(w // 4, 1)
    starts = np.arange(t_start, t_end - 2 * w + 1, step)
    if len(starts) > step_pairs:
        starts = starts[np.linspace(0, len(starts) - 1, step_pairs).astype(int)]
    for s in starts:
        ss.append(stat_fn(X[s:s + w], X[s + w:s + 2 * w]))
        ts.append(s + w)
    return np.asarray(ts), np.asarray(ss)


def pick_theta_split(ss, fa_target, spp, gap, w, ts=None):
    """Rate-based reference self-scan calibration (same semantics as
    run_e0_syn2.pick_theta): the smallest theta whose alarm-sample rate on the
    reference stream meets the budget (fa_target alarm samples per day, counted
    at the stream's native scoring resolution). Event merging downstream can
    only lower the event count below this rate. Replaces the earlier merged
    event-count rule, whose dense-alarm blob degeneracy made the budget inert.

    ts = score timestamps (sample units). When given, the span is the covered
    reference duration (ts[-1]-ts[0])*spp, matching run_e0_syn2.pick_theta;
    the pair-split series is subsampled, so len(ss)*spp would understate the
    span by the subsampling factor and starve the budget. ts=None keeps the
    legacy per-sample semantics for contiguous per-sample score arrays
    (e.g. GDN), where len(ss) already equals the sample count."""
    ss = np.asarray(ss, float)
    n = len(ss)
    if ts is not None and n >= 2:
        span_days = (float(ts[-1]) - float(ts[0])) * spp / 86400.0
    else:
        span_days = n * spp / 86400.0
    k = int(np.floor(fa_target * span_days))
    srt = np.sort(ss)
    if k >= n:
        return float(srt[0])
    return float(srt[n - k - 1])


def merge_events(times, on, gap):
    idx = np.where(on)[0]
    if len(idx) == 0:
        return []
    evs, t0, prev = [], times[idx[0]], times[idx[0]]
    for i in idx[1:]:
        if times[i] - prev > gap:
            evs.append((t0, prev + 1))
            t0 = times[i]
        prev = times[i]
    evs.append((t0, prev + 1))
    return evs


class WindowDivergenceBaseline:
    """B1/B5b/B5c 统一壳：窗口矩阵差异统计 + FA 匹配校准。"""

    def __init__(self, kind, w=128, blk=32, fa_target=0.5, spp=300.0, gap=256):
        self.kind, self.w, self.blk = kind, w, blk
        self.fa_target, self.spp, self.gap = fa_target, spp, gap

    def _matrix(self, Xw):
        with np.errstate(invalid="ignore", divide="ignore"):
            if self.kind == "corr":
                C = np.corrcoef(Xw, rowvar=False)
            elif self.kind == "pcorr":
                S = np.cov(Xw, rowvar=False)
                # Ledoit-Wolf 式收缩防奇异
                p = S.shape[0]
                mu = np.trace(S) / p
                S = (1 - 0.1) * S + 0.1 * mu * np.eye(p)
                # pinv: 短窗下即使收缩后仍可能数值奇异（近常量通道）
                C = np.linalg.pinv(S)
                d = np.sqrt(np.abs(np.diag(C)))
                C = C / np.maximum(np.outer(d, d), _EPS)
            else:
                raise ValueError(self.kind)
        return np.nan_to_num(C)

    def _stat(self, Xr, Xc):
        A, B = self._matrix(Xr), self._matrix(Xc)
        if self.kind == "corr":
            return float(np.linalg.norm(B - A, "fro"))
        p = A.shape[0]
        off = ~np.eye(p, dtype=bool)
        return float(np.abs(B - A)[off].max())


    def run_scores(self, X_ref, X_test, cal_pairs=200):
        """Reference and test score series (no thresholding) for operating-curve sweeps."""
        t_r, s_r = _series_from_stat(X_ref, self.w, self.blk, self._stat, self.w, len(X_ref))
        X = np.vstack([X_ref, X_test])
        t_q, s_q = _series_from_stat(X, self.w, self.blk, self._stat, len(X_ref), len(X))
        return t_r, s_r, t_q, s_q, self.gap

    def run(self, X_ref, X_test, cal_pairs=200):
        t_r, s_r = _series_from_stat(X_ref, self.w, self.blk, self._stat, self.w, len(X_ref))
        theta = pick_theta_split(s_r, self.fa_target, self.spp, self.gap, self.w, ts=t_r)
        X = np.vstack([X_ref, X_test])
        t_q, s_q = _series_from_stat(X, self.w, self.blk, self._stat, len(X_ref), len(X))
        evs = merge_events(t_q, s_q > theta, self.gap)
        return [(a - len(X_ref), b - len(X_ref)) for a, b in evs], dict(theta=theta)


class VARResidualBaseline:
    """B5c：参考段拟合 VAR(p)，测试滑窗残差协方差与参考残差协方差的白化距离。"""

    def __init__(self, w=128, p=4, blk=32, fa_target=0.5, spp=300.0, gap=256):
        self.w, self.p, self.blk = w, p, blk
        self.fa_target, self.spp, self.gap = fa_target, spp, gap

    def _fit(self, X):
        p = self.p
        Y, Z = X[p:], np.hstack([X[p - k: len(X) - k] for k in range(1, p + 1)])
        A = np.linalg.lstsq(Z, Y, rcond=1e-6)[0]  # LOCF 常数列下正规方程会奇异
        R = Y - Z @ A
        return A, R

    def _stat(self, Xr, Xc):
        A, Rr = self._fit(Xr)
        p = self.p
        Y, Z = Xc[p:], np.hstack([Xc[p - k: len(Xc) - k] for k in range(1, p + 1)])
        Rc = Y - Z @ A
        Sr = np.cov(Rr, rowvar=False)
        Sc = np.cov(Rc, rowvar=False)
        d = np.sqrt(np.maximum(np.diag(Sr), _EPS))
        Srn, Scn = Sr / np.outer(d, d), Sc / np.outer(d, d)
        return float(np.linalg.norm(Scn - Srn, "fro") / Srn.shape[0])


    def run_scores(self, X_ref, X_test, cal_pairs=120):
        """Reference and test score series (no thresholding) for operating-curve sweeps."""
        t_r, s_r = _series_from_stat(X_ref, self.w, self.blk, self._stat, self.w, len(X_ref))
        X = np.vstack([X_ref, X_test])
        t_q, s_q = _series_from_stat(X, self.w, self.blk, self._stat, len(X_ref), len(X))
        return t_r, s_r, t_q, s_q, self.gap

    def run(self, X_ref, X_test, cal_pairs=120):
        t_r, s_r = _series_from_stat(X_ref, self.w, self.blk, self._stat, self.w, len(X_ref))
        theta = pick_theta_split(s_r, self.fa_target, self.spp, self.gap, self.w, ts=t_r)
        X = np.vstack([X_ref, X_test])
        t_q, s_q = _series_from_stat(X, self.w, self.blk, self._stat, len(X_ref), len(X))
        evs = merge_events(t_q, s_q > theta, self.gap)
        return [(a - len(X_ref), b - len(X_ref)) for a, b in evs], dict(theta=theta)


class GrangerBaseline:
    """B2：候选对上岭回归双变量 Granger F 统计（lag=4），统计 = Σ_pairs |ΔGC|。"""

    def __init__(self, pairs, w=128, p=4, blk=32, fa_target=0.5, spp=300.0, gap=256):
        self.pairs, self.w, self.p, self.blk = pairs, w, p, blk
        self.fa_target, self.spp, self.gap = fa_target, spp, gap

    @staticmethod
    def _gc_pair(X, i, j, p, ridge=1e-3):
        """GC of i -> j：log(var(restricted)/var(full))，j 的自回归 vs i,j 联合。"""
        y = X[p:, j]
        Zr = np.hstack([X[p - k: len(X) - k, j:j + 1] for k in range(1, p + 1)])
        Zf = np.hstack([Zr] + [X[p - k: len(X) - k, i:i + 1] for k in range(1, p + 1)])
        def ols(Z):
            A = np.linalg.lstsq(Z, y, rcond=1e-6)[0]
            e = y - Z @ A
            return float(e @ e)
        sr, sf = ols(Zr), max(ols(Zf), _EPS)
        return max(np.log(sr / sf), 0.0)

    def _stat(self, Xr, Xc):
        tot = 0.0
        for (i, j) in self.pairs:
            tot += abs(self._gc_pair(Xc, i, j, self.p) - self._gc_pair(Xr, i, j, self.p))
        return tot


    def run_scores(self, X_ref, X_test, cal_pairs=60):
        """Reference and test score series (no thresholding) for operating-curve sweeps."""
        t_r, s_r = _series_from_stat(X_ref, self.w, self.blk, self._stat, self.w, len(X_ref))
        X = np.vstack([X_ref, X_test])
        t_q, s_q = _series_from_stat(X, self.w, self.blk, self._stat, len(X_ref), len(X))
        return t_r, s_r, t_q, s_q, self.gap

    def run(self, X_ref, X_test, cal_pairs=60):
        t_r, s_r = _series_from_stat(X_ref, self.w, self.blk, self._stat, self.w, len(X_ref))
        theta = pick_theta_split(s_r, self.fa_target, self.spp, self.gap, self.w, ts=t_r)
        X = np.vstack([X_ref, X_test])
        t_q, s_q = _series_from_stat(X, self.w, self.blk, self._stat, len(X_ref), len(X))
        evs = merge_events(t_q, s_q > theta, self.gap)
        return [(a - len(X_ref), b - len(X_ref)) for a, b in evs], dict(theta=theta)


class MIGraphBaseline:
    """B5a：粒域互信息图变化（统一 FA 匹配协议版；替代 v1 的零告警实现）。"""

    def __init__(self, K=4, w=128, blk=32, fa_target=0.5, spp=300.0, gap=256):
        self.K, self.w, self.blk = K, w, blk
        self.fa_target, self.spp, self.gap = fa_target, spp, gap

    def _mi_sum(self, Xr, Xc):
        from rdsg.stats import ipf_uniform
        g = Granulator(self.K).fit(Xr)
        Gr, Gc = g.transform(Xr), g.transform(Xc)
        tot = 0.0
        K = self.K
        for i in range(Gr.shape[1]):
            for j in range(Gr.shape[1]):
                if i == j:
                    continue
                Tr = np.bincount(Gr[:, i] * K + Gr[:, j], minlength=K * K).reshape(K, K).astype(float)
                Tc = np.bincount(Gc[:, i] * K + Gc[:, j], minlength=K * K).reshape(K, K).astype(float)
                P1 = ipf_uniform(Tr[None])[0]
                P2 = ipf_uniform(Tc[None])[0]
                def _mi(P):
                    indep = np.maximum(P.sum(-1, keepdims=True) * P.sum(-2, keepdims=True), _EPS)
                    terms = np.where(P > 0, P * np.log(P / indep), 0.0)
                    return float(terms.sum())
                tot += abs(_mi(P2) - _mi(P1))
        return float(tot)


    def run_scores(self, X_ref, X_test, cal_pairs=80):
        """Reference and test score series (no thresholding) for operating-curve sweeps."""
        if X_ref.shape[1] > 24:
            v = X_ref.std(axis=0)
            keep = np.argsort(-v)[:24]
            X_ref = X_ref[:, keep]
            X_test = X_test[:, keep]
        t_r, s_r = _series_from_stat(X_ref, self.w, self.blk, self._mi_sum, self.w, len(X_ref))
        X = np.vstack([X_ref, X_test])
        t_q, s_q = _series_from_stat(X, self.w, self.blk, self._mi_sum, len(X_ref), len(X))
        return t_r, s_r, t_q, s_q, self.gap

    def run(self, X_ref, X_test, cal_pairs=80, max_pairs=None):
        # 大通道数时按方差排序取 top 24 通道控制 O(n²) 成本
        if X_ref.shape[1] > 24:
            v = X_ref.std(axis=0)
            keep = np.argsort(-v)[:24]
            X_ref = X_ref[:, keep]
            X_test = X_test[:, keep]
        t_r, s_r = _series_from_stat(X_ref, self.w, self.blk, self._mi_sum, self.w, len(X_ref))
        theta = pick_theta_split(s_r, self.fa_target, self.spp, self.gap, self.w, ts=t_r)
        X = np.vstack([X_ref, X_test])
        t_q, s_q = _series_from_stat(X, self.w, self.blk, self._mi_sum, len(X_ref), len(X))
        evs = merge_events(t_q, s_q > theta, self.gap)
        return [(a - len(X_ref), b - len(X_ref)) for a, b in evs], dict(theta=theta)
