# -*- coding: utf-8 -*-
"""论文实验数据加载器：METR-LA / PEMS-BAY / Beijing Air / Gas Pipeline。

统一返回 (X, spp, meta)：X (T, n) float，NaN 已按载明策略处理；spp 秒/点。
"""
import os
import numpy as np
import pandas as pd
import h5py

BASE = os.environ.get("RDSG_DATA_DIR",
       os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
           os.path.abspath(__file__)))), "datasets"))


def _h5(path):
    with h5py.File(path, "r") as f:
        key = "df" if "df" in f else list(f.keys())[0]  # PEMS-BAY 发布版根键为 speed
        g = f[key]
        idx = g["axis1"][:]
        vals = g["block0_values"][:]
    return pd.DataFrame(vals, index=pd.to_datetime(idx, unit="ns"))


def load_metr_la(n_channels=40, rows=34000):
    df = _h5(os.path.join(BASE, "traffic", "metr-la.h5"))
    X = df.values.astype(float)[:rows]
    ok = ~np.isnan(X).any(axis=0)
    ch = [i for i, k in enumerate(ok) if k][:n_channels]
    return X[:, ch], 300.0, {"name": "METR-LA", "channels": len(ch), "rows": rows}


def load_pems_bay(n_channels=40, rows=40000):
    df = _h5(os.path.join(BASE, "traffic", "pems-bay.h5"))
    X = df.values.astype(float)[:rows]
    ok = ~np.isnan(X).any(axis=0)
    ch = [i for i, k in enumerate(ok) if k][:n_channels]
    return X[:, ch], 300.0, {"name": "PEMS-BAY", "channels": len(ch), "rows": rows}


POLLUTANTS = ["PM2.5", "PM10", "SO2", "NO2", "O3", "TEMP", "WSPM"]


def load_beijing_air(years=(2014, 2015), max_channels=32):
    """12 站点 × 7 变量；取指定年份、剔除窗口内含 NaN 的通道、按变量优先重排。

    仅取 7 个传感变量列（year/month/day/hour 等日历列不进入通道）；NaN 剔除
    逐“站点×变量”单元判定后按可用性取前 max_channels 个，再按变量优先重排，
    使同变量的跨站点通道相邻。"""
    files = sorted(os.listdir(os.path.join(BASE, "beijing_air", "PRSA_Data_20130301-20170228")))
    mats = []
    for fn in files:
        d = pd.read_csv(os.path.join(BASE, "beijing_air", "PRSA_Data_20130301-20170228", fn))
        d = d[(d.year >= years[0]) & (d.year <= years[1])]
        d = d[POLLUTANTS].apply(pd.to_numeric, errors="coerce")
        mats.append(d.values)
    T = min(m.shape[0] for m in mats)
    X = np.stack([m[:T] for m in mats], axis=1)  # (T, 12站点, 7变量)
    X = X.reshape(T, -1)                          # (T, 84)，站点主序
    # NaN 策略：逐通道线性插值 + 两端补齐（空气质量常规做法；统计口径载于论文 S.8）
    X = pd.DataFrame(X).interpolate(limit_direction="both").values
    keep = set(np.where(~np.isnan(X).any(axis=0))[0][:max_channels].tolist())
    # 变量优先重排：同变量的跨站点通道相邻
    order = [s * len(POLLUTANTS) + v
             for v in range(len(POLLUTANTS))
             for s in range(X.shape[1] // len(POLLUTANTS))
             if s * len(POLLUTANTS) + v in keep]
    return X[:, order], 3600.0, {"name": "Beijing-Air", "channels": len(order),
                                 "rows": T, "years": years}


GAS_COLS = ["setpoint", "gain", "reset rate", "deadband", "cycle time", "rate",
            "pressure measurement", "crc rate", "command response"]


def load_gas_pipeline():
    """MSU Gas Pipeline：进程通道 LOCF（轮询响应间保持），网络通道原样。

    返回 (X, spp, meta)：meta 含参考段（最长正常段）、测试段边界与 53 个标注攻击事件。
    """
    df = pd.read_csv(os.path.join(BASE, "gas_pipeline", "MainDataset.csv"), low_memory=False)
    for c in GAS_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    X = df[GAS_COLS].copy()
    X = X.ffill().bfill().values.astype(float)          # LOCF：轮询响应间物理量保持
    lab = (df["binary"].values == 1)
    cats = df["categorized"].values
    REF0, REF1 = 238860, 240480                          # 最长正常段（27 min）
    TST1 = len(df)
    # 事件：60 样本窗攻击密度≥0.3，合并 gap≤2 窗
    W = 60
    nb = (TST1 - REF1) // W
    dens = lab[REF1:REF1 + nb * W].reshape(nb, W).mean(axis=1)
    ab = dens >= 0.3
    events, i = [], 0
    while i < nb:
        if ab[i]:
            j, gap = i, 0
            while j < nb and gap <= 2:
                gap = 0 if ab[j] else gap + 1
                j += 1
            s, e = REF1 + i * W, REF1 + (j - gap) * W
            cc = sorted(set(cats[s:e]) - {0})
            events.append([s - REF1, e - REF1, "+".join(map(str, cc))])
            i = j
        else:
            i += 1
    X1 = X[REF0:TST1]
    X1 = (X1 - X1[: REF1 - REF0].mean(axis=0)) / np.maximum(X1[: REF1 - REF0].std(axis=0), 1e-9)
    meta = {"name": "Gas-Pipeline", "channels": X1.shape[1], "ref": (0, REF1 - REF0),
            "test": (REF1 - REF0, TST1 - REF0), "events": events, "cols": GAS_COLS}
    return X1, 1.0, meta



def load_hai20(channels=30, ds_factor=10):
    """HAI 20.07：train1 为攻击-free 参考段；test1+test2 为测试段（38 个标注攻击事件）。

    1s 采样按 ds_factor 中位数降采样。返回 (X, spp, meta)。
    """
    base = os.path.join(BASE, "hai", "hai-20.07")
    def _ds(path, cols=None):
        df = pd.read_csv(path, sep=";")
        if cols is None:
            num = [c for c in df.columns if c not in ("time", "attack", "attack_P1", "attack_P2", "attack_P3")]
            # 方差筛选在调用侧做；这里保留全部数值列
            keep = []
            for c in num:
                v = pd.to_numeric(df[c], errors="coerce")
                if v.notna().mean() > 0.999 and v.std() > 1e-9:
                    keep.append(c)
            cols = keep
        X = df[cols].astype(float)
        lab = df["attack"].astype(float)
        T = (len(X) // ds_factor) * ds_factor
        Xd = X[:T].values.reshape(-1, ds_factor, len(cols))
        Xd = np.median(Xd, axis=1)
        ld = lab[:T].values.reshape(-1, ds_factor).mean(axis=1)
        return Xd, ld, cols
    Xr, _, cols = _ds(os.path.join(base, "train1.csv"))
    Xt1, l1, _ = _ds(os.path.join(base, "test1.csv"), cols)
    Xt2, l2, _ = _ds(os.path.join(base, "test2.csv"), cols)
    # 方差筛通道（参考段）
    v = Xr.std(axis=0)
    keep = np.argsort(-v)[:channels]
    Xr = Xr[:, keep]
    Xt1 = Xt1[:, keep]
    Xt2 = Xt2[:, keep]
    mu, sd = Xr.mean(axis=0), np.maximum(Xr.std(axis=0), 1e-9)
    Xr = (Xr - mu) / sd
    Xt1 = (Xt1 - mu) / sd
    Xt2 = (Xt2 - mu) / sd
    X = np.vstack([Xr, Xt1, Xt2])
    n_ref = len(Xr)
    off1 = n_ref
    off2 = n_ref + len(Xt1)
    events = []
    for (a, b) in [(off1, off1 + len(Xt1)), (off2, off2 + len(Xt2))]:
        lab = np.zeros(b - a)
        if a == off1:
            lab[: len(l1)] = l1
        else:
            lab[: len(l2)] = l2
        W = 60
        nb = len(lab) // W
        dens = lab[:nb * W].reshape(nb, W).mean(axis=1)
        ab = dens >= 0.2
        i = 0
        while i < nb:
            if ab[i]:
                j, g = i, 0
                while j < nb and g <= 1:
                    g = 0 if ab[j] else g + 1
                    j += 1
                events.append([i * W, (j - g) * W, "attack"])
                i = j
            else:
                i += 1
    meta = {"name": "HAI-20.07", "channels": int(X.shape[1]), "ref": (0, n_ref),
            "test": (n_ref, len(X)), "events": events}
    return X, float(ds_factor), meta
