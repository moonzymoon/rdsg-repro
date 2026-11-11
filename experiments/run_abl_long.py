# -*- coding: utf-8 -*-
"""S.14 长训臂：AT / TranAD 在更大容量 + 收敛训练下的复评（A2 攻击线的投稿前自审）。

预注册口径（运行前固定，不按结果解释）：
- 架构：AT 用官方 Anomaly-Transformer 实现（同紧凑臂），容量 d_model=256、
  e_layers=3、n_heads=8（紧凑臂为 64/2/4）；TranAD 臂同样升到 256/3/8。
- 训练：至多 30 epoch，早停 patience=5，验证集 = 参考段窗口的末 15%
  （训练/验证切分在窗口级），权重取验证损失最优 epoch。
- 协议：与紧凑臂完全一致——参考段训练、注入测试段评分、同一速率校准、
  同一事件匹配（E2 协议、种子 1-5）。
- 声明纪律：这是 longer-training, larger-capacity arms，不称 official
  configuration（损失函数仍是论文披露过的简化型）。
- 解读规则：两种结果都如实进 S.14——长训臂若仍显著低于 RDSG，
  写"converged training does not close the gap"；若逼近或反超，
  如实报数并触发正文对比句的诚实改写。

输出：results/paper/abl_long.json（含每种子 recall/fa、早停 epoch、配置）。
"""
import os, sys, json, time, copy
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.normpath(os.path.join(HERE, ".."))
PROJECT = os.path.normpath(os.path.join(CODE, ".."))
sys.path.insert(0, CODE)
sys.path.insert(0, HERE)

AT_ROOT = os.environ.get("AT_ROOT",
                         os.path.normpath(os.path.join(PROJECT, "Anomaly-Transformer")))
sys.path.insert(0, AT_ROOT)
from model.AnomalyTransformer import AnomalyTransformer

from rdsg.metrics import event_metrics
from run_paper_e2 import LEN_BY_SPP, build_channel, make_specs, PAIR_STRATA_E
from rdsg import inject

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WIN = 128
FA_TARGET = 0.25
SEEDS = range(1, 6)
D_MODEL, N_HEADS, E_LAYERS = 256, 8, 3
MAX_EPOCHS, PATIENCE = 30, 5
VAL_FRAC = 0.15


def _results_root():
    d = os.path.dirname(os.path.abspath(__file__))
    for base in (d, os.path.dirname(d), os.path.dirname(os.path.dirname(d))):
        if os.path.isdir(os.path.join(base, "results")):
            return os.path.join(base, "results")
    return os.path.normpath(os.path.join(d, "..", "..", "results"))


OUT = os.path.join(_results_root(), "paper", "abl_long.json")


def _windows(X, win=WIN):
    n = len(X) - win + 1
    idx = np.arange(0, n, win // 4)
    ws = np.stack([X[i:i + win] for i in idx]).astype(np.float32)
    return idx, ws


def _kl_scores(model, ws):
    model.eval()
    scores = []
    with torch.no_grad():
        for i in range(0, len(ws), 32):
            batch = torch.tensor(ws[i:i + 32]).to(DEVICE)
            out, series, prior, sigma = model(batch)
            kl = 0
            for s_l, p_l in zip(series, prior):
                kl_l = (s_l * (torch.log(s_l + 1e-8) - torch.log(p_l + 1e-8))).sum(-1)
                kl += kl_l.mean(axis=(1, 2))
            scores.append(kl.cpu().numpy())
    return np.concatenate(scores)


def _rec_val_loss(model, ws_val):
    model.eval()
    tot = 0.0
    with torch.no_grad():
        for i in range(0, len(ws_val), 32):
            batch = torch.tensor(ws_val[i:i + 32]).to(DEVICE)
            out, series, prior, sigma = model(batch)
            tot += F.mse_loss(out, batch, reduction="sum").item()
    return tot / len(ws_val)


def train_long_at(X_ref):
    n_ch = X_ref.shape[1]
    model = AnomalyTransformer(win_size=WIN, enc_in=n_ch, c_out=n_ch,
                               d_model=D_MODEL, n_heads=N_HEADS, e_layers=E_LAYERS,
                               d_ff=D_MODEL * 2, dropout=0.0,
                               activation="gelu").to(DEVICE)
    _, ws = _windows(X_ref)
    n_val = max(1, int(len(ws) * VAL_FRAC))
    ws_tr, ws_va = ws[:-n_val], ws[-n_val:]
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    ws_t = torch.tensor(ws_tr).to(DEVICE)
    best_val, best_state, stop_ep = float("inf"), None, MAX_EPOCHS
    for ep in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(len(ws_t))
        for i in range(0, len(ws_t), 32):
            batch = ws_t[perm[i:i + 32]]
            opt.zero_grad()
            out, series, prior, sigma = model(batch)
            rec = F.mse_loss(out, batch)
            kl = 0
            for s_l, p_l in zip(series, prior):
                kl_l = (s_l * (torch.log(s_l + 1e-8) - torch.log(p_l + 1e-8))).sum(-1)
                kl += kl_l.mean()
            (rec + 0.01 * kl).backward()
            opt.step()
        v = _rec_val_loss(model, ws_va)
        if v < best_val - 1e-5:
            best_val, best_state, bad = v, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= PATIENCE:
                stop_ep = ep + 1
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, stop_ep


def at_long_pipeline(X_ref, X_test_inj, spp):
    model, stop_ep = train_long_at(X_ref)
    half = len(X_ref) // 2
    _, ref_ws = _windows(X_ref[half:])
    ref_scores = _kl_scores(model, ref_ws)
    t_idx, t_ws = _windows(X_test_inj)
    t_scores = _kl_scores(model, t_ws)
    span_days = len(ref_scores) * (WIN // 4) * spp / 86400.0
    k_allow = int(np.floor(FA_TARGET * span_days))
    srt = np.sort(ref_scores)
    theta = float(srt[len(srt) - k_allow - 1]) if k_allow < len(srt) else float(srt[0])
    alarm = t_scores > theta
    on = t_idx[alarm]
    events = []
    if len(on):
        t0, prev = on[0], on[0]
        for ix in on[1:]:
            if ix - prev > 4 * WIN:
                events.append((int(t0), int(prev + WIN)))
                t0 = ix
            prev = ix
        events.append((int(t0), int(prev + WIN)))
    return events, stop_ep


def train_long_tranad(X_ref):
    """TranAD 长臂：同容量升级（256/3/8）+ 30 epoch + 早停，复用 TranADModel。"""
    from run_abl_tranad import TranADModel
    n_ch = X_ref.shape[1]
    model = TranADModel(n_ch, win=WIN, d_model=D_MODEL, n_heads=N_HEADS,
                        e_layers=E_LAYERS).to(DEVICE)
    _, ws = _windows(X_ref)
    n_val = max(1, int(len(ws) * VAL_FRAC))
    ws_tr, ws_va = ws[:-n_val], ws[-n_val:]
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    ws_t = torch.tensor(ws_tr).to(DEVICE)

    def _val():
        model.eval()
        tot, cnt = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(ws_va), 32):
                b = torch.tensor(ws_va[i:i + 32]).to(DEVICE)
                out, rec_l = model(b, adversarial=False)
                tot += F.mse_loss(out, b, reduction="sum").item()
                cnt += len(b)
        return tot / cnt

    def _focus_np(w):
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(w), 32):
                b = torch.tensor(w[i:i + 32]).to(DEVICE)
                _, rec_c = model(b, adversarial=False)
                _, rec_a = model(b, adversarial=True)
                f = rec_c.mean(dim=-1) + 0.5 * rec_a.mean(dim=-1)
                out.append(f.cpu().numpy())
        return np.concatenate(out)

    best_val, best_state, stop_ep, bad = float("inf"), None, MAX_EPOCHS, 0
    for ep in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(len(ws_t))
        for i in range(0, len(ws_t), 32):
            batch = ws_t[perm[i:i + 32]]
            opt.zero_grad()
            out_c, _ = model(batch, adversarial=False)
            out_a, _ = model(batch, adversarial=True)
            loss = F.mse_loss(out_c, batch) + 0.1 * F.mse_loss(out_a, batch)
            loss.backward()
            opt.step()
        v = _val()
        if v < best_val - 1e-5:
            best_val, best_state, bad = v, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= PATIENCE:
                stop_ep = ep + 1
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, stop_ep, _focus_np


def tranad_long_pipeline(X_ref, X_test_inj, spp):
    model, stop_ep, score_fn = train_long_tranad(X_ref)
    half = len(X_ref) // 2
    _, ref_ws = _windows(X_ref[half:])
    ref_scores = score_fn(ref_ws)
    t_idx, t_ws = _windows(X_test_inj)
    t_scores = score_fn(t_ws)
    span_days = len(ref_scores) * (WIN // 4) * spp / 86400.0
    k_allow = int(np.floor(FA_TARGET * span_days))
    srt = np.sort(ref_scores)
    theta = float(srt[len(srt) - k_allow - 1]) if k_allow < len(srt) else float(srt[0])
    alarm = t_scores > theta
    on = t_idx[alarm]
    events = []
    if len(on):
        t0, prev = on[0], on[0]
        for ix in on[1:]:
            if ix - prev > 4 * WIN:
                events.append((int(t0), int(prev + WIN)))
                t0 = ix
            prev = ix
        events.append((int(t0), int(prev + WIN)))
    return events, stop_ep


def run_dataset(key, loader, res):
    from run_abl_tranad import TranADModel  # 复用 TranAD 臂模型结构
    X, spp, meta = loader()
    X_ref, X_test = X[:12000], X[12000:12000 + 18000]
    lengths = LEN_BY_SPP[spp]
    taus = (0, 1, -1)
    planner = build_channel(X_ref, taus, strata=(16, 24))[0]
    force_R, force_E = planner.strong_pairs_, planner.weak_pairs_
    dE0 = build_channel(X_ref, taus, strata=PAIR_STRATA_E, n_subset=6,
                        force_pairs=(force_R, force_E))[0]
    for seed in SEEDS:
        sd = str(seed)
        if sd in res.get(key, {}).get("at_long", {}) and sd in res.get(key, {}).get("tranad_long", {}):
            continue
        specs = make_specs(planner, dE0, X_ref, X_test, seed, lengths)
        X_test_inj = inject(X_test, specs, seed=seed)
        true_events = [(sp["start"], sp["start"] + sp["length"], sp["type"]) for sp in specs]
        if sd not in res.get(key, {}).get("at_long", {}):
            t0 = time.time()
            evs, stop_ep = at_long_pipeline(X_ref, X_test_inj, spp)
            m = event_metrics(true_events, [(a, b, None) for a, b in evs],
                              len(X_test), spp)
            res.setdefault(key, {}).setdefault("at_long", {})[sd] = dict(
                recall=m["recall"], fa_per_day=m["fa_per_day"], stop_epoch=stop_ep)
            print(f"[{key} s{seed}] AT-long R={m['recall']:.3f} fa={m['fa_per_day']:.2f} "
                  f"stop@{stop_ep} ({time.time()-t0:.0f}s)", flush=True)
            json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1, default=float)
        if sd not in res.get(key, {}).get("tranad_long", {}):
            t0 = time.time()
            evs, stop_ep = tranad_long_pipeline(X_ref, X_test_inj, spp)
            m = event_metrics(true_events, [(a, b, None) for a, b in evs],
                              len(X_test), spp)
            res.setdefault(key, {}).setdefault("tranad_long", {})[sd] = dict(
                recall=m["recall"], fa_per_day=m["fa_per_day"], stop_epoch=stop_ep)
            print(f"[{key} s{seed}] TranAD-long R={m['recall']:.3f} fa={m['fa_per_day']:.2f} "
                  f"stop@{stop_ep} ({time.time()-t0:.0f}s)", flush=True)
            json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1, default=float)


def main():
    from rdsg.datasets import load_metr_la, load_pems_bay, load_beijing_air
    loaders = {"metr_la": load_metr_la, "pems_bay": load_pems_bay,
               "beijing_air": load_beijing_air}
    res = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
    for key, loader in loaders.items():
        if (all(str(s) in res.get(key, {}).get("at_long", {}) for s in SEEDS)
                and all(str(s) in res.get(key, {}).get("tranad_long", {}) for s in SEEDS)):
            print(f"[{key}] AT-long complete, skip", flush=True)
            continue
        run_dataset(key, loader, res)
    # 汇总
    for key in res:
        arm = res[key].get("at_long", {})
        if arm:
            r = [v["recall"] for v in arm.values()]
            print(f"{key} AT-long mean recall = {np.mean(r):.3f} (n={len(r)})", flush=True)


if __name__ == "__main__":
    main()
