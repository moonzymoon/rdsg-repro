# -*- coding: utf-8 -*-
"""S.14 官方目标审计：Anomaly-Transformer 官方训练目标（极小极大关联差异，k=3，
官方默认容量 512/512/8/3 + ReLU + dropout 0.1）在本协议下的复评。

预注册口径（运行前固定）：
- 架构/损失/超参：官方 solver.py 的 loss1 = rec − k·series、loss2 = rec + k·prior
  （双反传极小极大，k=3，lr=1e-4），官方默认容量；窗口仍取协议窗 128（官方 win=100
  绑定其 1 分钟采样数据，本协议 5 分钟/1 小时网络沿用 128 以保持可比）。
- 早停：验证 loss1（rec − k·series）patience=3，至多 30 epoch。
- 评分：官方关联差异 AssDis = KL(series‖prior̂)+KL(prior̂‖series)（逐层平均），
  阈值走本协议参考段自扫描 0.25/day 匹配，事件合并 gap=4·WIN。
- 种子/网络：与 S.14 长臂相同（种子 1-5、三网络、同一注入套件）。
- 解读规则：如实报数——若官方目标臂显著逼近或反超家族，触发正文对比句诚实改写。

输出：results/paper/abl_at_official.json
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
MAX_EPOCHS, PATIENCE, K = 30, 3, 3
VAL_FRAC = 0.15


def _results_root():
    d = os.path.dirname(os.path.abspath(__file__))
    for base in (d, os.path.dirname(d), os.path.dirname(os.path.dirname(d))):
        if os.path.isdir(os.path.join(base, "results")):
            return os.path.join(base, "results")
    return os.path.normpath(os.path.join(d, "..", "..", "results"))


OUT = os.path.join(_results_root(), "paper", "abl_at_official.json")


def _windows(X, win=WIN):
    idx = np.arange(0, len(X) - win + 1, win // 4)
    ws = np.stack([X[i:i + win] for i in idx]).astype(np.float32)
    return idx, ws


def _kl(p, q):
    return (p * (torch.log(p + 1e-8) - torch.log(q + 1e-8))).sum(-1)


def _assdis(model, ws):
    """官方关联差异: 每层 KL(series‖prior̂)+KL(prior̂‖series), 窗口级均值。"""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(ws), 32):
            batch = torch.tensor(ws[i:i + 32]).to(DEVICE)
            _, series, prior, _ = model(batch)
            d = 0
            for s_l, p_l in zip(series, prior):
                ph = p_l / torch.unsqueeze(torch.sum(p_l, dim=-1), dim=-1).repeat(
                    1, 1, 1, WIN)
                d += _kl(s_l, ph).mean(axis=(1, 2)) + _kl(ph, s_l).mean(axis=(1, 2))
            out.append((d / len(prior)).cpu().numpy())
    return np.concatenate(out)


def _series_loss_train(series, prior):
    s = 0.0
    for s_l, p_l in zip(series, prior):
        ph = p_l / torch.unsqueeze(torch.sum(p_l, dim=-1), dim=-1).repeat(1, 1, 1, WIN)
        s = s + torch.mean(_kl(s_l, ph.detach())) + torch.mean(_kl(ph.detach(), s_l))
    return s / len(prior)


def _prior_loss_train(series, prior):
    p = 0.0
    for s_l, p_l in zip(series, prior):
        ph = p_l / torch.unsqueeze(torch.sum(p_l, dim=-1), dim=-1).repeat(1, 1, 1, WIN)
        p = p + torch.mean(_kl(ph, s_l.detach())) + torch.mean(_kl(s_l.detach(), ph))
    return p / len(prior)


def train_official_at(X_ref):
    n_ch = X_ref.shape[1]
    model = AnomalyTransformer(win_size=WIN, enc_in=n_ch, c_out=n_ch,
                               d_model=512, n_heads=8, e_layers=3, d_ff=512,
                               dropout=0.1, activation="relu").to(DEVICE)
    _, ws = _windows(X_ref)
    n_val = max(1, int(len(ws) * VAL_FRAC))
    ws_tr, ws_va = ws[:-n_val], ws[-n_val:]
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    ws_t = torch.tensor(ws_tr).to(DEVICE)
    best_val, best_state, bad, stop_ep = float("inf"), None, 0, MAX_EPOCHS
    for ep in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(len(ws_t))
        for i in range(0, len(ws_t), 32):
            batch = ws_t[perm[i:i + 32]]
            opt.zero_grad()
            out, series, prior, _ = model(batch)
            rec = F.mse_loss(out, batch)
            loss1 = rec - K * _series_loss_train(series, prior)
            loss2 = rec + K * _prior_loss_train(series, prior)
            loss1.backward(retain_graph=True)
            loss2.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            tot, cnt = 0.0, 0
            for i in range(0, len(ws_va), 32):
                b = torch.tensor(ws_va[i:i + 32]).to(DEVICE)
                out, series, prior, _ = model(b)
                rec = F.mse_loss(out, b)
                s = _series_loss_train(series, prior)
                tot += float(rec - K * s) * len(b)
                cnt += len(b)
            v = tot / cnt
        if v < best_val - 1e-6:
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


def official_pipeline(X_ref, X_test_inj, spp):
    model, stop_ep = train_official_at(X_ref)
    half = len(X_ref) // 2
    _, ref_ws = _windows(X_ref[half:])
    ref_scores = _assdis(model, ref_ws)
    t_idx, t_ws = _windows(X_test_inj)
    t_scores = _assdis(model, t_ws)
    span_days = len(ref_scores) * (WIN // 4) * spp / 86400.0
    k_allow = int(np.floor(FA_TARGET * span_days))
    srt = np.sort(ref_scores)
    theta = float(srt[len(srt) - k_allow - 1]) if k_allow < len(srt) else float(srt[0])
    on = t_idx[t_scores > theta]
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
        if sd in res.get(key, {}).get("at_official", {}):
            continue
        specs = make_specs(planner, dE0, X_ref, X_test, seed, lengths)
        X_test_inj = inject(X_test, specs, seed=seed)
        true_events = [(sp["start"], sp["start"] + sp["length"], sp["type"]) for sp in specs]
        t0 = time.time()
        evs, stop_ep = official_pipeline(X_ref, X_test_inj, spp)
        m = event_metrics(true_events, [(a, b, None) for a, b in evs],
                          len(X_test), spp)
        res.setdefault(key, {}).setdefault("at_official", {})[sd] = dict(
            recall=m["recall"], fa_per_day=m["fa_per_day"], stop_epoch=stop_ep)
        print(f"[{key} s{seed}] AT-official R={m['recall']:.3f} fa={m['fa_per_day']:.2f} "
              f"stop@{stop_ep} ({time.time()-t0:.0f}s)", flush=True)
        json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1, default=float)


def main():
    from rdsg.datasets import load_metr_la, load_pems_bay, load_beijing_air
    res = {}
    if os.path.exists(OUT):
        res = json.load(open(OUT, encoding="utf-8"))
    res.setdefault("config", {})["audit"] = "official-objective AT"
    res["config"].update(max_epochs=MAX_EPOCHS, patience=PATIENCE, k=K,
                         d_model=512, window=WIN)
    for key, loader in (("metr_la", load_metr_la), ("pems_bay", load_pems_bay),
                        ("beijing_air", load_beijing_air)):
        run_dataset(key, loader, res)
    print("done ->", OUT)


if __name__ == "__main__":
    main()
