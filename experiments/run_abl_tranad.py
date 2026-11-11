# -*- coding: utf-8 -*-
"""TranAD baseline: adversarially-trained transformer detector.

Simplified implementation: adversarial autoencoder with focus scoring.
Follows the key ideas of Tuli et al. VLDB 2022 (TranAD):
  - Transformer encoder-decoder reconstruction
  - Adversarial training (add noise to focus on harder samples)
  - Focus score = adversarial + reconstruction loss
Same calibration protocol as all other baselines.
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.normpath(os.path.join(HERE, ".."))
PROJECT = os.path.normpath(os.path.join(CODE, ".."))
sys.path.insert(0, CODE)
sys.path.insert(0, HERE)

from rdsg.metrics import match_events, event_metrics
from run_paper_e2 import LEN_BY_SPP, build_channel, make_specs, PAIR_STRATA_E
from rdsg import inject
def _results_root():
    """Results directory: the nearest ancestor that ships a results/ folder.

    Local repo: code/experiments/x.py -> <project>/results.
    Reviewer package: <pkg>/experiments/x.py -> <pkg>/results (the shipped
    frozen JSONs), so reruns merge with them instead of writing outside."""
    d = os.path.dirname(os.path.abspath(__file__))
    for base in (d, os.path.dirname(d), os.path.dirname(os.path.dirname(d))):
        if os.path.isdir(os.path.join(base, "results")):
            return os.path.join(base, "results")
    return os.path.normpath(os.path.join(d, "..", "..", "results"))


OUT = os.path.join(_results_root(), "paper", "abl_tranad.json")
WIN = 128
FA_TARGET = 0.25
SEEDS = range(1, 11)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class TranADModel(nn.Module):
    """Simplified TranAD: transformer encoder-decoder with adversarial focus."""

    def __init__(self, n_channels, win=WIN, d_model=64, n_heads=4, e_layers=2):
        super().__init__()
        self.win = win
        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=0.0, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)
        self.input_proj = nn.Linear(n_channels, d_model)
        # Decoder (mirrors encoder)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=0.0, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=e_layers)
        self.output_proj = nn.Linear(d_model, n_channels)
        # Adversarial noise scale (learned)
        self.noise_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, adversarial=False):
        """x: (B, L, C). Returns reconstructed x and adversarial score."""
        if adversarial and self.training:
            noise = torch.randn_like(x) * self.noise_scale.abs()
            x_noisy = x + noise
        else:
            x_noisy = x

        h = self.input_proj(x_noisy)
        h = self.encoder(h)
        # Self-decoding (no memory in simplified version)
        out = self.decoder(h, h)
        out = self.output_proj(out)

        # Reconstruction loss per timestep
        rec_loss = (out - x).pow(2).mean(dim=-1)  # (B, L)
        return out, rec_loss


class TranADDetector:
    def __init__(self, n_channels, win=WIN):
        self.model = TranADModel(n_channels, win).to(DEVICE)
        self.win = win

    def _windows(self, X):
        n = len(X) - self.win + 1
        idx = np.arange(0, n, self.win // 4)
        ws = np.stack([X[i:i + self.win] for i in idx])
        return idx, ws.astype(np.float32)

    def _score(self, X):
        """Focus score: reconstruction + adversarial loss per window."""
        self.model.eval()
        idx, ws = self._windows(X)
        scores = []
        with torch.no_grad():
            for i in range(0, len(ws), 32):
                batch = torch.tensor(ws[i:i+32]).to(DEVICE)
                # Clean pass
                _, rec_clean = self.model(batch, adversarial=False)
                # Adversarial pass
                _, rec_adv = self.model(batch, adversarial=True)
                # Focus score = mean reconstruction + adversarial amplification
                focus = rec_clean.mean(dim=-1) + 0.5 * rec_adv.mean(dim=-1)
                scores.append(focus.cpu().numpy())
        return np.concatenate(scores)

    def train(self, X_ref, epochs=5, lr=1e-4):
        self.model.train()
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        _, ws = self._windows(X_ref)
        ws_t = torch.tensor(ws).to(DEVICE)
        for ep in range(epochs):
            perm = torch.randperm(len(ws_t))
            total = 0
            for i in range(0, len(ws_t), 32):
                batch = ws_t[perm[i:i+32]]
                opt.zero_grad()
                out_clean, loss_clean = self.model(batch, adversarial=False)
                out_adv, loss_adv = self.model(batch, adversarial=True)
                rec = F.mse_loss(out_clean, batch)
                adv = loss_adv.mean()  # adversarial should increase reconstruction error on noisy input
                loss = rec + 0.1 * adv
                loss.backward()
                opt.step()
                total += loss.item()
            if ep == 0 or (ep + 1) % 2 == 0:
                print(f"    epoch {ep+1}/{epochs} loss={total/max(1, len(ws_t)//32):.4f}", flush=True)
        self.model.eval()


def tranad_pipeline(X_ref, X_test_inj, spp, fa_target=FA_TARGET):
    n_ch = X_ref.shape[1]
    det = TranADDetector(n_ch)
    det.train(X_ref, epochs=5)
    half = len(X_ref) // 2
    ref_scores = det._score(X_ref[half:])
    test_scores = det._score(X_test_inj)
    # Get window indices for test
    n_test = len(X_test_inj) - WIN + 1
    test_idx = np.arange(0, n_test, WIN // 4)

    span_days = len(ref_scores) * (WIN // 4) * spp / 86400.0
    k_allow = int(np.floor(fa_target * span_days))
    srt = np.sort(ref_scores)
    theta = float(srt[len(srt) - k_allow - 1]) if k_allow < len(srt) else float(srt[0])

    alarm = test_scores > theta
    gap = 4 * WIN
    on = test_idx[alarm]
    if len(on) == 0:
        return []
    events = []
    t0, prev = on[0], on[0]
    for idx in on[1:]:
        if idx - prev > gap:
            events.append((int(t0), int(prev + WIN)))
            t0 = idx
        prev = idx
    events.append((int(t0), int(prev + WIN)))
    return events


def run_dataset(key, loader):
    X, spp, meta = loader()
    X_ref, X_test = X[:12000], X[12000:12000 + 18000]
    lengths = LEN_BY_SPP[spp]
    taus = (0, 1, -1)
    planner = build_channel(X_ref, taus, strata=(16, 24))[0]
    force_R, force_E = planner.strong_pairs_, planner.weak_pairs_
    dR0 = planner
    dE0 = build_channel(X_ref, taus, strata=PAIR_STRATA_E, n_subset=6,
                        force_pairs=(force_R, force_E))[0]

    res = {}
    for seed in SEEDS:
        specs = make_specs(dR0, dE0, X_ref, X_test, seed, lengths)
        X_test_inj = inject(X_test, specs, seed=seed)
        true_events = [(sp["start"], sp["start"] + sp["length"], sp["type"]) for sp in specs]
        t0 = time.time()
        evs = tranad_pipeline(X_ref, X_test_inj, spp)
        pr3 = [(int(a), int(b), None) for a, b in evs]
        m = event_metrics(true_events, pr3, len(X_test), spp)
        res[str(seed)] = m
        print(f"  [{key} s{seed}] R={m['recall']:.3f} fa/d={m['fa_per_day']:.2f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    return {key: res}


def main():
    from rdsg.datasets import load_metr_la, load_pems_bay, load_beijing_air
    loaders = {"metr_la": load_metr_la, "pems_bay": load_pems_bay,
               "beijing_air": load_beijing_air}
    all_res = {}
    for key, loader in loaders.items():
        print(f"\n===== {key} =====", flush=True)
        all_res.update(run_dataset(key, loader))
        json.dump(all_res, open(OUT, "w"), indent=1)
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
