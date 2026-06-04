#!/usr/bin/env python3
"""Real-model check (v2): tau vs eta in a small AdamW-trained transformer.

Fixes over v1:
  (1) regime: train LONG at eta_peak so the loss reaches its NOISE FLOOR (bias gone),
      so that stepping the LR down makes the loss DROP (floor relaxation) rather than
      rise (under-training). We print the end-of-stable loss to check the plateau.
  (2) measurement: do NOT fit a global exponential on raw steps (bias drift + floor
      relaxation are superposed). Instead detrend with the post-step running minimum
      and read off tau via a SLIDING-WINDOW local relaxation estimate.
Theory: the floor-relaxation time tau ~ 1/eta.
"""
import sys
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

REPO = Path(__file__).resolve().parent

torch.manual_seed(0); np.random.seed(0)
dev = "mps" if torch.backends.mps.is_available() else "cpu"

V, N = 32, 200_000
rng = np.random.default_rng(1)
ntab = 2048
logits = rng.standard_normal((ntab, V)) * 2.5
def gen():
    seq = np.zeros(N, dtype=np.int64); h = 0
    P = np.exp(logits - logits.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
    for i in range(N):
        seq[i] = rng.choice(V, p=P[h]); h = (h * V + int(seq[i])) % ntab
    return seq
DATA = torch.tensor(gen(), device=dev)
CTX, BATCH = 48, 16

def batch(n=BATCH):
    ix = torch.randint(0, N - CTX - 1, (n,), device=dev)
    x = torch.stack([DATA[i:i+CTX] for i in ix]); y = torch.stack([DATA[i+1:i+CTX+1] for i in ix])
    return x, y
EVAL = [batch(128) for _ in range(4)]

class Block(nn.Module):
    def __init__(s, d, h):
        super().__init__(); s.a = nn.MultiheadAttention(d, h, batch_first=True)
        s.l1 = nn.LayerNorm(d); s.l2 = nn.LayerNorm(d)
        s.m = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
        s.register_buffer("mask", torch.triu(torch.ones(CTX, CTX)*float("-inf"), 1))
    def forward(s, x):
        z = s.l1(x); a,_ = s.a(z, z, z, attn_mask=s.mask, need_weights=False)
        x = x + a; return x + s.m(s.l2(x))
class GPT(nn.Module):
    def __init__(s, d=96, h=4, L=2):
        super().__init__(); s.te = nn.Embedding(V, d); s.pe = nn.Embedding(CTX, d)
        s.blocks = nn.ModuleList([Block(d, h) for _ in range(L)])
        s.ln = nn.LayerNorm(d); s.head = nn.Linear(d, V)
    def forward(s, x):
        z = s.te(x) + s.pe(torch.arange(x.shape[1], device=x.device))[None]
        for b in s.blocks: z = b(z)
        return s.head(s.ln(z))

@torch.no_grad()
def evloss(model):
    model.eval(); t = sum(F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1)).item() for x, y in EVAL)
    model.train(); return t / len(EVAL)

WARM, STABLE_END, TOTAL, EVERY = 400, 7000, 13000, 10
def run(eta_peak, eta_low, beta2=0.95):
    m = GPT().to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=eta_peak, betas=(0.9, beta2), weight_decay=0.0)
    rec = []
    for step in range(TOTAL):
        lr = eta_peak*step/WARM if step < WARM else (eta_peak if step < STABLE_END else eta_low)
        for g in opt.param_groups: g["lr"] = lr
        x, y = batch(); loss = F.cross_entropy(m(x).reshape(-1, V), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % EVERY == 0: rec.append((step, evloss(m)))
    return np.array(rec)

def sliding_tau(rec):
    """Detrended relaxation time of the post-step transient.
    peak = the PRE-step (eta_peak) equilibrium loss (NOT the already-relaxing post points);
    asymptote = mean of the late post region; tau = steps to fall (1-1/e) of the drop."""
    step, L = rec[:, 0], rec[:, 1]
    peak = L[step < STABLE_END][-3:].mean()                  # pre-step equilibrium
    post = step >= STABLE_END
    t = step[post] - STABLE_END
    y = np.convolve(L[post], np.ones(3)/3, mode="same")      # light smoothing
    Linf = y[-15:].mean()
    drop = peak - Linf
    if drop <= 1e-3:           # loss did not drop -> not in noise-floor regime
        return np.nan, peak, Linf
    target = peak - (1 - 1/np.e) * drop
    below = np.where(y <= target)[0]
    return (t[below[0]] if len(below) else np.nan), peak, Linf

def main():
    eta_peak = 3e-3
    print(f"device={dev}; v2 small GPT; eta_peak={eta_peak}; stable until {STABLE_END}", flush=True)
    print(f"  {'eta_low':>9s} {'L@stable_end':>12s} {'L_final':>9s} {'drop?':>6s} {'tau':>7s}", flush=True)
    rows = []; curves = {}
    for div in [4, 8, 16, 32]:
        elo = eta_peak/div
        c = run(eta_peak, elo)
        curves[f"div{div}"] = c
        Lstable = c[np.argmin(np.abs(c[:,0]-(STABLE_END-EVERY))),1]
        tau, peak, Linf = sliding_tau(c)
        dropped = "yes" if (peak-Linf) > 1e-3 else "NO"
        rows.append((elo, tau, dropped))
        print(f"  {elo:9.2e} {Lstable:12.4f} {c[-1,1]:9.4f} {dropped:>6s} {tau if np.isfinite(tau) else -1:7.0f}", flush=True)
    np.savez(str(REPO.parent/"results"/"small_model_curves.npz"), **curves)
    etas = np.array([r[0] for r in rows]); taus = np.array([r[1] for r in rows])
    ok = np.isfinite(taus) & np.array([r[2]=="yes" for r in rows])
    print("-"*50, flush=True)
    if ok.sum() >= 2:
        p = -np.polyfit(np.log(etas[ok]), np.log(taus[ok]), 1)[0]
        print(f"  log-log slope p = {p:.2f}  (theory tau∝1/eta => p=1; used {ok.sum()} pts)")
    else:
        print(f"  inconclusive: only {ok.sum()} usable points (loss must DROP after step-down).")

if __name__ == "__main__":
    main()
