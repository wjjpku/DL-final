"""
train.py -- Generate real loss curves to (a) reproduce the MPL+DropRelaxS pipeline
end-to-end on data we control, and (b) GENERALIZE the non-adiabatic-lag claim to a
real transformer the original authors never trained.

Llama-2-style decoder (RMSNorm + RoPE + SwiGLU, no biases, weight tying), byte-level
enwik8, AdamW (beta1=0.9, beta2=0.95 -- the regime the paper's theory is about).

Runs a suite of LR schedules at several model scales and saves each loss curve to
results/curves/<scale>_<sched>.csv  (columns: step,lr,train_loss,eval_loss).

Schedules mirror the MPL public protocol (cosine/constant/wsd/wsdld + two-stage
"wsdcon" probes at several stage-2 LRs for the tau ~ 1/eta measurement).
"""
import os, sys, math, time, json, urllib.request, zipfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

DEV = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATADIR = os.path.join(ROOT, "data")
CURVEDIR = os.path.join(ROOT, "results", "curves")
os.makedirs(DATADIR, exist_ok=True)
os.makedirs(CURVEDIR, exist_ok=True)


# ----------------------------- data -----------------------------
# Byte-level corpus decoded from the local GPT-2-tokenized wikitext (see prep_bytes.py).
# Byte-level (vocab 256) keeps the output head cheap so 3 scales x 9 schedules is fast on a
# laptop GPU; the non-adiabatic-lag effect is an optimizer phenomenon, independent of tokenization.
DEFAULT_DATA = DATADIR
VOCAB = 256


def get_data(data_dir=DEFAULT_DATA):
    tr = np.memmap(os.path.join(data_dir, "wiki_train.u8"), dtype=np.uint8, mode="r")
    va = np.memmap(os.path.join(data_dir, "wiki_val.u8"), dtype=np.uint8, mode="r")
    return tr, va


def get_batch(split_arr, block, bs, gen):
    ix = torch.randint(len(split_arr) - block - 1, (bs,), generator=gen).numpy()
    idx = ix[:, None] + np.arange(block + 1)[None, :]          # (bs, block+1)
    chunk = np.asarray(split_arr[idx], dtype=np.int64)         # one vectorized gather (fast)
    x = torch.from_numpy(chunk[:, :-1]).pin_memory()
    y = torch.from_numpy(chunk[:, 1:]).pin_memory()
    return x.to(DEV, non_blocking=True), y.to(DEV, non_blocking=True)


# ----------------------------- model -----------------------------
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__(); self.w = nn.Parameter(torch.ones(d)); self.eps = eps
    def forward(self, x):
        return self.w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


def rope_cache(T, hd, base=10000.0, device=DEV):
    inv = 1.0 / (base ** (torch.arange(0, hd, 2, device=device).float() / hd))
    t = torch.arange(T, device=device).float()
    f = torch.outer(t, inv)
    return torch.cos(f), torch.sin(f)


def apply_rope(x, cos, sin):
    # x: (B,H,T,hd)
    B, H, T, hd = x.shape
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos = cos[:T].view(1, 1, T, hd // 2); sin = sin[:T].view(1, 1, T, hd // 2)
    o1 = x1 * cos - x2 * sin
    o2 = x1 * sin + x2 * cos
    return torch.stack([o1, o2], dim=-1).flatten(-2)


class Block(nn.Module):
    def __init__(self, d, nh):
        super().__init__()
        self.nh = nh; self.hd = d // nh
        self.n1 = RMSNorm(d); self.n2 = RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        hidden = int(8 / 3 * d); hidden = (hidden + 63) // 64 * 64
        self.w1 = nn.Linear(d, hidden, bias=False)
        self.w3 = nn.Linear(d, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d, bias=False)
    def forward(self, x, cos, sin):
        B, T, C = x.shape
        h = self.n1(x)
        q, k, v = self.qkv(h).split(C, dim=2)
        q = q.view(B, T, self.nh, self.hd).transpose(1, 2)
        k = k.view(B, T, self.nh, self.hd).transpose(1, 2)
        v = v.view(B, T, self.nh, self.hd).transpose(1, 2)
        q = apply_rope(q, cos, sin); k = apply_rope(k, cos, sin)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        o = o.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.proj(o)
        h = self.n2(x)
        x = x + self.w2(F.silu(self.w1(h)) * self.w3(h))
        return x


class GPT(nn.Module):
    def __init__(self, vocab=256, d=384, nh=6, nl=6, block=256):
        super().__init__()
        self.block = block; self.d = d; self.nh = nh
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([Block(d, nh) for _ in range(nl)])
        self.nf = RMSNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.emb.weight
        cos, sin = rope_cache(block, d // nh)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.apply(self._init)
    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)
    def forward(self, idx, targets=None):
        x = self.emb(idx)
        for b in self.blocks:
            x = b(x, self.cos, self.sin)
        x = self.nf(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
    def nparams(self):
        return sum(p.numel() for p in self.parameters())


# ----------------------------- schedules -----------------------------
def sched_cosine(total, peak, end, warm):
    e = np.empty(total)
    for t in range(total):
        if t < warm: e[t] = peak * (t + 1) / warm
        else:
            pr = (t - warm) / max(total - warm, 1)
            e[t] = end + 0.5 * (peak - end) * (1 + math.cos(math.pi * pr))
    return e

def sched_const(total, peak, warm):
    e = np.full(total, peak); e[:warm] = peak * (np.arange(1, warm + 1) / warm); return e

def sched_wsd(total, decay_start, peak, end, warm):   # (1-sqrt) style decay
    e = sched_const(total, peak, warm)
    dec = np.arange(decay_start, total); fr = (dec - decay_start) / max(total - decay_start, 1)
    e[decay_start:] = (math.sqrt(peak) * (1 - fr) + math.sqrt(end) * fr) ** 2
    return e

def sched_wsdld(total, decay_start, peak, end, warm):  # linear decay
    e = sched_const(total, peak, warm)
    dec = np.arange(decay_start, total); fr = (dec - decay_start) / max(total - decay_start, 1)
    e[decay_start:] = peak * (1 - fr) + end * fr
    return e

def sched_wsdcon(total, drop_step, peak, stage2, warm):  # two-stage: peak then constant stage2
    e = sched_const(total, peak, warm); e[drop_step:] = stage2; return e


# ----------------------------- train one schedule -----------------------------
@torch.no_grad()
def eval_loss(model, val, block, bs, n_eval, gen):
    model.eval(); tot = 0.0
    for _ in range(n_eval):
        x, y = get_batch(val, block, bs, gen)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, l = model(x, y)
        tot += l.item()
    model.train(); return tot / n_eval


def train_one(name, etas, cfg, train_arr, val_arr, fine_after=None, fine_every=4):
    out = os.path.join(CURVEDIR, name + ".csv")
    if os.path.exists(out):
        print(f"  [skip] {name} exists"); return
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    gen = torch.Generator().manual_seed(cfg["seed"])
    eval_gen = torch.Generator().manual_seed(12345)   # fixed eval batches across runs
    model = GPT(vocab=VOCAB, d=cfg["d"], nh=cfg["nh"], nl=cfg["nl"], block=cfg["block"]).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1.0, betas=(0.9, 0.95),
                            weight_decay=cfg["wd"], eps=1e-8)
    total = len(etas)
    bs, block = cfg["bs"], cfg["block"]
    log_every = cfg["log_every"]
    rows = []
    ema = None
    t0 = time.time()
    for step in range(total):
        lr = float(etas[step])
        for g in opt.param_groups: g["lr"] = lr
        x, y = get_batch(train_arr, block, bs, gen)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        # NOTE: no per-step loss.item() -- that forces a GPU sync every step and ~halves throughput
        # on small models. We sync only at logging steps.
        fine = (fine_after is not None and step >= fine_after and step % fine_every == 0)
        if step % log_every == 0 or fine or step == total - 1:
            lv = loss.item()
            ema = lv if ema is None else 0.9 * ema + 0.1 * lv
            ev = eval_loss(model, val_arr, block, bs, cfg["n_eval"], eval_gen)
            rows.append((step, lr, ema, ev))
            if step % (log_every * 20) == 0:
                dt = time.time() - t0
                print(f"  {name} step {step}/{total} lr={lr:.2e} ema={ema:.4f} eval={ev:.4f} "
                      f"({(step+1)/max(dt,1e-9):.0f} it/s)", flush=True)
    import csv
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["step", "lr", "train_loss", "eval_loss"]); w.writerows(rows)
    print(f"  [done] {name} -> {out}  ({time.time()-t0:.0f}s, {model.nparams()/1e6:.1f}M params)", flush=True)


# ----------------------------- driver -----------------------------
SCALES = {
    "s":  dict(d=256, nh=4, nl=4, block=256, bs=48),   # ~ few M
    "m":  dict(d=384, nh=6, nl=6, block=256, bs=48),   # ~ 11M
    "l":  dict(d=512, nh=8, nl=8, block=256, bs=40),   # ~ 25M
}
PEAK = 1.5e-3
ENDF = 0.1          # end LR = ENDF * peak
WARM = 400
TOTAL = 6000
DECAY_START = 4000
WSDCON_TOTAL = 6000
WSDCON_DROP = 3000
STAGE2 = [0.5e-4, 1.0e-4, 2.0e-4, 4.0e-4, 8.0e-4]   # tau ~ 1/eta probes (16x range)


def build_schedules():
    end = PEAK * ENDF
    S = {}
    S["cosine"] = sched_cosine(TOTAL, PEAK, end, WARM)
    S["constant"] = sched_const(TOTAL, PEAK, WARM)
    S["wsd"] = sched_wsd(TOTAL, DECAY_START, PEAK, end, WARM)
    S["wsdld"] = sched_wsdld(TOTAL, DECAY_START, PEAK, end, WARM)
    for s2 in STAGE2:
        tag = f"wsdcon_{int(round(s2*1e5))}"   # e.g. wsdcon_5 for 5e-5
        S[tag] = sched_wsdcon(WSDCON_TOTAL, WSDCON_DROP, PEAK, s2, WARM)
    return S


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", default="m,s,l")
    ap.add_argument("--data_dir", default=DEFAULT_DATA)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--n_eval", type=int, default=20)
    ap.add_argument("--only", default="")   # comma list of schedule names to restrict
    args = ap.parse_args()
    print(f"device={DEV}", flush=True)
    if DEV == "cuda":
        print(torch.cuda.get_device_name(0), flush=True)
    train_arr, val_arr = get_data(args.data_dir)
    print(f"data: train={len(train_arr)/1e6:.1f}M tokens val={len(val_arr)/1e6:.1f}M tokens vocab={VOCAB}", flush=True)
    scheds = build_schedules()
    if args.only:
        keep = set(args.only.split(","))
        scheds = {k: v for k, v in scheds.items() if k in keep}
    json.dump({k: v.tolist() for k, v in scheds.items()},
              open(os.path.join(CURVEDIR, "schedules.json"), "w"))
    for sc in args.scales.split(","):
        cfg = dict(SCALES[sc]); cfg.update(seed=args.seed, wd=args.wd,
                                           log_every=args.log_every, n_eval=args.n_eval)
        print(f"\n===== SCALE {sc}: {cfg} =====", flush=True)
        for name, etas in scheds.items():
            # dense sampling in the post-drop relaxation window of wsdcon curves (for tau ~ 1/eta)
            fa = (WSDCON_DROP - 40) if name.startswith("wsdcon") else None
            train_one(f"{sc}_{name}", etas, cfg, train_arr, val_arr, fine_after=fa, fine_every=4)


if __name__ == "__main__":
    main()
