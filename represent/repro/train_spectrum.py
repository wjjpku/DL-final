"""Attempt 3: Hessian spectroscopy across an LR drop (pre-registered in
results/formula_lab/repin_prereg.json).

Resumes the floor-ladder trunk (G3-validated), drops to eta2 at step 3000,
and at pre-registered dS offsets measures the top-16 spectrum of (a) the
preconditioned Hessian P^-1/2 H P^-1/2 with P = sqrt(v_hat)+eps from the
optimizer's own state (PRIMARY; AdamW edge = 38/eta), and (b) the raw CE
Hessian (secondary).  fp32 double-backward, math SDPA, fixed probe batches.

Output: results/curves_spectrum_<scale>/<arm>.csv        (loss curve)
        results/curves_spectrum_<scale>/<arm>_spec.csv   (spectra)
Usage:  python train_spectrum.py --scale m --data_dir /root/dlf/data
          [--seed 1337] [--only spec_e10]
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train as T
from train_floor2 import SCALES, TRUNK, PEAK, make_trunk, restore

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROBES = [0, 10, 25, 50, 100, 200, 400, 800, 1600, 2400, 3200, 3999]
PROBES_B12_EXTRA = [5600, 7999]
ARMS = {
    "spec_e10": dict(eta2=1e-4, bs2=48, t2=4000),
    "spec_e40": dict(eta2=4e-4, bs2=48, t2=4000),
    "spec_nodrop": dict(eta2=1.5e-3, bs2=48, t2=4000),
    "spec_e10_b12": dict(eta2=1e-4, bs2=12, t2=8000),
    "spec_e10_b192": dict(eta2=1e-4, bs2=192, t2=4000),
}
K_TOP, M_PRE, M_RAW = 16, 64, 32
N_PROBE_BATCHES, PROBE_BS, PROBE_SEED = 4, 48, 424242
BETA2, EPS = 0.95, 1e-8


def cdir(scale):
    d = os.path.join(ROOT, "results", f"curves_spectrum_{scale}")
    os.makedirs(d, exist_ok=True)
    return d


def probe_batches(trd, block):
    gen = torch.Generator().manual_seed(PROBE_SEED)
    return [T.get_batch(trd, block, PROBE_BS, gen)
            for _ in range(N_PROBE_BATCHES)]


def flat_cpu(vs):
    return torch.cat([v.detach().reshape(-1).float().cpu() for v in vs])


def unflatten_to(flat, like, dev):
    out, off = [], 0
    for p in like:
        n = p.numel()
        out.append(flat[off:off + n].view(p.shape).to(dev))
        off += n
    return out


def hvp_avg(model, params, batches, vec):
    Hv = [torch.zeros_like(p) for p in params]
    for x, y in batches:
        with sdpa_kernel([SDPBackend.MATH]):
            _, loss = model(x, y)
        g = torch.autograd.grad(loss, params, create_graph=True)
        dot = sum((gi * vi).sum() for gi, vi in zip(g, vec))
        h = torch.autograd.grad(dot, params)
        for a, b in zip(Hv, h):
            a.add_(b.detach(), alpha=1.0 / len(batches))
        del g, dot, h
    return Hv


def precond_half_inv(opt, params):
    """s^{-1/2} with s = sqrt(v_hat)+eps, from the live optimizer state."""
    out = []
    for p in params:
        st = opt.state[p]
        t = int(st["step"]) if not torch.is_tensor(st["step"]) \
            else int(st["step"].item())
        vhat = st["exp_avg_sq"] / (1.0 - BETA2 ** t)
        out.append((vhat.sqrt() + EPS).rsqrt())
    return out


def lanczos_topk(opfn, params, m_iters, seed):
    """Full-reorthogonalized Lanczos; basis stored on CPU.  Returns
    (top-k eigenvalues desc, max relative Ritz residual among top-k)."""
    dim = sum(p.numel() for p in params)
    gen = torch.Generator().manual_seed(seed)
    v0 = torch.randn(dim, generator=gen)
    v0 /= v0.norm()
    V = torch.zeros(m_iters + 1, dim)
    V[0] = v0
    alphas, betas = [], []
    for j in range(m_iters):
        vj = unflatten_to(V[j], params, T.DEV)
        Hv = opfn(vj)
        w = flat_cpu(Hv)
        alpha = float(torch.dot(w, V[j]))
        alphas.append(alpha)
        w -= V[: j + 1].T @ (V[: j + 1] @ w)   # full reorth (covers alpha+beta)
        beta = float(w.norm())
        if beta < 1e-8 or j == m_iters - 1:
            betas.append(beta)
            break
        betas.append(beta)
        V[j + 1] = w / beta
    a = np.array(alphas)
    b = np.array(betas[: len(alphas) - 1])
    Tm = np.diag(a) + np.diag(b, 1) + np.diag(b, -1)
    evals, evecs = np.linalg.eigh(Tm)
    order = np.argsort(evals)[::-1][:K_TOP]
    top = evals[order]
    # Ritz residual |beta_m * last eigvec component| / |lambda|
    bl = betas[len(alphas) - 1] if len(betas) >= len(alphas) else 0.0
    res = np.abs(bl * evecs[-1, order]) / np.maximum(np.abs(top), 1e-12)
    return top, float(res[: min(4, len(res))].max())


def probe_spectrum(model, opt, params, batches):
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        with sdpa_kernel([SDPBackend.MATH]):
            ploss = float(np.mean([model(x, y)[1].item() for x, y in batches]))
    pinv = precond_half_inv(opt, params)
    op_pre = lambda v: [h * s for h, s in zip(
        hvp_avg(model, params, batches, [vi * s for vi, s in zip(v, pinv)]),
        pinv)]
    op_raw = lambda v: hvp_avg(model, params, batches, v)
    lp, resp = lanczos_topk(op_pre, params, M_PRE, 7)
    lr_, resr = lanczos_topk(op_raw, params, M_RAW, 7)
    model.zero_grad(set_to_none=True)
    return ploss, lp, resp, lr_, resr


def run_arm(scale, seed, blob, tag, trd, vad):
    sfx = "" if seed == 1337 else f"_s{seed}"
    out = os.path.join(cdir(scale), f"{tag}{sfx}.csv")
    outs = os.path.join(cdir(scale), f"{tag}{sfx}_spec.csv")
    if os.path.exists(outs):
        print(f"[skip] {scale}/{tag}{sfx}", flush=True)
        return
    cfg = ARMS[tag]
    eta2, bs2, n2 = cfg["eta2"], cfg["bs2"], cfg["t2"]
    probes = set(PROBES + (PROBES_B12_EXTRA if bs2 == 12 else []))
    model, opt, gen = restore(scale, seed, blob)
    eval_gen = torch.Generator().manual_seed(12345)
    block = SCALES[scale]["block"]
    batches = probe_batches(trd, block)
    params = [p for p in model.parameters() if p.requires_grad]
    rows, srows = [], []
    ema, nclip, t0 = None, 0, time.time()

    def do_probe(ds):
        ploss, lp, resp, lr_, resr = probe_spectrum(model, opt, params, batches)
        srows.append([TRUNK + ds, ds, ploss, nclip] + list(lp) + [resp]
                     + list(lr_) + [resr])
        print(f"[probe] {tag} dS={ds} S_pre={lp[0]:.1f} lam={lr_[0]:.1f} "
              f"res={resp:.3f} ({time.time()-t0:.0f}s)", flush=True)

    do_probe(0)
    for i in range(n2):
        step = TRUNK + i
        for g in opt.param_groups:
            g["lr"] = eta2
        x, y = T.get_batch(trd, block, bs2, gen)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        nclip += int(float(gn) > 1.0)
        opt.step()
        dense = step <= TRUNK + 1600 and step % 4 == 0
        tail = i >= n2 * 0.75 and step % max(n2 // 80, 10) == 0
        if dense or tail or step % 200 == 0 or i == n2 - 1:
            lv = loss.item()
            ema = lv if ema is None else 0.9 * ema + 0.1 * lv
            ev = T.eval_loss(model, vad, block, 48, 40 if tail else 10,
                             eval_gen)
            rows.append((step, eta2, ema, ev))
        if (i + 1) in probes:
            do_probe(i + 1)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "lr", "train_loss", "eval_loss"])
        w.writerows(rows)
    with open(outs, "w", newline="") as f:
        w = csv.writer(f)
        hdr = (["step", "dS", "probe_loss", "nclip"]
               + [f"lp{i+1}" for i in range(K_TOP)] + ["res_p"]
               + [f"lr{i+1}" for i in range(K_TOP)] + ["res_r"])
        w.writerow(hdr)
        w.writerows(srows)
    print(f"[done] {scale}/{tag}{sfx} ({time.time()-t0:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", required=True, choices=list(SCALES))
    ap.add_argument("--data_dir", default="/root/dlf/data")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    trd = np.memmap(os.path.join(a.data_dir, "wiki_train.u8"), dtype=np.uint8,
                    mode="r")
    vad = np.memmap(os.path.join(a.data_dir, "wiki_val.u8"), dtype=np.uint8,
                    mode="r")
    blob = make_trunk(a.scale, a.seed, trd)
    tags = [a.only] if a.only else list(ARMS)
    for tag in tags:
        run_arm(a.scale, a.seed, blob, tag, trd, vad)
    print(f"SPECTRUM {a.scale} s{a.seed} DONE", flush=True)


if __name__ == "__main__":
    main()
