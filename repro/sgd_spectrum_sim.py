#!/usr/bin/env python3
"""Find which beyond-quadratic ingredient produces MPL's gamma (eta^-gamma).

Deterministic mean-loss SGD simulation on a quadratic-with-power-law-spectrum,
plus optional beyond-quadratic ingredients:
  * sharpen kappa : progressive sharpening, lambda(t)=lambda*(1+kappa*u(t)),
    u=S(t)/S_total. (Late, low-LR steps see HIGHER curvature -> annealing benefit
    realizes faster late = the sign of MPL's eta^-gamma.)
  * cexp          : noise loss-weight ~ lambda^cexp (cexp>0 emphasises the
    high-curvature modes where the eta^2 lambda^2 nonlinearity lives).
  * noise=loss    : injected variance ~ current excess loss.

For each ingredient config we simulate cosine/WSD curves, fit the MPL law and a
gamma=0 variant, and report whether gamma becomes NECESSARY and what value it takes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from reproduce_cosine_to_wsd import build_lrs, huber_log_residual  # noqa: E402

WARMUP = 2160

SCHED = {"cosine_24000": "cosine_24000.csv", "cosine_72000": "cosine_72000.csv",
         "wsd_20000_24000": "wsd_20000_24000.csv", "wsdcon_9": "wsdcon_9.csv"}
TRAIN = ["cosine_24000", "cosine_72000"]
TEST = ["wsd_20000_24000", "wsdcon_9"]


# ------------------------------ simulator ------------------------------------
def make_spectrum(a=-0.5, cexp=-0.5, bias_loss0=2.6, floor_scale=0.5, n_bias=140, n_noise=240):
    lam_b = np.geomspace(0.02, 5.0, n_bias)
    g = lam_b ** a
    g = g * (bias_loss0 / g.sum())
    lam_n = np.geomspace(0.05, 1.5e3, n_noise)          # cap so eta*lambda<1 (stable) even sharpened
    hw = lam_n ** cexp                                   # loss weight of each noise mode
    floor_unit = np.sum(hw * (3e-4 / (2.0 * lam_n)))     # const-eta floor at peak LR
    hw = hw * (floor_scale / floor_unit)
    return (lam_b, g), (lam_n, hw)


ETA_PEAK, ETA_END = 3e-4, 3e-5


def simulate(lrs, bias, noise, Lmin=2.5, noise_mode="const", eos_s=0.0):
    """eos_s: edge-of-stability coupling. The effective curvature of the NOISE modes
    (which set the annealing relaxation) tracks the LR as lam_eff = lam*(eta_peak/eta)^s
    -- the EoS signature sharpness ~ c/eta. Prediction: fitted gamma ~= eos_s."""
    lam_b, g = bias
    lam_n, hw = noise
    bfac = np.ones_like(lam_b)
    var = np.zeros_like(lam_n)
    loss = np.empty(len(lrs))
    for t in range(len(lrs)):
        eta = lrs[t]
        bfac = (1.0 - eta * lam_b) ** 2 * bfac                       # backbone: static spectrum
        eta_s = min(max(eta, ETA_END), ETA_PEAK)
        ln = lam_n * (ETA_PEAK / eta_s) ** eos_s                     # EoS: curvature anti-scales with LR
        inj = eta * eta
        if noise_mode == "loss" and t > 0:
            inj *= max(loss[t - 1] - Lmin, 1e-6)
        d2n = np.minimum((1.0 - eta * ln) ** 2, 1.0)
        var = np.minimum(d2n * var + inj, 1e8)
        loss[t] = Lmin + np.dot(g, bfac) + np.dot(hw, var)
    return loss


# ------------------------------ MPL fit (8 params incl. S_W) -----------------
def _ld_vec(lrs, steps, C, beta, gamma, nk=400):
    """Vectorised LD on a coarse decrement grid (fast; <1% err for fitting)."""
    T = len(lrs); S = np.cumsum(lrs)
    kj = np.unique(np.linspace(1, T - 1, min(nk, T - 1)).astype(int))
    prev = np.concatenate([[0], kj[:-1]])
    a = lrs[prev] - lrs[kj]                          # total decrement over each cell
    b = C * np.power(lrs[kj], -gamma)
    ref = S[prev]
    Ss = S[steps]
    gap = Ss[:, None] - ref[None, :]
    mask = (kj[None, :] <= steps[:, None]) & (gap > 0)
    kernel = 1.0 - np.power(1.0 + b[None, :] * np.maximum(gap, 0.0), -beta)
    return np.sum(np.where(mask, -a[None, :] * kernel, 0.0), axis=1)


def mpl_pred(p, lrs, steps):
    L0, A, alpha, B, C, beta, gamma, SW = p
    s1 = np.cumsum(lrs)[steps] + SW
    ld = _ld_vec(lrs, steps, C, beta, gamma)
    return L0 + A * np.power(s1, -alpha) + B * ld


BOUNDS = [(0, 10), (1e-8, 100), (0.05, 1.5), (1e-8, 1e6),
          (0.05, 20), (0.05, 1.5), (1e-4, 1.5), (0.0, 5.0)]


def fit(curves, free, init, stride=6):
    free = sorted(free); init = np.array(init, float)
    bnd = [BOUNDS[i] for i in free]
    sub = [(lrs, steps[::stride], loss[::stride]) for lrs, steps, loss in curves]

    def asm(pf):
        f = init.copy(); f[free] = pf; return f

    def obj(pf):
        pr, ys = [], []
        for lrs, steps, loss in sub:
            v = mpl_pred(asm(pf), lrs, steps)
            if not np.all(np.isfinite(v)) or np.any(v <= 0):
                return 1e18
            pr.append(v); ys.append(loss)
        return huber_log_residual(np.concatenate(ys), np.concatenate(pr))

    b = init[free]; best, bf = None, np.inf
    for x0 in [b, b * 0.7, b * 1.3, b * 1.6, b * 0.5]:
        r = minimize(obj, x0, method="L-BFGS-B", bounds=bnd, options={"maxiter": 250})
        if r.fun < bf:
            bf, best = r.fun, r.x
    return asm(best)


def mae(p, lrs, steps, loss):
    return float(np.mean(np.abs(loss - mpl_pred(p, lrs, steps))))


# ------------------------------ experiment -----------------------------------
def run_config(name, eos_s=0.0, a=-0.5, cexp=-0.5, noise_mode="const"):
    bias, noise = make_spectrum(a=a, cexp=cexp)
    curves = {}
    for nm, fn in SCHED.items():
        lrs = build_lrs(fn)
        loss = simulate(lrs, bias, noise, noise_mode=noise_mode, eos_s=eos_s)
        idx = np.unique(np.linspace(WARMUP + 200, len(lrs) - 1, 250).astype(int))
        curves[nm] = (lrs, idx, loss[idx])
    tr = [curves[n] for n in TRAIN]
    init = [2.0, 1.0, 0.4, 100.0, 2.0, 0.5, 0.5, 0.3]
    full = fit(tr, [0, 1, 2, 3, 4, 5, 6, 7], init)
    g0 = fit(tr, [0, 1, 2, 3, 4, 5, 7], (full.copy()).tolist()[:6] + [1e-4] + [full[7]])
    fte = np.mean([mae(full, *curves[n]) for n in TEST])
    gte = np.mean([mae(g0, *curves[n]) for n in TEST])
    need = gte > 1.3 * fte
    print(f"{name:18} eos_s={eos_s:4.2f} | gamma*={full[6]:.3f} | "
          f"full_te={fte:.4f} g0_te={gte:.4f} needs_gamma={'YES' if need else 'no '} | "
          f"gamma~=s? {'YES' if abs(full[6]-eos_s)<0.15 else 'no'}", flush=True)
    return dict(eos_s=eos_s, gamma=float(full[6]), full_test=fte, g0_test=gte, needs=need)


def main():
    print("=== KEY TEST: 曲率随LR标度 lam_eff~eta^(-s) 是否产生 gamma~=s ? ===", flush=True)
    rows = [run_config(f"eos s={s}", eos_s=s) for s in (0.0, 0.3, 0.5, 0.64, 0.8, 1.0)]
    print("\n--- 汇总: eos_s vs 拟合 gamma ---", flush=True)
    for r in rows:
        print(f"  s={r['eos_s']:.2f} -> gamma*={r['gamma']:.3f}", flush=True)
    import json
    (REPO.parent / "results" / "eos_gamma.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
