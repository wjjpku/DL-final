"""
engine.py  --  Core numerical engine for reproducing & extending

  "Learning-Rate Schedules Are Not Adiabatic:
   A Rate-Dependent Correction for Loss-Curve Prediction" (Jiaju Wu)

Everything here is numpy-only (CPU, deterministic via seeds). Three pieces:

  1. adamw_nqm(...)      -- from-scratch AdamW on a Noisy Quadratic Model (NQM).
                           This is the paper's controlled cross-check (sec.4 i').
                           We run *real* AdamW (real m, v EMAs, real preconditioner)
                           on an ensemble of replicas, so regime behaviour EMERGES
                           rather than being assumed.
  2. nqm_linear_Leq(...) -- closed-form equilibrium loss L_eq(eta) and relaxation
                           time tau for the noise-dominated linear (SGD-with-eta/s)
                           approximation. Used as ground-truth cross-check.
  3. droprelaxS(...)     -- the paper's rate-dependent kernel  Eq.(4):
                              sum_{t'} exp(-lambda_slow (S_t - S_t')) * drop_t'
                           plus MPL law, fitting helpers, and tau-measurement.

Theory recap (Hessian eigenbasis, mode i: curvature lambda_i, grad-noise std s_i):
  * SGD step eta:        theta+ = (1-eta*lam)theta - eta*xi
                         V+ = (1-eta*lam)^2 V + eta^2 s^2
                         V* = eta^2 s^2 / (1-(1-eta lam)^2) ~ eta s^2/(2 lam)
                         relax multiplier (1-eta lam)^2 ~ e^{-2 eta lam}  => tau ~ 1/(eta lam)
  * AdamW, noise-dominated (v*~s^2): effective step eta_eff = eta/s, so
                         V* ~ eta s/(2 lam),  L_eq ~ (eta/4) sum_i s_i,
                         per-step rate 2 eta_eff lam = 2 eta lam/s  => tau ~ 1/eta (p=1)
                         S-time rate lambda_slow = 2 lam/s (preconditioned curvature).
  * signal-dominated (v*~lam^2 V): self-consistent => tau ~ eta^{-2/3} (p=2/3).
  * amplitude identity: sum_i w_i = dL_eq/deta.
"""
import numpy as np
from scipy.optimize import curve_fit


# ----------------------------------------------------------------------------
# 1. From-scratch AdamW on a noisy quadratic (ensemble of replicas)
# ----------------------------------------------------------------------------
def adamw_nqm(lambdas, sigmas, etas, beta1=0.9, beta2=0.999, eps=1e-8,
              n_rep=4000, seed=0, theta0=None, weight_decay=0.0,
              dtype=np.float32):
    """Simulate AdamW on L(theta) = 0.5 * sum_i lambda_i * theta_i^2.

    Stochastic gradient per mode i, replica r:  g = lambda_i theta + xi,
    xi ~ N(0, sigma_i^2)  (gradient noise; independent across modes/replicas/steps).

    Args
      lambdas : (d,) curvatures (>0)
      sigmas  : (d,) gradient-noise std per mode
      etas    : (T,) per-step learning-rate schedule
      n_rep   : number of independent replicas averaged for the expected loss
      theta0  : optional (d,) or (n_rep,d) initial deviation; default zeros
    Returns
      loss : (T,) expected loss 0.5 sum_i lambda_i * mean_rep(theta_i^2),
             recorded BEFORE each step (loss[0] is the loss at theta0).
    """
    rng = np.random.default_rng(seed)
    lambdas = np.asarray(lambdas, dtype)
    sigmas = np.asarray(sigmas, dtype)
    etas = np.asarray(etas, dtype)
    d = lambdas.shape[0]
    T = etas.shape[0]

    if theta0 is None:
        theta = np.zeros((n_rep, d), dtype)
    else:
        theta = np.broadcast_to(np.asarray(theta0, dtype), (n_rep, d)).copy()
    m = np.zeros((n_rep, d), dtype)
    v = np.zeros((n_rep, d), dtype)
    loss = np.empty(T, np.float64)
    lam = lambdas[None, :]
    sig = sigmas[None, :]
    b1, b2 = dtype(beta1), dtype(beta2)
    for t in range(T):
        loss[t] = 0.5 * (np.mean(theta * theta, axis=0, dtype=np.float64) @ lambdas.astype(np.float64))
        g = lam * theta + rng.standard_normal((n_rep, d), dtype=dtype) * sig
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * (g * g)
        mhat = m / (1 - beta1 ** (t + 1))
        vhat = v / (1 - beta2 ** (t + 1))
        theta = theta - etas[t] * (mhat / (np.sqrt(vhat) + eps) + weight_decay * theta)
    return loss


def equilibrate(lambdas, sigmas, eta, n_steps=4000, **kw):
    """Run at constant eta and return final (theta, m, v) state by re-implementing
    the loop but returning state. Used to start a relaxation from true equilibrium."""
    rng = np.random.default_rng(kw.get("seed", 0))
    dtype = kw.get("dtype", np.float32)
    n_rep = kw.get("n_rep", 4000)
    beta1 = kw.get("beta1", 0.9); beta2 = kw.get("beta2", 0.999); eps = kw.get("eps", 1e-8)
    lambdas = np.asarray(lambdas, dtype); sigmas = np.asarray(sigmas, dtype)
    d = lambdas.shape[0]
    theta = np.zeros((n_rep, d), dtype); m = np.zeros((n_rep, d), dtype); v = np.zeros((n_rep, d), dtype)
    lam = lambdas[None, :]; sig = sigmas[None, :]
    b1, b2 = dtype(beta1), dtype(beta2)
    for t in range(n_steps):
        g = lam * theta + rng.standard_normal((n_rep, d), dtype=dtype) * sig
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * (g * g)
        mhat = m / (1 - beta1 ** (t + 1)); vhat = v / (1 - beta2 ** (t + 1))
        theta = theta - eta * (mhat / (np.sqrt(vhat) + eps))
    return theta, m, v


def adamw_nqm_from_state(lambdas, sigmas, etas, state, beta1=0.9, beta2=0.999,
                         eps=1e-8, seed=1, dtype=np.float32, t_offset=10**6):
    """Continue AdamW from a given (theta,m,v) state (e.g. post-equilibration),
    so bias-correction is effectively off (t_offset large). Returns loss trace."""
    rng = np.random.default_rng(seed)
    lambdas = np.asarray(lambdas, dtype); sigmas = np.asarray(sigmas, dtype)
    etas = np.asarray(etas, dtype)
    theta, m, v = (x.copy() for x in state)
    T = etas.shape[0]
    loss = np.empty(T, np.float64)
    lam = lambdas[None, :]; sig = sigmas[None, :]
    b1, b2 = dtype(beta1), dtype(beta2)
    for t in range(T):
        loss[t] = 0.5 * (np.mean(theta * theta, axis=0, dtype=np.float64) @ lambdas.astype(np.float64))
        g = lam * theta + rng.standard_normal(theta.shape, dtype=dtype) * sig
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * (g * g)
        # bias correction with large offset -> ~1
        theta = theta - etas[t] * (m / (np.sqrt(v) + eps))
    return loss


# ----------------------------------------------------------------------------
# 2. Closed-form noise-dominated (AdamW ~ SGD with eta_eff = eta/s) cross-check
# ----------------------------------------------------------------------------
def nqm_linear_Leq(lambdas, sigmas, eta):
    """Exact equilibrium loss for the linear noise-dominated approximation
    (AdamW with eta_eff = eta/s).  V* = eta_eff^2 s^2 / (1-(1-eta_eff lam)^2)."""
    lambdas = np.asarray(lambdas, float); sigmas = np.asarray(sigmas, float)
    eta_eff = eta / sigmas
    a = 1 - eta_eff * lambdas
    Vstar = (eta_eff ** 2) * (sigmas ** 2) / (1 - a ** 2)
    return 0.5 * np.sum(lambdas * Vstar)


def nqm_linear_tau(lambdas, sigmas, eta):
    """Per-mode relaxation time (in steps) of the linear approximation,
    tau_i = -1/(2 ln(1-eta_eff lam_i)); dominated by slowest mode."""
    lambdas = np.asarray(lambdas, float); sigmas = np.asarray(sigmas, float)
    eta_eff = eta / sigmas
    a = 1 - eta_eff * lambdas
    tau = -1.0 / (2 * np.log(np.abs(a)))
    return tau


# ----------------------------------------------------------------------------
# 3. DropRelaxS kernel, MPL law, fitting & tau measurement
# ----------------------------------------------------------------------------
def cumS(etas):
    """Cumulative LR S(t) = sum_{i<=t} eta_i (1-indexed cumulative)."""
    return np.cumsum(np.asarray(etas, float))


def drops(etas):
    """Per-step positive LR decrements drop_t = max(eta_{t-1}-eta_t, 0)."""
    etas = np.asarray(etas, float)
    d = np.zeros_like(etas)
    d[1:] = np.maximum(etas[:-1] - etas[1:], 0.0)
    return d


def droprelaxS(etas, lambda_slow):
    """Paper Eq.(4) kernel (unit amplitude):
        K(t) = sum_{t'<=t} exp(-lambda_slow (S_t - S_t')) drop_t'.
    Computed with an O(T) recursion in S-time:
        K_t = exp(-lambda_slow (S_t - S_{t-1})) K_{t-1} + drop_t.
    Returns K (T,)."""
    etas = np.asarray(etas, float)
    S = cumS(etas)
    dr = drops(etas)
    K = np.zeros_like(etas)
    acc = 0.0
    Sprev = 0.0
    for t in range(len(etas)):
        decay = np.exp(-lambda_slow * (S[t] - Sprev))
        acc = acc * decay + dr[t]
        K[t] = acc
        Sprev = S[t]
    return K


def droprelaxS_twoexp(etas, lam1, lam2, w1):
    """Two-exponential mixture kernel (prediction v: spectral mixture)."""
    return w1 * droprelaxS(etas, lam1) + (1 - w1) * droprelaxS(etas, lam2)


# ---- MPL law (Luo et al. 2025) ----
def mpl_loss(etas, L0, A, alpha, B, C, beta, gamma, S_warmup=0.0):
    """Multi-Power Law.  L = L0 + A (S+Sw)^{-alpha} - B sum_k drop_k * G(eta_k^{-gamma} Sk(t))
    with G(x)=1-(1+Cx)^{-beta}, Sk(t)=S(t)-S(k).  (Sign of LD term: loss DROP.)"""
    etas = np.asarray(etas, float)
    S = cumS(etas)
    dr = drops(etas)
    T = len(etas)
    main = L0 + A * (S + S_warmup) ** (-alpha)
    # loss-drop term
    LD = np.zeros(T)
    # eta_k for decrement at step k is eta_k (post). Use etas as eta_k.
    nz = np.where(dr > 0)[0]
    for k in nz:
        Sk = S - S[k]                       # Sk(t) = S(t)-S(k); valid for t>=k
        Sk = np.where(np.arange(T) >= k, Sk, 0.0)
        x = (etas[k] ** (-gamma)) * Sk
        G = 1 - (1 + C * x) ** (-beta)
        LD += B * dr[k] * G
    return main - LD


def _binned_decrements(etas, max_dec=600):
    """Return (S_full, eta_k, S_k, drop_k) for the loss-drop sum. If there are more than
    max_dec decrements (e.g. cosine), bin them by cumulative LR S (drop-weighted means).
    The MPL kernel G is smooth in S, so this is accurate and ~10x faster for dense schedules."""
    etas = np.asarray(etas, float)
    S = cumS(etas); dr = drops(etas); nz = np.where(dr > 0)[0]
    if nz.size <= max_dec:
        return S, etas[nz], S[nz], dr[nz]
    Snz = S[nz]
    edges = np.linspace(Snz[0], Snz[-1] + 1e-12, max_dec + 1)
    idx = np.clip(np.searchsorted(edges, Snz, side="right") - 1, 0, max_dec - 1)
    dsum = np.bincount(idx, weights=dr[nz], minlength=max_dec)
    Swt = np.bincount(idx, weights=dr[nz] * Snz, minlength=max_dec)
    ewt = np.bincount(idx, weights=dr[nz] * etas[nz], minlength=max_dec)
    keep = dsum > 0
    return S, ewt[keep] / dsum[keep], Swt[keep] / dsum[keep], dsum[keep]


def mpl_loss_at(etas, query_steps, L0, A, alpha, B, C, beta, gamma, S_warmup=0.0, max_dec=600):
    """Vectorized MPL prediction evaluated ONLY at `query_steps` (fast for dense-decrement
    schedules like cosine). Decrements are binned to <=max_dec by S for speed."""
    S, eta_k, S_k, drop_k = _binned_decrements(etas, max_dec)
    q = np.asarray(query_steps, int)
    main = L0 + A * (S[q] + S_warmup) ** (-alpha)
    if eta_k.size == 0:
        return main
    M = np.maximum(S[q][:, None] - S_k[None, :], 0.0)     # (Q, m)
    x = (eta_k[None, :] ** (-gamma)) * M
    G = 1.0 - (1.0 + C * x) ** (-beta)
    LD = B * (drop_k[None, :] * G).sum(axis=1)
    return main - LD


# ---- tau measurement from a relaxation transient ----
def measure_tau(loss, t0, fit_len=None, floor=None, t=None):
    """Fit loss[t0:] - L_floor ~ amp * exp(-(t-t0)/tau).
    If floor is None it is fit jointly. Returns dict(tau, amp, floor, r2).
    `t` (optional): explicit time axis (same length as loss) for IRREGULARLY-sampled data;
    if given, tau is in those time units (e.g. steps), not sample indices."""
    loss = np.asarray(loss, float)
    T = len(loss)
    t1 = T if fit_len is None else min(T, t0 + fit_len)
    if t is None:
        t = np.arange(t0, t1) - t0
    else:
        t = np.asarray(t, float)[t0:t1] - np.asarray(t, float)[t0]
    y = loss[t0:t1]
    tau0 = max((t[-1] - t[0]) / 5.0, 1.0)   # in the ACTUAL fit-axis units (steps if explicit t)
    if floor is None:
        def f(t, amp, tau, fl):
            return fl + amp * np.exp(-t / tau)
        amp0 = max(y[0] - y[-1], 1e-9)
        p0 = [amp0, tau0, y[-1]]
        bounds = ([0, 1e-6, -np.inf], [np.inf, np.inf, np.inf])
        try:
            popt, _ = curve_fit(f, t, y, p0=p0, bounds=bounds, maxfev=20000)
        except Exception:
            return dict(tau=np.nan, amp=np.nan, floor=np.nan, r2=np.nan)
        amp, tau, fl = popt
        pred = f(t, *popt)
    else:
        fl = floor
        def f(t, amp, tau):
            return fl + amp * np.exp(-t / tau)
        amp0 = max(y[0] - fl, 1e-9)
        p0 = [amp0, tau0]
        try:
            popt, _ = curve_fit(f, t, y, p0=p0,
                                bounds=([0, 1e-6], [np.inf, np.inf]), maxfev=20000)
        except Exception:
            return dict(tau=np.nan, amp=np.nan, floor=fl, r2=np.nan)
        amp, tau = popt
        pred = f(t, *popt)
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2) + 1e-30
    return dict(tau=float(tau), amp=float(amp), floor=float(fl), r2=float(1 - ss_res / ss_tot))


def fit_powerlaw(x, y):
    """Fit y = c * x^{-p}; return (p, c, r2) via log-log least squares."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    lx, ly = np.log(x), np.log(y)
    A = np.vstack([lx, np.ones_like(lx)]).T
    coef, *_ = np.linalg.lstsq(A, ly, rcond=None)
    slope, intercept = coef
    p = -slope
    pred = A @ coef
    r2 = 1 - np.sum((ly - pred) ** 2) / (np.sum((ly - ly.mean()) ** 2) + 1e-30)
    return float(p), float(np.exp(intercept)), float(r2)


# ---- LR schedules (match MPL public protocol) ----
def warmup(peak, n_warm, steps):
    w = np.minimum(np.arange(1, steps + 1) / max(n_warm, 1), 1.0)
    return peak * w


def cosine_lrs(total, peak=3e-4, end=3e-5, n_warm=2160):
    eta = np.empty(total)
    for t in range(total):
        if t < n_warm:
            eta[t] = peak * (t + 1) / n_warm
        else:
            prog = (t - n_warm) / max(total - n_warm, 1)
            eta[t] = end + 0.5 * (peak - end) * (1 + np.cos(np.pi * prog))
    return eta


def const_lrs(total, peak=3e-4, n_warm=2160):
    eta = np.full(total, peak)
    eta[:n_warm] = peak * (np.arange(1, n_warm + 1) / n_warm)
    return eta


def wsd_lrs(total, decay_start, peak=3e-4, end=3e-5, n_warm=2160):
    """warmup-stable-decay: stable until decay_start, then sqrt-ish/linear decay to end."""
    eta = const_lrs(total, peak, n_warm)
    dec = np.arange(decay_start, total)
    frac = (dec - decay_start) / max(total - decay_start, 1)
    eta[decay_start:] = peak * (1 - frac) + end * frac   # linear decay (WSD-linear default)
    return eta


def wsd_sqrt_lrs(total, decay_start, peak=3e-4, end=3e-5, n_warm=2160):
    eta = const_lrs(total, peak, n_warm)
    dec = np.arange(decay_start, total)
    frac = (dec - decay_start) / max(total - decay_start, 1)
    eta[decay_start:] = (np.sqrt(peak) * (1 - frac) + np.sqrt(end) * frac) ** 2
    return eta


def two_stage_lrs(total, peak=3e-4, lr_b=9e-5, step=8000, n_warm=2160):
    """wsdcon: stable peak then drop to constant lr_b at `step`."""
    eta = const_lrs(total, peak, n_warm)
    eta[step:] = lr_b
    return eta


if __name__ == "__main__":
    # ---- self-test ----
    np.set_printoptions(precision=4)
    d = 8
    rng = np.random.default_rng(0)
    lambdas = np.geomspace(0.5, 5.0, d)
    sigmas = np.full(d, 1.0)            # noise-dominated, uniform noise
    print("== self-test: tau ~ 1/eta (AdamW noise-dominated) ==")
    taus = []
    etalist = [3e-3, 6e-3, 1.2e-2, 2.4e-2]
    for eta_lo in etalist:
        st = equilibrate(lambdas, sigmas, eta=3e-2, n_steps=3000, n_rep=3000, seed=0)
        loss = adamw_nqm_from_state(lambdas, sigmas, np.full(4000, eta_lo), st, seed=1)
        r = measure_tau(loss, t0=0)
        taus.append(r["tau"])
        print(f"  eta={eta_lo:.4f}  tau={r['tau']:.1f}  r2={r['r2']:.3f}")
    p, c, r2 = fit_powerlaw(etalist, taus)
    print(f"  power-law fit: tau ~ eta^-{p:.3f}  (predict p=1)  r2={r2:.3f}")

    print("== self-test: droprelaxS kernel & dLeq/deta ==")
    eta = wsd_lrs(6000, 4000)
    K = droprelaxS(eta, lambda_slow=10.0)
    print(f"  K max={K.max():.4e}, K[end]={K[-1]:.4e}")
    de = 1e-6
    dLeq = (nqm_linear_Leq(lambdas, sigmas, 3e-2 + de) - nqm_linear_Leq(lambdas, sigmas, 3e-2 - de)) / (2 * de)
    print(f"  dLeq/deta (numeric) = {dLeq:.4f},  (1/4)sum s = {0.25*np.sum(sigmas):.4f}")
    print("self-test done.")
