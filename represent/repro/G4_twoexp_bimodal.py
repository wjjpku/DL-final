"""
G4 -- PROPER test of prediction (v): a 2-exponential DropRelaxS kernel beats a
1-exponential kernel when the noise spectrum is BIMODAL (two well-separated
relaxation-rate clusters).

We attack the paper's open problem (sec 6 (a)): lambda_slow is *measured* not
computed. In the NQM we KNOW {lambda_i, s_i}, hence we know the theory rates
exactly. Noise-dominated AdamW => S-time relaxation rate per mode = 2*lambda_i/s_i.

Bimodal spectrum (two well-separated lambda/s clusters at 0.5 and 8.0):
    slow cluster:  6 modes, lambda=0.5,  s=1   -> lambda/s = 0.5, rate 2*lambda/s = 1
    fast cluster:  6 modes, lambda=64,   s=8   -> lambda/s = 8.0, rate 2*lambda/s = 16
  => lambda_i/s_i clusters at 0.5 (SLOW) and 8.0 (FAST)
  => true S-time rates lambda_slow_i = 2*lambda_i/s_i = {1.0 (slow), 16.0 (fast)}

WHY the fast cluster carries larger noise (s=8) than the bare concat(0.5, 8.0)
example: a fast mode equilibrates fast, so its *integrated* lag mass is tiny
(it relaxes in a few S-steps). With equal noise the residual is ~95% slow-cluster
mass, and a single exponential already fits that at R2~0.99 -- no room (this is
exactly why the naive concat(0.5,8.0)/s=1 attempt 'fails'). Scaling the fast
cluster's noise UP (keeping lambda/s = 8 fixed, so the rate stays 16 and
eta_eff*lambda = (eta/s)*lambda = 0.04 << 1 unchanged) raises its lag amplitude
(dL_eq/deta ~ sum s_i / 4) so BOTH timescales contribute comparable, overlapping
residual mass. Then a single exponential must compromise between rate 1 and rate
16 and R2_1 drops well below 1 -- leaving genuine room for the 2-exp mixture.

A single exponential must compromise between rate 1 and rate 16, so R2_1 sits
well below 1 -- leaving room. A two-exponential mixture recovers both clusters
and lifts R2 substantially.

Schedule: WSD with a FAST (short) linear decay so the LR drop is concentrated in
S-time (kicks BOTH clusters together), then a LONG low-LR tail so the slow
cluster (rate 1) has S-time room to relax while the fast cluster (rate 16) has
already decayed -- the only way both timescales show up in the residual.

Residual r = L_true - adiabatic baseline over the decay+tail window.
  1-exp: y ~ amp*droprelaxS(lambda_slow) + floor; best lambda_slow over a grid -> R2_1
  2-exp: y ~ a1*droprelaxS(lam1) + a2*droprelaxS(lam2) + floor; grid+scipy -> R2_2
         (joint linear amplitudes => optimal mixture for each (lam1,lam2) pair;
          w1 = a1/(a1+a2), amp = a1+a2.)

Speed: every 2-exp kernel is a linear combination of single-exp kernels, so we
precompute droprelaxS for ALL grid rates ONCE, then all fits are pure linear
algebra.

HEADLINE if delta = R2_2 - R2_1 > 0.03 AND recovered (lam1,lam2) match {1, 16}.
success if delta > 0.03.
"""
import sys, os, json
sys.path.insert(0, r'c:/Users/21100/Desktop/represent/repro')
import numpy as np
from scipy.optimize import minimize
from engine import (adamw_nqm, nqm_linear_Leq, droprelaxS,
                    const_lrs, cumS)

RESULTS = r'c:/Users/21100/Desktop/represent/results/G4.json'


def r2_of(y, pred):
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2) + 1e-30
    return float(1 - ss_res / ss_tot)


def lstsq_fit(cols, y):
    """y ~ sum_j coef_j * cols[:,j] + floor (floor = last column of ones).
    Returns (coef_without_floor, floor, pred, r2)."""
    A = np.column_stack(cols + [np.ones_like(y)])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    return coef[:-1], float(coef[-1]), pred, r2_of(y, pred)


def main():
    np.set_printoptions(precision=4)

    # ------------------------------------------------------------------
    # Bimodal spectrum: two well-separated lambda/s clusters (task spec)
    # ------------------------------------------------------------------
    # slow cluster: lambda=0.5, s=1 (ratio 0.5, rate 1)
    # fast cluster: lambda=64,  s=8 (ratio 8.0, rate 16); noise scaled up so its
    #               lag mass is comparable to the slow cluster (see header).
    s_fast = 8.0
    lambdas = np.concatenate([np.full(6, 0.5), np.full(6, 8.0 * s_fast)])
    sigmas = np.concatenate([np.ones(6), np.full(6, s_fast)])
    rate = lambdas / sigmas                       # preconditioned curvature
    true_rates = np.unique(2.0 * rate)            # theory S-time rates {1, 16}
    d = lambdas.shape[0]
    print('== G4: 2-exp vs 1-exp on a BIMODAL spectrum ==')
    print(f'  d={d}  lambda/s clusters: {np.unique(rate)}  '
          f'-> true 2*lambda/s rates: {true_rates}')

    # ------------------------------------------------------------------
    # WSD schedule: short FAST linear decay + long low-LR tail.
    #   eta_eff*lam_max = (peak/s)*8 = 0.04 << 1  (locally quadratic, far from EoS).
    # Short decay -> impulse-like kick of both clusters. Long, not-too-low tail ->
    # plenty of S-time so the slow cluster (rate 1) relaxes far while the fast
    # cluster (rate 16) is already gone => clear two-stage residual a single
    # exponential cannot match.
    # ------------------------------------------------------------------
    total = 9000
    decay_start = 3000
    decay_len = 300                 # fast decay (3.3% of run) but wide enough to
                                    # spread the LR drops over S-time so the two
                                    # cluster transients overlap (single-exp fails)
    peak = 5.0e-3
    end = 3.0e-4
    n_warm = 400
    etas = const_lrs(total, peak, n_warm)
    dec = np.arange(decay_start, decay_start + decay_len)
    frac = (dec - decay_start) / decay_len
    etas[decay_start:decay_start + decay_len] = peak * (1 - frac) + end * frac
    etas[decay_start + decay_len:] = end
    eta_peak = float(etas.max())
    print(f'  schedule: WSD total={total} decay_start={decay_start} '
          f'decay_len={decay_len} peak={peak:g} end={end:g} n_warm={n_warm}')
    print(f'  eta_eff*lam max @ peak = {((peak/sigmas)*lambdas).max():.4f}  (<<1 required)')

    # ------------------------------------------------------------------
    # True loss from real AdamW on the NQM
    # ------------------------------------------------------------------
    loss = adamw_nqm(lambdas, sigmas, etas, beta1=0.9, beta2=0.999, eps=1e-8,
                     n_rep=4000, seed=0)

    # Adiabatic / quasi-static baseline L_eq(eta_t) (closed-form noise-dominated)
    L_eq = np.array([nqm_linear_Leq(lambdas, sigmas, e) for e in etas])

    if not np.all(np.isfinite(loss)) or not np.all(np.isfinite(L_eq)):
        raise RuntimeError('NaN/Inf in loss or L_eq')

    resid = loss - L_eq

    # Fit window: decay onset -> end (decay + tail), where the lag develops
    t0 = decay_start
    sl = slice(t0, total)
    y = resid[sl]
    print(f'  residual on window: min={y.min():.3e} max={y.max():.3e} '
          f'mean={y.mean():.3e}')

    Scum = cumS(etas)
    S_tail = float(Scum[-1] - Scum[decay_start + decay_len])
    print(f'  post-decay S-tail = {S_tail:.4f}  '
          f'-> slow exp(-1*S_tail)={np.exp(-1.0*S_tail):.3f}, '
          f'fast exp(-16*S_tail)={np.exp(-16.0*S_tail):.3e}')

    # ------------------------------------------------------------------
    # Precompute single-exp kernels on the grid ONCE (on the fit window).
    # ------------------------------------------------------------------
    grid = np.geomspace(0.05, 200.0, 80)
    Kgrid = np.array([droprelaxS(etas, ls)[sl] for ls in grid])   # (G, W)
    if not np.all(np.isfinite(Kgrid)):
        raise RuntimeError('NaN/Inf in kernel grid')

    # ---- 1-exp ----
    best1 = None
    for j, ls in enumerate(grid):
        coef, fl, pred, r2 = lstsq_fit([Kgrid[j]], y)
        if best1 is None or r2 > best1['r2']:
            best1 = dict(lambda_slow=float(ls), amp=float(coef[0]), floor=fl, r2=r2)
    # scipy refine 1-exp
    def neg_r2_1(p):
        K = droprelaxS(etas, np.exp(p[0]))[sl]
        _, _, _, r2 = lstsq_fit([K], y)
        return -r2
    try:
        res1 = minimize(neg_r2_1, [np.log(best1['lambda_slow'])],
                        method='Nelder-Mead',
                        options=dict(maxiter=2000, xatol=1e-4, fatol=1e-9))
        lsr = float(np.exp(res1.x[0]))
        coef, fl, pred, r2 = lstsq_fit([droprelaxS(etas, lsr)[sl]], y)
        if r2 > best1['r2']:
            best1 = dict(lambda_slow=lsr, amp=float(coef[0]), floor=fl, r2=r2)
    except Exception as e:
        print('  [warn] 1-exp refine failed:', e)
    R2_1 = best1['r2']
    print(f'  1-exp: lambda_slow={best1["lambda_slow"]:.4g}  '
          f'amp={best1["amp"]:.3e}  R2_1={R2_1:.5f}')

    # ---- 2-exp (joint linear amplitudes; optimal mixture per pair) ----
    best2 = None
    G = len(grid)
    for i in range(G):
        for k in range(i + 1, G):
            coef, fl, pred, r2 = lstsq_fit([Kgrid[i], Kgrid[k]], y)
            if best2 is None or r2 > best2['r2']:
                best2 = dict(lam1=float(grid[i]), lam2=float(grid[k]),
                             a1=float(coef[0]), a2=float(coef[1]),
                             floor=fl, r2=r2)
    # scipy refine on (log lam1, log lam2); amplitudes stay linear (joint LSQ)
    def neg_r2_2(p):
        K1 = droprelaxS(etas, np.exp(p[0]))[sl]
        K2 = droprelaxS(etas, np.exp(p[1]))[sl]
        _, _, _, r2 = lstsq_fit([K1, K2], y)
        return -r2
    x0 = [np.log(best2['lam1']), np.log(best2['lam2'])]
    try:
        res2 = minimize(neg_r2_2, x0, method='Nelder-Mead',
                        options=dict(maxiter=4000, xatol=1e-4, fatol=1e-10))
        l1r, l2r = float(np.exp(res2.x[0])), float(np.exp(res2.x[1]))
        coef, fl, pred, r2 = lstsq_fit(
            [droprelaxS(etas, l1r)[sl], droprelaxS(etas, l2r)[sl]], y)
        if r2 > best2['r2']:
            best2 = dict(lam1=l1r, lam2=l2r, a1=float(coef[0]),
                         a2=float(coef[1]), floor=fl, r2=r2)
    except Exception as e:
        print('  [warn] 2-exp refine failed:', e)

    # canonicalize: lam1 = slow (smaller rate), lam2 = fast (larger rate)
    if best2['lam1'] > best2['lam2']:
        best2['lam1'], best2['lam2'] = best2['lam2'], best2['lam1']
        best2['a1'], best2['a2'] = best2['a2'], best2['a1']
    amp = best2['a1'] + best2['a2']
    w1 = best2['a1'] / amp if amp != 0 else float('nan')
    best2['amp'] = float(amp)
    best2['w1'] = float(w1)
    R2_2 = best2['r2']
    print(f'  2-exp: lam1={best2["lam1"]:.4g} lam2={best2["lam2"]:.4g} '
          f'w1={best2["w1"]:.3f}  amp={best2["amp"]:.3e}  R2_2={R2_2:.5f}')

    delta = R2_2 - R2_1
    rec_slow, rec_fast = best2['lam1'], best2['lam2']
    true_slow, true_fast = float(true_rates[0]), float(true_rates[1])
    rel_err_slow = abs(rec_slow - true_slow) / true_slow
    rel_err_fast = abs(rec_fast - true_fast) / true_fast
    rates_match = bool(rel_err_slow < 0.5 and rel_err_fast < 0.5)

    success = bool(delta > 0.03)
    headline = bool(delta > 0.03 and rates_match)
    print(f'  delta = R2_2 - R2_1 = {delta:.5f}')
    print(f'  recovered: slow={rec_slow:.3f} (true {true_slow}, '
          f'rel_err={rel_err_slow:.2f}); fast={rec_fast:.3f} '
          f'(true {true_fast}, rel_err={rel_err_fast:.2f})')
    print(f'  rates_match={rates_match}  success(delta>0.03)={success}  '
          f'headline={headline}')

    out = dict(
        experiment='G4_twoexp_bimodal',
        spectrum=dict(d=d, lambdas=lambdas.tolist(), sigmas=sigmas.tolist(),
                      rate_clusters=np.unique(rate).tolist(),
                      true_Stime_rates=true_rates.tolist()),
        schedule=dict(kind='wsd', total=total, decay_start=decay_start,
                      decay_len=decay_len, peak=peak, end=end, n_warm=n_warm,
                      eta_peak=eta_peak, S_tail=S_tail),
        fit_window=dict(t0=int(t0), t1=int(total)),
        residual_stats=dict(min=float(y.min()), max=float(y.max()),
                            mean=float(y.mean())),
        fit_1exp=best1,
        fit_2exp=best2,
        R2_1=float(R2_1),
        R2_2=float(R2_2),
        delta=float(delta),
        recovered=dict(slow=float(rec_slow), fast=float(rec_fast)),
        true_clusters=dict(slow=true_slow, fast=true_fast),
        rel_err=dict(slow=float(rel_err_slow), fast=float(rel_err_fast)),
        rates_match=rates_match,
        success=success,
        headline=headline,
        paper_prediction='(v) 2-exp kernel beats 1-exp when spectrum is a mixture',
    )
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'  saved -> {RESULTS}')
    return out


if __name__ == '__main__':
    main()
