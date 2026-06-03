#!/usr/bin/env python3
"""River-Valley loss law (RV-EoS) — theory-first derivation, then experiment.

Mechanism (corrected from the GD edge-of-stability mistake; see
docs/core/river_valley_derivation.md):

  Adam/transformer pretraining lives in a *river valley* (Wen et al. 2024):
  a flat "river" direction carrying true progress, and sharp "mountain"
  directions across which the iterate oscillates. The observed loss is

      L(t) = L0 + A * S(t)^-alpha            # river-bottom progress (bias)
                 + (B/2) * eta_t / (2 - eta_t H_t)   # mountain oscillation penalty

  The penalty is the *exact* stationary excess loss of discrete SGD on a
  quadratic mode of curvature H_t with step eta_t and gradient noise:

      Var(x) = eta^2 sigma^2 / (1-(1-eta H)^2) = eta sigma^2 / (H (2 - eta H))
      P_osc  = (1/2) H Var = (sigma^2/2) * eta / (2 - eta H).

  Edge of stability lives here automatically: eta H -> 2 makes P_osc blow up
  (elevated stable-phase loss); eta -> 0 makes P_osc -> 0 (decay "reveals
  accumulated progress").

  The sharpness H_t is NOT instantaneous. Progressive sharpening relaxes it
  toward the adaptive-EoS edge value u*/eta_t with a timescale tau (in steps):

      H_{t+1} = H_t + (1/tau) (u*/eta_t - H_t),     u* < 2.

  - tau -> 0 (instant tracking): eta H == u*, P_osc ∝ eta -> MPL's gamma=0
    limit (a pure instantaneous floor; this is the case the floor experiment
    already falsified).
  - tau > 0 (lag): a sharp eta drop outruns H, so eta H departs from u* and
    P_osc stops being ∝ eta -> history dependence. Cosine (gradual) keeps H on
    the edge (small residual); WSD (sharp) does not (residual). This lag is
    the physical origin of MPL's empirical eta_k^{-gamma} kernel-speed factor.

7 free parameters (same count as MPL), every one physical:
  L0, A, alpha  river/bias backbone
  B             penalty amplitude (= sigma^2)
  ustar         adaptive-EoS edge ratio (MUST fit < 2 to be physical)
  tau           progressive-sharpening timescale (steps; MUST fit > 0)
  H0            sharpness at end of warmup

Honest transfer protocol (identical to MPL baseline): fit on the two cosine
curves only, predict the five held-out WSD curves. Multi-restart L-BFGS-B,
same huber-log objective as fit_mpl. No test-set tuning.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.signal import lfilter

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from reproduce_cosine_to_wsd import (  # noqa: E402
    Curve, load_curve, compute_s1, huber_log_residual, metrics,
    mpl_predict, MPL_PRECOMPUTED_INIT,
    TRAIN_CURVES, TEST_CURVES, SCALES, WARMUP, PEAK_LR,
)

VCAP = 1.98  # clip eta*H below the hard edge=2 for numerical safety


def sharpness_track(lrs: np.ndarray, ustar: float, tau: float, H0: float) -> np.ndarray:
    """Integrate H_{t+1}=H_t+(1/tau)(u*/eta_t - H_t) per step, post-warmup.

    The recurrence is linear: H[t] = a*H[t-1] + (1-a)*target[t] with a=1-1/tau
    -- a first-order EMA of the AEoS target u*/eta_t. Vectorised in C via
    scipy.signal.lfilter (was a python loop -> billions of iters across the
    optimiser; this is the same arithmetic, ~1000x faster).

    H is held at H0 through warmup (eta ramps from 0 there; the AEoS target
    u*/eta is meaningless until eta reaches its plateau)."""
    T = len(lrs)
    eta_floor = PEAK_LR * 1e-4
    start = min(max(WARMUP, 1), T)
    eta_post = np.maximum(lrs[start:], eta_floor)
    target = ustar / eta_post
    a = 1.0 - 1.0 / tau
    # y[n] = (1-a) x[n] + a y[n-1], with y[-1] = H0  ->  zi = a*H0
    y, _ = lfilter([1.0 - a], [1.0, -a], target, zi=[a * H0])
    H = np.empty(T, dtype=np.float64)
    H[start:] = y
    H[:start] = y[0] if y.size else H0
    return H


def rv_predict(params: np.ndarray, curve: Curve) -> np.ndarray:
    L0, A, alpha, B, ustar, tau, H0 = params
    s1 = compute_s1(curve)
    H = sharpness_track(curve.lrs, ustar, tau, H0)
    v = np.clip(curve.lrs * H, 0.0, VCAP)
    penalty_full = (B * 0.5) * curve.lrs / (2.0 - v)
    penalty = penalty_full[curve.step]
    return L0 + A * np.power(s1, -alpha) + penalty


def fit_rv(curves: list[Curve], scale: str) -> tuple[np.ndarray, float]:
    min_loss = min(float(c.loss.min()) for c in curves)
    # eta_peak * H_peak = u* at the edge -> H_peak ~ u*/PEAK_LR ~ 2/3e-4 ~ 6.7e3
    Hpk = 1.8 / PEAK_LR
    inits = []
    for ustar in (1.2, 1.6, 1.9):
        for tau in (50.0, 300.0, 1500.0):
            inits.append(np.array([min_loss - 0.05, 0.5, 0.5, 10.0, ustar, tau, Hpk]))
    bounds = [
        (0.0, 10.0),      # L0
        (1e-8, 100.0),    # A
        (1e-4, 3.0),      # alpha
        (1e-10, 1e5),     # B = sigma^2
        (0.1, 1.99),      # ustar  (physical: < 2)
        (1.0, 1e5),       # tau    (physical: > 0)
        (10.0, 5e4),      # H0
    ]

    def objective(params: np.ndarray) -> float:
        pred_all, loss_all = [], []
        for c in curves:
            pred = rv_predict(params, c)
            if np.any(~np.isfinite(pred)) or np.any(pred <= 0):
                return 1e18
            pred_all.append(pred)
            loss_all.append(c.loss)
        return huber_log_residual(np.concatenate(loss_all), np.concatenate(pred_all))

    best_x, best_f = None, float("inf")
    for init in inits:
        res = minimize(objective, x0=init, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 600, "ftol": 1e-11})
        if res.fun < best_f:
            best_f, best_x = float(res.fun), res.x
    return best_x, best_f


def main() -> None:
    print("=" * 78)
    print("RV-EoS (river-valley) vs MPL — honest cosine->WSD transfer")
    print("=" * 78)
    agg = {"rv": [], "mpl": []}
    params_table = {}
    for scale in SCALES:
        train = [load_curve(scale, n) for n in TRAIN_CURVES]
        test = [load_curve(scale, n) for n in TEST_CURVES]

        p_rv, f_rv = fit_rv(train, scale)
        # MPL baseline: the repo's official precomputed fit (re-fitting the
        # O(T^2) LD convolution from scratch is ~1e11 ops and pointless -- these
        # are the published, properly-optimised MPL params for these curves).
        p_mpl = MPL_PRECOMPUTED_INIT[scale]
        f_mpl = huber_log_residual(
            np.concatenate([c.loss for c in train]),
            np.concatenate([mpl_predict(p_mpl, c) for c in train]))
        params_table[scale] = (p_rv, p_mpl)

        print(f"\n[{scale}M]  train huber: RV={f_rv:.5f}  MPL={f_mpl:.5f}")
        L0, A, al, B, us, tau, H0 = p_rv
        print(f"  RV params: L0={L0:.3f} A={A:.3f} alpha={al:.3f} B(sigma^2)={B:.4g} "
              f"u*={us:.3f} tau={tau:.0f} H0={H0:.0f}")
        print(f"    -> edge check u*<2: {'OK' if us < 2 else 'FAIL'};  "
              f"H_peak target=u*/eta_peak={us/PEAK_LR:.0f}")
        print(f"  {'curve':22s} {'MAE_rv':>9s} {'MAE_mpl':>9s} {'R2_rv':>8s} {'R2_mpl':>8s}")
        for c in test:
            m_rv = metrics(c.loss, rv_predict(p_rv, c))
            m_mpl = metrics(c.loss, mpl_predict(p_mpl, c))
            agg["rv"].append(m_rv["mae"])
            agg["mpl"].append(m_mpl["mae"])
            print(f"  {c.name:22s} {m_rv['mae']:9.5f} {m_mpl['mae']:9.5f} "
                  f"{m_rv['r2']:8.4f} {m_mpl['r2']:8.4f}")

    rv = np.array(agg["rv"]); mpl = np.array(agg["mpl"])
    print("\n" + "=" * 78)
    print(f"TEST mean MAE  RV={rv.mean():.5f}   MPL={mpl.mean():.5f}   "
          f"({'RV better' if rv.mean() < mpl.mean() else 'MPL better'} "
          f"by {abs(rv.mean()-mpl.mean())/mpl.mean()*100:.1f}%)")
    print(f"RV wins {int((rv < mpl).sum())}/{len(rv)} curves")
    # scale-invariance of the physical params
    print("\nScale-invariance of RV physical params (CV across 25/100/400M):")
    names = ["L0", "A", "alpha", "B", "ustar", "tau", "H0"]
    P = np.array([params_table[s][0] for s in SCALES])
    for j, nm in enumerate(names):
        col = P[:, j]
        print(f"  {nm:6s} {col[0]:10.3g} {col[1]:10.3g} {col[2]:10.3g}   "
              f"CV={col.std()/abs(col.mean())*100:5.1f}%")


if __name__ == "__main__":
    main()
