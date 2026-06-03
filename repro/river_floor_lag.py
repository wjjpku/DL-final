#!/usr/bin/env python3
"""River-floor sharpness-lag correction on top of FROZEN MPL.

Mechanism (docs/core/river_valley_derivation.md, §"new mechanism"): the true
sharpness lambda_t is a state variable that lags the schedule with timescale
tau_s. After a *sharp* LR drop the system is still oscillating as if at the old
high LR (its sharpness has not relaxed yet), so oscillation energy that has not
been dissipated keeps the loss elevated -- a positive decay-phase residual.
Cosine decays gradually, so lambda tracks the schedule and the residual ~ 0.

Proxy: tilde_eta = EMA_{tau_s}(eta) (lagged LR). Correction term, added to the
*frozen published* MPL prediction, with only TWO new parameters:

      Delta L(t) = kappa * (tilde_eta_t - eta_t)_+ .

Design choices that make this a constraint, not added flexibility:
  - Every MPL parameter is FROZEN at MPL_PRECOMPUTED_INIT (the SOTA baseline).
  - The term is ~dormant on smooth cosine (tilde_eta ~ eta), so it cannot fit
    cosine noise. To identify (kappa, tau_s) honestly we therefore train on
    cosine x2 + ONE sharp-decay curve (wsdcon_3) and TEST on the held-out
    wsd / wsdld / wsdcon_9 / wsdcon_18. (User pre-approved non cos->wsd splits.)

Kill criteria (decided before fitting):
  kappa > 0, tau_s > 0, both scale-invariant across 25/100/400M, AND held-out
  test MAE strictly below frozen MPL. Otherwise: honest negative.
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
    Curve, load_curve, huber_log_residual, metrics,
    mpl_predict, MPL_PRECOMPUTED_INIT, SCALES, WARMUP, PEAK_LR,
)

TRAIN = ["cosine_24000.csv", "cosine_72000.csv", "wsdcon_3.csv"]
TEST = ["wsd_20000_24000.csv", "wsdld_20000_24000.csv",
        "wsdcon_9.csv", "wsdcon_18.csv"]


def lagged_eta(lrs: np.ndarray, tau_s: float) -> np.ndarray:
    """EMA of the LR: tilde_eta[t] = a*tilde_eta[t-1] + (1-a)*eta[t], a=1-1/tau_s.
    Initialised at the peak LR (post-warmup plateau)."""
    a = 1.0 - 1.0 / tau_s
    z = a * PEAK_LR
    y, _ = lfilter([1.0 - a], [1.0, -a], lrs, zi=[z])
    return y


def correction(curve: Curve, kappa: float, tau_s: float) -> np.ndarray:
    te = lagged_eta(curve.lrs, tau_s)
    drop = np.clip(te - curve.lrs, 0.0, None)      # (tilde_eta - eta)_+
    return kappa * drop[curve.step]


def predict(p_mpl, curve, kappa, tau_s):
    return mpl_predict(p_mpl, curve) + correction(curve, kappa, tau_s)


def fit_correction(p_mpl, train_curves):
    """Fit only (kappa, tau_s); MPL frozen.

    The frozen-MPL prediction is identical every eval, so precompute it once per
    curve (the O(T*n) LD convolution was the whole cost). The objective then only
    recomputes the cheap correction term."""
    base = [mpl_predict(p_mpl, c) for c in train_curves]
    loss = [c.loss for c in train_curves]
    la = np.concatenate(loss)

    def obj(x):
        kappa, tau_s = x
        if tau_s <= 1.0:
            return 1e18
        pa = []
        for c, b in zip(train_curves, base):
            pr = b + correction(c, kappa, tau_s)
            if np.any(~np.isfinite(pr)) or np.any(pr <= 0):
                return 1e18
            pa.append(pr)
        return huber_log_residual(la, np.concatenate(pa))

    best_x, best_f = None, float("inf")
    for k0 in (0.0, 50.0, 200.0, 1000.0):
        for t0 in (20.0, 100.0, 500.0, 2000.0):
            res = minimize(obj, x0=np.array([k0, t0]), method="Nelder-Mead",
                           options={"maxiter": 2000, "xatol": 1e-6, "fatol": 1e-12})
            if res.fun < best_f:
                best_f, best_x = float(res.fun), res.x
    return best_x, best_f


def main():
    print("=" * 78)
    print("River-floor sharpness-lag correction on FROZEN MPL")
    print(" train = cosine x2 + wsdcon_3 ;  test = wsd / wsdld / wsdcon_9 / wsdcon_18")
    print("=" * 78)
    params = {}
    agg_corr, agg_mpl = [], []
    for scale in SCALES:
        p_mpl = MPL_PRECOMPUTED_INIT[scale]
        train = [load_curve(scale, n) for n in TRAIN]
        test = [load_curve(scale, n) for n in TEST]
        (kappa, tau_s), ftr = fit_correction(p_mpl, train)
        params[scale] = (kappa, tau_s)
        print(f"\n[{scale}M] fitted kappa={kappa:.4g}  tau_s={tau_s:.1f}  (train huber={ftr:.5f})")
        print(f"  {'curve':22s} {'MAE_corr':>10s} {'MAE_mpl':>10s} {'dMAE%':>7s}")
        for c in test:
            m_c = metrics(c.loss, predict(p_mpl, c, kappa, tau_s))
            m_m = metrics(c.loss, mpl_predict(p_mpl, c))
            agg_corr.append(m_c["mae"]); agg_mpl.append(m_m["mae"])
            d = (m_c["mae"] - m_m["mae"]) / m_m["mae"] * 100
            print(f"  {c.name:22s} {m_c['mae']:10.5f} {m_m['mae']:10.5f} {d:7.1f}")

    corr = np.array(agg_corr); mpl = np.array(agg_mpl)
    print("\n" + "=" * 78)
    print(f"HELD-OUT mean MAE  corrected={corr.mean():.5f}   frozen-MPL={mpl.mean():.5f}")
    if corr.mean() < mpl.mean():
        print(f"  -> correction BETTER by {(1-corr.mean()/mpl.mean())*100:.1f}% ; "
              f"wins {int((corr<mpl).sum())}/{len(corr)}")
    else:
        print(f"  -> correction WORSE by {(corr.mean()/mpl.mean()-1)*100:.1f}% ; "
              f"wins {int((corr<mpl).sum())}/{len(corr)}")
    print("\nScale-invariance / sign check of the 2 new params:")
    ks = np.array([params[s][0] for s in SCALES])
    ts = np.array([params[s][1] for s in SCALES])
    print(f"  kappa  {ks[0]:.3g} {ks[1]:.3g} {ks[2]:.3g}  CV={ks.std()/abs(ks.mean())*100:.0f}%  "
          f"{'(>0 OK)' if (ks>0).all() else '(FAIL: some <=0)'}")
    print(f"  tau_s  {ts[0]:.3g} {ts[1]:.3g} {ts[2]:.3g}  CV={ts.std()/abs(ts.mean())*100:.0f}%")


if __name__ == "__main__":
    main()
