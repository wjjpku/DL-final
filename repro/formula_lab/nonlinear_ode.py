#!/usr/bin/env python3
"""Nonlinear-relaxation ODE law: a structural replacement candidate.

State equation (S-time), one new parameter r* :

    r_{t+1} = r_t * exp(-lam0 * (1 + r_t/rstar) * eta_t) + chi_w * d_t

with d_t the (optionally eta-weighted) normalized LR drop.  Linear limit
rstar -> inf recovers the paper's one-pole law exactly.

Mechanism: large excursions relax faster.  One parameter should jointly
explain (i) probe-vs-sharp fitted-lambda mismatch (15-19 vs 0.5-5),
(ii) concentration-dependent visible amplitude (probe c ~ 0.1-0.26 vs
sharp 0.43-0.60), (iii) the probe amplitude deficit pattern.

Decisive tests (public curves, frozen MPL backbone):
  T-A joint family fit: ONE (lam0, rstar, kappa) on probes+sharp pooled,
      per scale -- compare pooled SSE/R2 vs best linear one-pole with
      (lam, kappa) likewise pooled (same DOF count: 3 vs 2; also compare
      linear with per-family kappa = 4 DOF oracle).
  T-B probes-only -> sharp transfer (the deployment test): fit (lam0,
      rstar, kappa) on the 3 wsdcon probes only, evaluate wsd+wsdld MAE.
      Baselines: linear lr@10 -17.1%, pow d=0.5@10 -28.6%, best grid -34.5%.
  T-C cross-scale stability of (lam0, rstar).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from reproduce_cosine_to_wsd import (  # noqa: E402
    load_curve, mpl_predict, metrics, MPL_PRECOMPUTED_INIT, SCALES, PEAK_LR,
)
from formula_lab.lab import DECAY, PROBES  # noqa: E402

CURVES6 = ["cosine_72000.csv"] + DECAY + PROBES


BIN = 8
_PREP_CACHE: dict = {}


def _prep(curve, delta: float):
    """Bin the dense schedule: per segment eta_sum and (weighted) drop mass."""
    key = (curve.scale, curve.name, delta)
    if key in _PREP_CACHE:
        return _PREP_CACHE[key]
    eta = curve.lrs.astype(np.float64)
    drop = np.zeros_like(eta)
    drop[1:] = np.maximum(eta[:-1] - eta[1:], 0.0)
    if delta != 0.0:
        drop = drop * np.power(np.maximum(eta / PEAK_LR, 1e-12), delta)
    drop = drop / PEAK_LR
    n = len(eta)
    nbins = (n + BIN - 1) // BIN
    pad = nbins * BIN - n
    eta_p = np.pad(eta, (0, pad))
    drop_p = np.pad(drop, (0, pad))
    eta_seg = eta_p.reshape(nbins, BIN).sum(axis=1)
    drop_seg = drop_p.reshape(nbins, BIN).sum(axis=1)
    seg_of_step = np.asarray(curve.step, dtype=np.int64) // BIN
    out = (eta_seg, drop_seg, seg_of_step)
    _PREP_CACHE[key] = out
    return out


def ode_response(curve, lam0: float, rstar: float, kappa: float,
                 delta: float = 0.0) -> np.ndarray:
    """Integrate the nonlinear state on the binned grid (decay-then-deposit
    per segment; exact in the linear limit when lam0*eta_seg << 1)."""
    eta_seg, drop_seg, seg_of_step = _prep(curve, delta)
    r = 0.0
    out = np.empty(len(eta_seg), dtype=np.float64)
    inv = 1.0 / max(rstar, 1e-12)
    for i in range(len(eta_seg)):
        r = r * np.exp(-lam0 * (1.0 + r * inv) * eta_seg[i]) + kappa * drop_seg[i]
        out[i] = r
    return out[seg_of_step]


def ode_response_exact(curve, lam0, rstar, kappa, delta=0.0):
    """Step-exact reference for validating the binned integrator."""
    eta = curve.lrs.astype(np.float64)
    drop = np.zeros_like(eta)
    drop[1:] = np.maximum(eta[:-1] - eta[1:], 0.0)
    if delta != 0.0:
        drop = drop * np.power(np.maximum(eta / PEAK_LR, 1e-12), delta)
    drop = drop / PEAK_LR
    r = 0.0
    out = np.empty_like(eta)
    inv = 1.0 / max(rstar, 1e-12)
    for t in range(len(eta)):
        r = r * np.exp(-lam0 * (1.0 + r * inv) * eta[t]) + kappa * drop[t]
        out[t] = r
    return out[np.asarray(curve.step, dtype=np.int64)]


def fit_ode(scale: str, fit_curves: list[str], delta: float = 0.0,
            linear: bool = False):
    """NLS fit of (lam0, rstar, kappa) [or (lam, kappa) linear] on pooled
    residuals of fit_curves."""
    p = MPL_PRECOMPUTED_INIT[scale]
    data = []
    for n in fit_curves:
        c = load_curve(scale, n)
        data.append((c, c.loss - mpl_predict(p, c)))

    def sse(theta):
        if linear:
            lam0, kap = np.exp(theta)
            rs = 1e18
        else:
            lam0, rs, kap = np.exp(theta)
        tot = 0.0
        for c, resid in data:
            pred = ode_response(c, lam0, rs, kap, delta)
            tot += float(np.sum((resid - pred) ** 2))
        return tot

    best = None
    inits = ([[np.log(10.0), np.log(0.05)]] if linear else
             [[np.log(10.0), np.log(r0), np.log(0.05)]
              for r0 in [0.003, 0.01, 0.03, 0.1]])
    for x0 in inits:
        res = minimize(sse, x0, method="Nelder-Mead",
                       options={"maxiter": 600, "xatol": 1e-4, "fatol": 1e-14})
        if best is None or res.fun < best.fun:
            best = res
    th = np.exp(best.x)
    return th, best.fun


def pooled_r2(scale, curves, params, delta=0.0, linear=False):
    p = MPL_PRECOMPUTED_INIT[scale]
    ys, ps = [], []
    for n in curves:
        c = load_curve(scale, n)
        ys.append(c.loss - mpl_predict(p, c))
        if linear:
            lam0, kap = params
            ps.append(ode_response(c, lam0, 1e18, kap, delta))
        else:
            lam0, rs, kap = params
            ps.append(ode_response(c, lam0, rs, kap, delta))
    y = np.concatenate(ys); f = np.concatenate(ps)
    ss = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - float(np.sum((y - f) ** 2)) / ss


def eval_sharp(scale, params, delta=0.0, linear=False):
    p = MPL_PRECOMPUTED_INIT[scale]
    out = []
    for n in DECAY:
        c = load_curve(scale, n)
        base = mpl_predict(p, c)
        if linear:
            lam0, kap = params
            pred = base + ode_response(c, lam0, 1e18, kap, delta)
        else:
            lam0, rs, kap = params
            pred = base + ode_response(c, lam0, rs, kap, delta)
        m0 = metrics(c.loss, base)["mae"]
        m1 = metrics(c.loss, pred)["mae"]
        out.append((m0, m1))
    return out


def main():
    # integrator validation
    print("== binned-vs-exact integrator check ==")
    for n in ["wsd_20000_24000.csv", "wsdcon_9.csv"]:
        c = load_curve("100", n)
        for th in [(10.0, 1e18, 0.05), (15.0, 0.01, 0.08)]:
            a = ode_response(c, *th)
            b = ode_response_exact(c, *th)
            print(f"  {n:22s} {th}: rel.err="
                  f"{np.max(np.abs(a-b))/max(np.max(np.abs(b)),1e-12):.2e}")

    print("=" * 86)
    print("T-A joint family fit (probes+sharp pooled, per scale)")
    print("=" * 86)
    print(f"{'scale':>5} {'model':>14} {'lam0':>7} {'rstar':>9} {'kappa':>8} "
          f"{'R2(joint)':>9} {'R2 sharp':>9} {'R2 probe':>9}")
    fam = DECAY + PROBES
    for scale in SCALES:
        for tag, lin in [("linear 1-pole", True), ("nonlinear ODE", False)]:
            th, _ = fit_ode(scale, fam, linear=lin)
            r2j = pooled_r2(scale, fam, th, linear=lin)
            r2s = pooled_r2(scale, DECAY, th, linear=lin)
            r2p = pooled_r2(scale, PROBES, th, linear=lin)
            if lin:
                print(f"{scale:>4}M {tag:>14} {th[0]:7.2f} {'inf':>9} {th[1]:8.4f} "
                      f"{r2j:9.3f} {r2s:9.3f} {r2p:9.3f}")
            else:
                print(f"{scale:>4}M {tag:>14} {th[0]:7.2f} {th[1]:9.4f} {th[2]:8.4f} "
                      f"{r2j:9.3f} {r2s:9.3f} {r2p:9.3f}")

    print()
    print("=" * 86)
    print("T-B probes-only calibration -> held-out sharp (deployment test)")
    print("  baselines: linear lr@10 -17.1% | pow d=0.5@10 -28.6% | grid best -34.5%")
    print("=" * 86)
    for tag, lin, delta in [("linear (lam,kap)", True, 0.0),
                            ("nonlinear ODE", False, 0.0),
                            ("nonlinear + d=0.5", False, 0.5)]:
        m0s, m1s, wins = [], [], 0
        ths = {}
        for scale in SCALES:
            th, _ = fit_ode(scale, PROBES, delta=delta, linear=lin)
            ths[scale] = th
            for m0, m1 in eval_sharp(scale, th, delta=delta, linear=lin):
                m0s.append(m0); m1s.append(m1); wins += int(m1 < m0)
        d = 100.0 * (np.mean(m1s) / np.mean(m0s) - 1.0)
        pstr = " | ".join(
            (f"{s}: lam0={ths[s][0]:.1f},k={ths[s][1]:.3f}" if lin else
             f"{s}: lam0={ths[s][0]:.1f},r*={ths[s][1]:.4f},k={ths[s][2]:.3f}")
            for s in SCALES)
        print(f"{tag:20s} {d:+6.1f}% {wins}/6   {pstr}")


if __name__ == "__main__":
    main()
