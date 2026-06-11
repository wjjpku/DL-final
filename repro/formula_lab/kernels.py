#!/usr/bin/env python3
"""Kernel feature generators for non-adiabatic correction variants.

The correction feature is

    phi_K(t) = sum_{k <= t} K(S_t - S_k) * relu(eta_{k-1} - eta_k) / eta_peak,

evaluated at the logged curve steps.  K is a normalized relaxation kernel with
K(0) = 1.  The existing paper law is the one-pole kernel K(u) = exp(-lam*u).

All kernels are parameterized so that `lam` is the initial decay rate
-K'(0)/K(0), which keeps parameters comparable across families:

  exp1(lam)            : exp(-lam u)                      (paper baseline)
  exp2(lam1, lam2, w)  : w exp(-lam1 u) + (1-w) exp(-lam2 u)
  lomax(lam, shape)    : (1 + lam u / shape)^(-shape)     (Gamma mixture of
                         exponentials; shape -> inf recovers exp1)
  stretched(lam, q)    : exp(-(lam u)^q) (q<=1)           (broad mixture)

Weighted-drop variants multiply each decrement by (eta_k / eta_peak)^delta.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from reproduce_cosine_to_wsd import Curve, PEAK_LR  # noqa: E402


def _drops(eta: np.ndarray) -> np.ndarray:
    drop = np.zeros_like(eta)
    drop[1:] = np.maximum(eta[:-1] - eta[1:], 0.0)
    return drop


def conv_feature(curve: Curve, kernel, eta_weight_delta: float = 0.0,
                 peak_lr: float | None = None) -> np.ndarray:
    """Generic O(T_out * T) chunked convolution of drops with kernel K(dS).

    kernel: callable u -> K(u) (vectorized, K(0)=1).
    Matches deep_stime.stime_feature conventions: S = cumsum(eta), the drop at
    step k contributes K(S_t - S_k); output normalized by peak LR and indexed
    at curve.step.
    """
    peak = float(peak_lr if peak_lr is not None else PEAK_LR)
    eta = curve.lrs.astype(np.float64)
    drop = _drops(eta)
    if eta_weight_delta != 0.0:
        w = np.power(np.maximum(eta / peak, 1e-12), eta_weight_delta)
        drop = drop * w
    S = np.cumsum(eta)
    out_steps = np.asarray(curve.step, dtype=np.int64)
    nz = np.nonzero(drop)[0]
    if len(nz) == 0:
        return np.zeros(len(out_steps), dtype=np.float64)
    S_k = S[nz]
    d_k = drop[nz]
    out = np.empty(len(out_steps), dtype=np.float64)
    chunk = max(1, int(4e7 // max(len(nz), 1)))
    for i0 in range(0, len(out_steps), chunk):
        idx = out_steps[i0:i0 + chunk]
        dS = S[idx][:, None] - S_k[None, :]
        mask = dS >= 0.0
        K = np.where(mask, kernel(np.maximum(dS, 0.0)), 0.0)
        out[i0:i0 + chunk] = K @ d_k
    return out / peak


def exp1_feature(curve: Curve, lam: float, **kw) -> np.ndarray:
    """One-pole kernel via exact recurrence (fast path, matches deep_stime)."""
    if kw.get("eta_weight_delta", 0.0) != 0.0:
        return conv_feature(curve, lambda u: np.exp(-lam * u), **kw)
    peak = float(kw.get("peak_lr") or PEAK_LR)
    eta = curve.lrs.astype(np.float64)
    drop = _drops(eta)
    decay = np.exp(-lam * eta)
    s = np.empty_like(eta)
    acc = 0.0
    for t in range(len(eta)):
        acc = acc * decay[t] + drop[t]
        s[t] = acc
    return (s / peak)[curve.step]


def exp2_feature(curve: Curve, lam1: float, lam2: float, w: float, **kw) -> np.ndarray:
    f1 = exp1_feature(curve, lam1, **kw)
    f2 = exp1_feature(curve, lam2, **kw)
    return w * f1 + (1.0 - w) * f2


def lomax_feature(curve: Curve, lam: float, shape: float, **kw) -> np.ndarray:
    """K(u) = (1 + lam*u/shape)^(-shape): Gamma(shape) mixture of exponentials.

    Computed by Gauss-Laguerre quadrature over the Gamma rate mixture so the
    O(T) recurrence stays usable: K(u) = E_a[exp(-a u)], a ~ Gamma(shape, scale
    = lam/shape).  32 nodes give ~1e-10 relative accuracy for shape >= 0.05.
    """
    n_nodes = 32
    x, wq = np.polynomial.laguerre.laggauss(n_nodes)
    # E[exp(-a u)] with a ~ Gamma(k, theta): int exp(-a u) a^(k-1) e^(-a/theta)
    # / (Gamma(k) theta^k) da.  Substitute a = theta * x:
    # = (1/Gamma(k)) int x^(k-1) e^(-x) exp(-theta x u) dx
    k = float(shape)
    theta = lam / k
    from math import gamma as _g
    weights = wq * np.power(x, k - 1.0) / _g(k)
    feats = np.zeros(len(curve.step), dtype=np.float64)
    for xi, wi in zip(x, weights):
        if wi < 1e-14:
            continue
        feats = feats + wi * exp1_feature(curve, theta * xi, **kw)
    return feats


def lomax_feature_exact(curve: Curve, lam: float, shape: float, **kw) -> np.ndarray:
    return conv_feature(curve, lambda u: np.power(1.0 + lam * u / shape, -shape), **kw)


def stretched_feature(curve: Curve, lam: float, q: float, **kw) -> np.ndarray:
    return conv_feature(curve, lambda u: np.exp(-np.power(lam * u, q)), **kw)


def make_feature(spec: dict, curve: Curve) -> np.ndarray:
    """spec: {'family': 'exp1'|'exp2'|'lomax'|'stretched', params...}"""
    fam = spec["family"]
    kw = {}
    if "eta_weight_delta" in spec:
        kw["eta_weight_delta"] = spec["eta_weight_delta"]
    if "peak_lr" in spec:
        kw["peak_lr"] = spec["peak_lr"]
    if fam == "exp1":
        return exp1_feature(curve, spec["lam"], **kw)
    if fam == "exp2":
        return exp2_feature(curve, spec["lam1"], spec["lam2"], spec["w"], **kw)
    if fam == "lomax":
        return lomax_feature(curve, spec["lam"], spec["shape"], **kw)
    if fam == "stretched":
        return stretched_feature(curve, spec["lam"], spec["q"], **kw)
    raise ValueError(fam)
