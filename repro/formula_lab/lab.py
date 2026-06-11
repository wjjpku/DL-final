#!/usr/bin/env python3
"""Formula lab: parameterized non-adiabatic correction features + protocols.

Feature spec (dict):
  form   : 'lr'     -> drops d_k = (eta_{k-1}-eta_k)_+ / eta_peak        (paper)
           'floor'  -> drops d_k = [(eta_{k-1}/peak)^p - (eta_k/peak)^p]_+
           'pow'    -> d_k = (eta_{k-1}-eta_k)_+ * (eta_k/peak)^delta / peak
           'affine' -> d_k = (eta_{k-1}-eta_k)_+ * ((1-rho)+rho*eta_k/peak) / peak
  p / delta / rho : the corresponding exponent / mix
  lam    : S-time one-pole rate (default 10)
  eta_at : 'post' (default) or 'pre' or 'mid' -- which eta the weight uses
           ('floor' form has no placement ambiguity)

All features are computed by the exact O(T) recurrence (one-pole kernel),
normalized so they are dimensionless, and indexed at curve.step.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from reproduce_cosine_to_wsd import (  # noqa: E402
    Curve, load_curve, mpl_predict, metrics, compute_s1,
    MPL_PRECOMPUTED_INIT, PEAK_LR, SCALES,
)

DECAY = ["wsd_20000_24000.csv", "wsdld_20000_24000.csv"]
PROBES = ["wsdcon_3.csv", "wsdcon_9.csv", "wsdcon_18.csv"]


# ----------------------------- features -----------------------------
def weighted_drops(eta: np.ndarray, spec: dict, peak: float) -> np.ndarray:
    form = spec.get("form", "lr")
    raw = np.zeros_like(eta)
    raw[1:] = np.maximum(eta[:-1] - eta[1:], 0.0)
    if form == "floor":
        p = float(spec["p"])
        x = np.maximum(eta / peak, 0.0) ** p
        d = np.zeros_like(eta)
        d[1:] = np.maximum(x[:-1] - x[1:], 0.0)
        return d
    if form == "lr":
        return raw / peak
    at = spec.get("eta_at", "post")
    eref = np.empty_like(eta)
    if at == "post":
        eref[:] = eta
    elif at == "pre":
        eref[0] = eta[0]
        eref[1:] = eta[:-1]
    else:  # mid
        eref[0] = eta[0]
        eref[1:] = 0.5 * (eta[:-1] + eta[1:])
    x = np.maximum(eref / peak, 1e-12)
    if form == "pow":
        return raw * np.power(x, float(spec["delta"])) / peak
    if form == "affine":
        rho = float(spec["rho"])
        return raw * ((1.0 - rho) + rho * x) / peak
    raise ValueError(form)


def feature(curve: Curve, spec: dict, peak_lr: float | None = None) -> np.ndarray:
    peak = float(peak_lr if peak_lr is not None else PEAK_LR)
    lam = float(spec.get("lam", 10.0))
    eta = curve.lrs.astype(np.float64)
    d = weighted_drops(eta, spec, peak)
    decay = np.exp(-lam * eta)
    s = np.empty_like(eta)
    acc = 0.0
    for t in range(len(eta)):
        acc = acc * decay[t] + d[t]
        s[t] = acc
    return s[np.asarray(curve.step, dtype=np.int64)]


def fit_origin(x: np.ndarray, y: np.ndarray):
    kap = float(np.dot(x, y) / max(np.dot(x, x), 1e-18))
    resid = y - kap * x
    ss = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid ** 2)) / ss if ss > 0 else float("nan")
    return kap, r2


# ----------------------------- floor probes -----------------------------
def probe_floors(scale: str):
    """Equilibrium floors from wsdcon stage-2 finals minus A*S^-alpha backbone."""
    p = MPL_PRECOMPUTED_INIT[scale]
    L0, A, alpha = p[0], p[1], p[2]
    etas, floors = [], []
    for n, mul in [("wsdcon_3.csv", 3), ("wsdcon_9.csv", 9), ("wsdcon_18.csv", 18)]:
        c = load_curve(scale, n)
        s1 = compute_s1(c)
        backbone = L0 + A * np.power(s1, -alpha)
        floors.append(float(np.mean((c.loss - backbone)[-5:])))
        etas.append(mul * 1e-5)
    return np.array(etas), np.array(floors)


def probe_floor_powerlaw(scale: str) -> tuple[float, float]:
    """Fit floor = a*(eta/peak)^p on the 3 wsdcon probes; return (a, p)."""
    etas, floors = probe_floors(scale)
    ok = floors > 0
    x = np.log(etas[ok] / PEAK_LR)
    y = np.log(floors[ok])
    p, loga = np.polyfit(x, y, 1)
    return float(np.exp(loga)), float(p)


def probe_linear_slope(scale: str) -> float:
    etas, floors = probe_floors(scale)
    return float(np.polyfit(etas, floors, 1)[0])


# ----------------------------- amplitude chains -----------------------------
def kappa_fit_sharp(scale: str, spec: dict) -> float:
    """In-sample origin-LS kappa on this scale's wsd+wsdld residuals."""
    p = MPL_PRECOMPUTED_INIT[scale]
    xs, ys = [], []
    for n in DECAY:
        c = load_curve(scale, n)
        ys.append(c.loss - mpl_predict(p, c))
        xs.append(feature(c, spec))
    return fit_origin(np.concatenate(xs), np.concatenate(ys))[0]


def kappa_pred(scale: str, spec: dict, chain: str) -> float:
    """Per-scale predicted amplitude (before the universal c factor).

    chain 'probe-linear'  : eta_peak * dL_eq/deta (paper; only for form='lr')
    chain 'probe-power'   : a from floor power-law fit -- for form='floor' the
                            feature already carries (eta/peak)^p so kappa = a;
                            for 'pow' features kappa = a*p (chi at eta_peak).
    chain 'mpl-B'         : eta_peak * B from the frozen MPL params.
    """
    if chain == "probe-linear":
        return probe_linear_slope(scale) * PEAK_LR
    if chain == "probe-power":
        a, p = probe_floor_powerlaw(scale)
        form = spec.get("form", "lr")
        if form == "floor":
            return a
        if form == "pow":
            return a * p
        return a  # lr/affine: best-effort
    if chain == "mpl-B":
        return float(MPL_PRECOMPUTED_INIT[scale][3]) * PEAK_LR
    raise ValueError(chain)


# ----------------------------- protocols -----------------------------
def table1_protocol(spec: dict, chain: str, use_probe_p: bool = False,
                    verbose: bool = False) -> dict:
    """deep_predict.py chain, generalized.

    Per scale: ratio c_s = kappa_fit_sharp / kappa_pred; target uses LOO mean
    of other scales' c, kappa = c_loo * kappa_pred(target).  If use_probe_p,
    the floor exponent p/delta is set per-scale from the target's own probes
    (no sharp-curve information).
    """
    specs = {}
    for s in SCALES:
        sp = dict(spec)
        if use_probe_p:
            _, p = probe_floor_powerlaw(s)
            if sp.get("form") == "floor":
                sp["p"] = p
            elif sp.get("form") == "pow":
                sp["delta"] = max(p - 1.0, 0.0)
        specs[s] = sp

    ratio = {s: kappa_fit_sharp(s, specs[s]) / kappa_pred(s, specs[s], chain)
             for s in SCALES}
    rows = []
    for tgt in SCALES:
        c_loo = float(np.mean([ratio[s] for s in SCALES if s != tgt]))
        kappa = c_loo * kappa_pred(tgt, specs[tgt], chain)
        p = MPL_PRECOMPUTED_INIT[tgt]
        for n in DECAY:
            cu = load_curve(tgt, n)
            base = mpl_predict(p, cu)
            m0 = metrics(cu.loss, base)["mae"]
            m1 = metrics(cu.loss, base + kappa * feature(cu, specs[tgt]))["mae"]
            rows.append({"scale": tgt, "curve": n, "mae_mpl": m0, "mae_corr": m1,
                         "delta_pct": 100.0 * (m1 / m0 - 1.0), "kappa": kappa,
                         "c_loo": c_loo})
            if verbose:
                print(f"  {tgt:>4}M {n:18s} {m0:.5f} -> {m1:.5f} "
                      f"({rows[-1]['delta_pct']:+.1f}%) kappa={kappa:.4f}")
    m0 = float(np.mean([r["mae_mpl"] for r in rows]))
    m1 = float(np.mean([r["mae_corr"] for r in rows]))
    return {"rows": rows, "mae_mpl": m0, "mae_corr": m1,
            "delta_pct": 100.0 * (m1 / m0 - 1.0),
            "wins": int(sum(r["mae_corr"] < r["mae_mpl"] for r in rows)),
            "ratios": ratio}


def leave_one_sharp_protocol(spec: dict) -> dict:
    """current_law_more_data_calibration 'other_sharp_only' protocol:
    calibrate kappa on the OTHER sharp curve (same scale), test held-out sharp."""
    rows = []
    for scale in SCALES:
        p = MPL_PRECOMPUTED_INIT[scale]
        for held in DECAY:
            cal = [n for n in DECAY if n != held][0]
            cc = load_curve(scale, cal)
            kappa = max(0.0, fit_origin(feature(cc, spec),
                                        cc.loss - mpl_predict(p, cc))[0])
            cu = load_curve(scale, held)
            base = mpl_predict(p, cu)
            m0 = metrics(cu.loss, base)["mae"]
            m1 = metrics(cu.loss, base + kappa * feature(cu, spec))["mae"]
            rows.append({"scale": scale, "curve": held, "mae_mpl": m0,
                         "mae_corr": m1, "kappa": kappa,
                         "delta_pct": 100.0 * (m1 / m0 - 1.0)})
    m0 = float(np.mean([r["mae_mpl"] for r in rows]))
    m1 = float(np.mean([r["mae_corr"] for r in rows]))
    return {"rows": rows, "mae_mpl": m0, "mae_corr": m1,
            "delta_pct": 100.0 * (m1 / m0 - 1.0),
            "wins": int(sum(r["mae_corr"] < r["mae_mpl"] for r in rows))}


def insample_r2(spec: dict) -> dict:
    """Pooled origin-LS R^2 on sharp residuals per scale (paper Fig.2 metric)."""
    out = {}
    for scale in SCALES:
        p = MPL_PRECOMPUTED_INIT[scale]
        xs, ys = [], []
        for n in DECAY:
            c = load_curve(scale, n)
            ys.append(c.loss - mpl_predict(p, c))
            xs.append(feature(c, spec))
        _, r2 = fit_origin(np.concatenate(xs), np.concatenate(ys))
        out[scale] = r2
    return out
