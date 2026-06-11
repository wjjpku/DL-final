#!/usr/bin/env python3
"""Quick exploration: which relaxation-kernel family best explains the
well-fit-MPL residual, and which transfers probe->sharp best?

Protocols (per scale, public curves, precomputed MPL backbone):
  P-A in-sample : fit kernel params + single kappa on pooled residuals of all
                  6 curves; report R^2 (pooled / sharp-only).
  P-B probe cal : fit kernel params + kappa on the three wsdcon probes only;
                  evaluate MAE change vs MPL on held-out wsd + wsdld.

Kernel families: exp1 (paper baseline, lam free + lam=10), exp2, lomax,
stretched, and eta-weighted exp1.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from reproduce_cosine_to_wsd import (  # noqa: E402
    Curve, load_curve, mpl_predict, metrics, MPL_PRECOMPUTED_INIT, SCALES, PEAK_LR,
)
from formula_lab.kernels import exp1_feature  # noqa: E402

CURVES = [
    "cosine_72000.csv",
    "wsd_20000_24000.csv",
    "wsdld_20000_24000.csv",
    "wsdcon_3.csv",
    "wsdcon_9.csv",
    "wsdcon_18.csv",
]
SHARP = ["wsd_20000_24000.csv", "wsdld_20000_24000.csv"]
PROBES = ["wsdcon_3.csv", "wsdcon_9.csv", "wsdcon_18.csv"]
BIN = 32


class BinnedConv:
    """Precomputed (dS matrix, binned drops) so kernel evals are one matmul."""

    def __init__(self, curve: Curve, eta_weight_delta: float = 0.0):
        eta = curve.lrs.astype(np.float64)
        drop = np.zeros_like(eta)
        drop[1:] = np.maximum(eta[:-1] - eta[1:], 0.0)
        if eta_weight_delta != 0.0:
            drop = drop * np.power(np.maximum(eta / PEAK_LR, 1e-12), eta_weight_delta)
        S = np.cumsum(eta)
        n = len(eta)
        nbins = (n + BIN - 1) // BIN
        bid = np.arange(n) // BIN
        mass = np.bincount(bid, weights=drop, minlength=nbins)
        sw = np.bincount(bid, weights=drop * S, minlength=nbins)
        keep = mass > 0
        self.d = mass[keep]
        self.S_k = sw[keep] / mass[keep]  # drop-mass centroid in S-time
        self.S_out = S[np.asarray(curve.step, dtype=np.int64)]
        self.dS = self.S_out[:, None] - self.S_k[None, :]
        self.valid = self.dS >= 0.0
        self.dS = np.maximum(self.dS, 0.0)

    def feature(self, kernel) -> np.ndarray:
        K = np.where(self.valid, kernel(self.dS), 0.0)
        return (K @ self.d) / PEAK_LR


def kernel_fn(family: str, p: np.ndarray):
    if family == "exp1":
        lam = p[0]
        return lambda u: np.exp(-lam * u)
    if family == "exp2":
        l1, l2, w = p[0], p[1], 1.0 / (1.0 + np.exp(-p[2]))
        return lambda u: w * np.exp(-l1 * u) + (1 - w) * np.exp(-l2 * u)
    if family == "lomax":
        lam, shape = p[0], p[1]
        return lambda u: np.power(1.0 + lam * u / shape, -shape)
    if family == "stretched":
        lam, q = p[0], p[1]
        return lambda u: np.exp(-np.power(np.maximum(lam * u, 1e-300), q))
    raise ValueError(family)


FAMILIES = {
    # family: (param names, init, transform to constrain positive, bounds note)
    "exp1": (["lam"], np.array([10.0])),
    "exp2": (["lam1", "lam2", "logit_w"], np.array([3.0, 30.0, 0.0])),
    "lomax": (["lam", "shape"], np.array([10.0, 1.0])),
    "stretched": (["lam", "q"], np.array([10.0, 0.7])),
}
POSITIVE = {"exp1": [0], "exp2": [0, 1], "lomax": [0, 1], "stretched": [0, 1]}


def fit_kappa_pooled(feats: list[np.ndarray], resids: list[np.ndarray]) -> float:
    x = np.concatenate(feats)
    y = np.concatenate(resids)
    xx = float(np.dot(x, x))
    return 0.0 if xx <= 1e-18 else max(0.0, float(np.dot(x, y) / xx))


def sse(feats, resids, kappa) -> float:
    return float(sum(np.sum((r - kappa * f) ** 2) for f, r in zip(feats, resids)))


def eval_family(scale: str, family: str, fit_curves: list[str],
                eta_weight_delta: float = 0.0, fixed_params: np.ndarray | None = None):
    convs = {c: BinnedConv(load_curve(scale, c), eta_weight_delta) for c in CURVES}
    resid = {}
    base_pred = {}
    for c in CURVES:
        cur = load_curve(scale, c)
        base_pred[c] = mpl_predict(MPL_PRECOMPUTED_INIT[scale], cur)
        resid[c] = cur.loss - base_pred[c]

    names, init = FAMILIES[family]

    def feats_for(p, curves):
        kf = kernel_fn(family, p)
        return [convs[c].feature(kf) for c in curves]

    def objective(logp):
        p = np.array(logp, dtype=float)
        for i in POSITIVE[family]:
            p[i] = np.exp(p[i])
        f = feats_for(p, fit_curves)
        r = [resid[c] for c in fit_curves]
        k = fit_kappa_pooled(f, r)
        return sse(f, r, k)

    if fixed_params is not None:
        p_best = fixed_params
    else:
        x0 = np.array(FAMILIES[family][1], dtype=float)
        for i in POSITIVE[family]:
            x0[i] = np.log(x0[i])
        res = minimize(objective, x0, method="Nelder-Mead",
                       options={"maxiter": 400, "xatol": 1e-3, "fatol": 1e-12})
        p_best = res.x.copy()
        for i in POSITIVE[family]:
            p_best[i] = np.exp(p_best[i])

    f_fit = feats_for(p_best, fit_curves)
    kappa = fit_kappa_pooled(f_fit, [resid[c] for c in fit_curves])

    # metrics
    out = {"scale": scale, "family": family, "params": dict(zip(names, np.atleast_1d(p_best).tolist())),
           "kappa": kappa, "eta_weight_delta": eta_weight_delta, "fit_on": ",".join(fit_curves)}
    all_f = feats_for(p_best, CURVES)
    for label, group in [("pooled", CURVES), ("sharp", SHARP)]:
        f = [all_f[CURVES.index(c)] for c in group]
        r = [resid[c] for c in group]
        y = np.concatenate(r)
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        out[f"r2_{label}"] = 1.0 - sse(f, r, kappa) / ss_tot if ss_tot > 0 else float("nan")
    deltas = []
    for c in SHARP:
        cur = load_curve(scale, c)
        f = all_f[CURVES.index(c)]
        mae0 = metrics(cur.loss, base_pred[c])["mae"]
        mae1 = metrics(cur.loss, base_pred[c] + kappa * f)["mae"]
        out[f"mae_{c.split('_')[0]}"] = mae1
        out[f"mae0_{c.split('_')[0]}"] = mae0
        deltas.append(100.0 * (mae1 / mae0 - 1.0))
    out["sharp_delta_pct"] = float(np.mean(deltas))
    return out


def main():
    # validation: binned conv vs exact recurrence (exp1, lam=10)
    print("== validation: BinnedConv vs exact recurrence (exp1) ==")
    for scale in ["100"]:
        for cname in ["wsd_20000_24000.csv", "cosine_72000.csv", "wsdcon_9.csv"]:
            cur = load_curve(scale, cname)
            for lam in [2.0, 10.0, 30.0]:
                exact = exp1_feature(cur, lam)
                approx = BinnedConv(cur).feature(lambda u: np.exp(-lam * u))
                err = np.max(np.abs(exact - approx)) / max(np.max(exact), 1e-12)
                print(f"  {cname:24s} lam={lam:5.1f} rel.err={err:.2e}")

    rows = []
    print("\n== P-A in-sample pooled fit (kernel params + kappa on all 6 curves) ==")
    hdr = f"{'scale':>5s} {'family':>10s} {'params':>34s} {'kappa':>8s} {'R2pool':>7s} {'R2sharp':>8s} {'dMAE%':>7s}"
    print(hdr)
    for scale in SCALES:
        for family in FAMILIES:
            r = eval_family(scale, family, CURVES)
            rows.append({"protocol": "A", **r})
            pstr = " ".join(f"{k}={v:.3g}" for k, v in r["params"].items())
            print(f"  {scale:>4s} {family:>10s} {pstr:>34s} {r['kappa']:8.4f} "
                  f"{r['r2_pooled']:7.3f} {r['r2_sharp']:8.3f} {r['sharp_delta_pct']:+7.1f}")
        # baseline fixed lam=10 exp1
        r = eval_family(scale, "exp1", CURVES, fixed_params=np.array([10.0]))
        rows.append({"protocol": "A-fixed10", **r})
        print(f"  {scale:>4s} {'exp1@10':>10s} {'lam=10 (fixed)':>34s} {r['kappa']:8.4f} "
              f"{r['r2_pooled']:7.3f} {r['r2_sharp']:8.3f} {r['sharp_delta_pct']:+7.1f}")

    print("\n== P-B probe calibration (fit on wsdcon_3/9/18, test on wsd+wsdld) ==")
    print(hdr)
    for scale in SCALES:
        for family in FAMILIES:
            r = eval_family(scale, family, PROBES)
            rows.append({"protocol": "B", **r})
            pstr = " ".join(f"{k}={v:.3g}" for k, v in r["params"].items())
            print(f"  {scale:>4s} {family:>10s} {pstr:>34s} {r['kappa']:8.4f} "
                  f"{r['r2_pooled']:7.3f} {r['r2_sharp']:8.3f} {r['sharp_delta_pct']:+7.1f}")
        r = eval_family(scale, "exp1", PROBES, fixed_params=np.array([10.0]))
        rows.append({"protocol": "B-fixed10", **r})
        print(f"  {scale:>4s} {'exp1@10':>10s} {'lam=10 (fixed)':>34s} {r['kappa']:8.4f} "
              f"{r['r2_pooled']:7.3f} {r['r2_sharp']:8.3f} {r['sharp_delta_pct']:+7.1f}")

    print("\n== P-C eta-weighted drops, exp1 lam fit (delta scan, in-sample) ==")
    for scale in SCALES:
        for delta in [-0.5, -0.25, 0.25, 0.5]:
            r = eval_family(scale, "exp1", CURVES, eta_weight_delta=delta)
            rows.append({"protocol": "C", **r})
            pstr = " ".join(f"{k}={v:.3g}" for k, v in r["params"].items())
            print(f"  {scale:>4s} {'exp1':>10s} d={delta:+.2f} {pstr:>26s} {r['kappa']:8.4f} "
                  f"{r['r2_pooled']:7.3f} {r['r2_sharp']:8.3f} {r['sharp_delta_pct']:+7.1f}")

    import json
    out = REPO.parent / "results" / "formula_lab" / "explore_kernels.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
