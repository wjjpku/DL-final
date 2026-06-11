#!/usr/bin/env python3
"""Hostile-referee checks for the eta-weighted-drop (delta=0.5) claim.

Checks:
 (a) capacity/selection artifact -> null-weight controls (eta_pre, remtime)
 (b) lam fixed at 10 in both arms (in-sample pooled)
 (c) out-of-sample: kappa from wsdcon probes only (lam=10), eval wsd+wsdld
 (d) per-curve / per-region breakdown
 (e) backbone-misfit compensation -> linear (L0,A,B) refit + FWL poly control
 (f) 25M lam degeneracy -> fixed lam=10 at 25M
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

REPRO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPRO))
from reproduce_cosine_to_wsd import (  # noqa: E402
    load_curve, metrics, MPL_PRECOMPUTED_INIT, SCALES, PEAK_LR,
    compute_s1, compute_ld,
)
from formula_lab.kernels import conv_feature  # noqa: E402

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

# weight arms: (mode, delta)
ARMS = [
    ("none", 0.0),
    ("eta_post", 0.25),
    ("eta_post", 0.5),
    ("eta_post", 0.75),
    ("eta_post", 1.0),
    ("eta_pre", 0.5),
    ("remtime", 0.5),
]


def weight_array(curve, mode: str, delta: float):
    eta = curve.lrs.astype(np.float64)
    n = len(eta)
    if mode == "none":
        return None
    if mode == "eta_post":
        return np.power(np.maximum(eta / PEAK_LR, 1e-12), delta)
    if mode == "eta_pre":
        pre = np.concatenate(([eta[0]], eta[:-1]))
        return np.power(np.maximum(pre / PEAK_LR, 1e-12), delta)
    if mode == "remtime":
        k = np.arange(n, dtype=np.float64)
        return np.power(np.maximum((n - k) / n, 1e-12), delta)
    raise ValueError(mode)


class WConv:
    """Binned drop convolution with an arbitrary per-step weight on drops."""

    def __init__(self, curve, w=None):
        eta = curve.lrs.astype(np.float64)
        drop = np.zeros_like(eta)
        drop[1:] = np.maximum(eta[:-1] - eta[1:], 0.0)
        if w is not None:
            drop = drop * w
        S = np.cumsum(eta)
        n = len(eta)
        nbins = (n + BIN - 1) // BIN
        bid = np.arange(n) // BIN
        mass = np.bincount(bid, weights=drop, minlength=nbins)
        sw = np.bincount(bid, weights=drop * S, minlength=nbins)
        keep = mass > 0
        self.d = mass[keep]
        self.S_k = sw[keep] / mass[keep]
        self.S_out = S[np.asarray(curve.step, dtype=np.int64)]
        self.dS = self.S_out[:, None] - self.S_k[None, :]
        self.valid = self.dS >= 0.0
        self.dS = np.maximum(self.dS, 0.0)

    def feature(self, lam: float) -> np.ndarray:
        K = np.where(self.valid, np.exp(-lam * self.dS), 0.0)
        return (K @ self.d) / PEAK_LR


def fit_kappa(feats, resids):
    x = np.concatenate(feats)
    y = np.concatenate(resids)
    xx = float(np.dot(x, x))
    return 0.0 if xx <= 1e-18 else max(0.0, float(np.dot(x, y) / xx))


def pooled_r2(feats, resids, kappa):
    y = np.concatenate(resids)
    sse = float(sum(np.sum((r - kappa * f) ** 2) for f, r in zip(feats, resids)))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - sse / ss_tot if ss_tot > 0 else float("nan")


def fit_lam(convs, resids, curves, lam0=10.0):
    """Fit lam (log-param Nelder-Mead) + pooled kappa on given curves."""

    def obj(loglam):
        lam = float(np.exp(loglam[0]))
        f = [convs[c].feature(lam) for c in curves]
        r = [resids[c] for c in curves]
        k = fit_kappa(f, r)
        return float(sum(np.sum((rr - k * ff) ** 2) for ff, rr in zip(f, r)))

    res = minimize(obj, np.array([np.log(lam0)]), method="Nelder-Mead",
                   options={"maxiter": 200, "xatol": 1e-3, "fatol": 1e-14})
    return float(np.exp(res.x[0]))


def main():
    rng_report = {}

    # ---- preload everything --------------------------------------------
    data = {}  # scale -> dict
    for scale in SCALES:
        curves = {c: load_curve(scale, c) for c in CURVES}
        p = MPL_PRECOMPUTED_INIT[scale]
        L0, A, alpha, B, C, beta, gamma = p
        s1 = {c: compute_s1(curves[c]) for c in CURVES}
        ld = {c: compute_ld(curves[c], C, beta, gamma) for c in CURVES}
        base = {c: L0 + A * np.power(s1[c], -alpha) + B * ld[c] for c in CURVES}
        resid = {c: curves[c].loss - base[c] for c in CURVES}
        convs = {}
        for mode, delta in ARMS:
            for c in CURVES:
                w = weight_array(curves[c], mode, delta)
                convs[(mode, delta, c)] = WConv(curves[c], w)
        data[scale] = dict(curves=curves, base=base, resid=resid, convs=convs,
                           s1=s1, ld=ld, params=p)

    # ---- validation of binned conv vs exact (weighted case) ------------
    print("== validation: WConv vs exact conv_feature (eta_post 0.5, lam=10) ==")
    for c in ["wsd_20000_24000.csv", "wsdcon_9.csv", "cosine_72000.csv"]:
        cur = data["100"]["curves"][c]
        exact = conv_feature(cur, lambda u: np.exp(-10.0 * u), eta_weight_delta=0.5)
        approx = data["100"]["convs"][("eta_post", 0.5, c)].feature(10.0)
        err = np.max(np.abs(exact - approx)) / max(np.max(exact), 1e-12)
        print(f"  {c:24s} rel.err={err:.2e}")

    def arm_eval(scale, mode, delta, lam, fit_curves, resid_override=None,
                 base_override=None):
        """Fit pooled kappa on fit_curves at given lam; return full breakdown."""
        D = data[scale]
        resid = resid_override if resid_override is not None else D["resid"]
        base = base_override if base_override is not None else D["base"]
        feats = {c: D["convs"][(mode, delta, c)].feature(lam) for c in CURVES}
        kappa = fit_kappa([feats[c] for c in fit_curves],
                          [resid[c] for c in fit_curves])
        out = {"scale": scale, "mode": mode, "delta": delta, "lam": lam,
               "fit_on": "+".join(x.split("_")[0] + x.split("_")[1][:2] if "wsdcon" in x else x.split("_")[0] for x in fit_curves),
               "kappa": kappa}
        out["r2_pooled"] = pooled_r2([feats[c] for c in CURVES],
                                     [resid[c] for c in CURVES], kappa)
        out["r2_sharp"] = pooled_r2([feats[c] for c in SHARP],
                                    [resid[c] for c in SHARP], kappa)
        per = {}
        for c in CURVES:
            cur = D["curves"][c]
            mae0 = metrics(cur.loss, base[c])["mae"]
            mae1 = metrics(cur.loss, base[c] + kappa * feats[c])["mae"]
            r = resid[c]
            sse = float(np.sum((r - kappa * feats[c]) ** 2))
            sst = float(np.sum((r - r.mean()) ** 2))
            per[c] = {"dmae_pct": 100.0 * (mae1 / mae0 - 1.0),
                      "r2": 1.0 - sse / sst if sst > 0 else float("nan"),
                      "mae0": mae0, "mae1": mae1, "n": len(r)}
        out["per_curve"] = per
        out["sharp_dmae_pct"] = float(np.mean([per[c]["dmae_pct"] for c in SHARP]))
        # cooldown half-split for sharp curves
        halves = {}
        for c in SHARP:
            cur = D["curves"][c]
            cool = np.asarray(cur.step) >= 20000
            idx = np.where(cool)[0]
            h1, h2 = idx[: len(idx) // 2], idx[len(idx) // 2:]
            for tag, ii in [("early", h1), ("late", h2)]:
                e0 = float(np.mean(np.abs(resid[c][ii])))
                e1 = float(np.mean(np.abs(resid[c][ii] - kappa * feats[c][ii])))
                halves[f"{c.split('_')[0]}_{tag}"] = 100.0 * (e1 / e0 - 1.0)
        out["cooldown_halves_dmae_pct"] = halves
        return out

    def fmt(r):
        per = r["per_curve"]
        cs = " ".join(f"{c.replace('.csv','').replace('_20000_24000',''):>9s}={per[c]['dmae_pct']:+6.1f}"
                      for c in CURVES)
        return (f"  {r['scale']:>4s} {r['mode']:>8s} d={r['delta']:.2f} lam={r['lam']:6.2f} "
                f"k={r['kappa']:.4f} R2pool={r['r2_pooled']:+.3f} R2sharp={r['r2_sharp']:+.3f} "
                f"sharp dMAE={r['sharp_dmae_pct']:+6.1f}% | per-curve dMAE%: {cs}")

    results = []

    # ---- (b)+(f): in-sample, lam FIXED at 10, all arms ------------------
    print("\n== (b)/(f) in-sample pooled fit, lam=10 FIXED, kappa only ==")
    for scale in SCALES:
        for mode, delta in ARMS:
            r = arm_eval(scale, mode, delta, 10.0, CURVES)
            r["protocol"] = "insample_lam10"
            results.append(r)
            print(fmt(r))

    # ---- (a): in-sample, lam free, key arms ------------------------------
    print("\n== (a) in-sample pooled fit, lam FREE (fit on all 6), key arms ==")
    for scale in SCALES:
        for mode, delta in [("none", 0.0), ("eta_post", 0.5), ("eta_pre", 0.5),
                            ("remtime", 0.5)]:
            convs = {c: data[scale]["convs"][(mode, delta, c)] for c in CURVES}
            lam = fit_lam(convs, data[scale]["resid"], CURVES)
            r = arm_eval(scale, mode, delta, lam, CURVES)
            r["protocol"] = "insample_lamfree"
            results.append(r)
            print(fmt(r))

    # ---- (c): OOS probe calibration, lam=10 fixed ------------------------
    print("\n== (c) OOS: kappa fit on wsdcon probes only, lam=10, eval wsd+wsdld ==")
    for scale in SCALES:
        for mode, delta in ARMS:
            r = arm_eval(scale, mode, delta, 10.0, PROBES)
            r["protocol"] = "oos_probe_lam10"
            results.append(r)
            print(fmt(r))

    # ---- (c'): OOS probe calibration, lam fit on probes ------------------
    print("\n== (c') OOS: lam AND kappa fit on probes only, eval wsd+wsdld ==")
    for scale in SCALES:
        for mode, delta in [("none", 0.0), ("eta_post", 0.5)]:
            convs = {c: data[scale]["convs"][(mode, delta, c)] for c in CURVES}
            lam = fit_lam(convs, data[scale]["resid"], PROBES)
            r = arm_eval(scale, mode, delta, lam, PROBES)
            r["protocol"] = "oos_probe_lamfree"
            results.append(r)
            print(fmt(r))

    # ---- (d)/(P6): per-curve kappas, lam=10, arms none vs eta_post 0.5 ---
    print("\n== (d) per-curve origin-LS kappa (lam=10): consistency across curves ==")
    for scale in SCALES:
        for mode, delta in [("none", 0.0), ("eta_post", 0.5)]:
            D = data[scale]
            ks = {}
            for c in CURVES:
                f = D["convs"][(mode, delta, c)].feature(10.0)
                ks[c] = fit_kappa([f], [D["resid"][c]])
            kp = fit_kappa([D["convs"][(mode, delta, c)].feature(10.0) for c in PROBES],
                           [D["resid"][c] for c in PROBES])
            ksh = fit_kappa([D["convs"][(mode, delta, c)].feature(10.0) for c in SHARP],
                            [D["resid"][c] for c in SHARP])
            vals = np.array([ks[c] for c in CURVES])
            cv = float(np.std(vals) / np.mean(vals)) if np.mean(vals) > 0 else float("nan")
            print(f"  {scale:>4s} {mode:>8s} d={delta:.2f} per-curve k: "
                  + " ".join(f"{ks[c]:.4f}" for c in CURVES)
                  + f" | CV={cv:.2f} | k_probes={kp:.4f} k_sharp={ksh:.4f} "
                  f"ratio={kp / ksh if ksh > 0 else float('nan'):.3f}")
            results.append({"protocol": "percurve_kappa", "scale": scale,
                            "mode": mode, "delta": delta,
                            "kappas": {c: ks[c] for c in CURVES},
                            "k_probes": kp, "k_sharp": ksh, "cv": cv})

    # ---- (e1): refit backbone (L0, A, B) linearly on all 6 curves --------
    print("\n== (e1) backbone refit control: (L0,A,B) linear LS on all 6 curves ==")
    for scale in SCALES:
        D = data[scale]
        X, Y = [], []
        for c in CURVES:
            X.append(np.stack([np.ones_like(D["s1"][c]),
                               np.power(D["s1"][c], -MPL_PRECOMPUTED_INIT[scale][2]),
                               D["ld"][c]], axis=1))
            Y.append(D["curves"][c].loss)
        X = np.concatenate(X)
        Y = np.concatenate(Y)
        coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
        L0n, An, Bn = (float(v) for v in coef)
        alpha = MPL_PRECOMPUTED_INIT[scale][2]
        base2 = {c: L0n + An * np.power(D["s1"][c], -alpha) + Bn * D["ld"][c]
                 for c in CURVES}
        resid2 = {c: D["curves"][c].loss - base2[c] for c in CURVES}
        print(f"  {scale:>4s} refit L0={L0n:.4f} A={An:.4f} B={Bn:.1f} "
              f"(frozen: L0={MPL_PRECOMPUTED_INIT[scale][0]:.4f} "
              f"A={MPL_PRECOMPUTED_INIT[scale][1]:.4f} B={MPL_PRECOMPUTED_INIT[scale][3]:.1f})")
        for mode, delta in [("none", 0.0), ("eta_post", 0.5)]:
            r = arm_eval(scale, mode, delta, 10.0, CURVES,
                         resid_override=resid2, base_override=base2)
            r["protocol"] = "insample_lam10_refitLAB"
            results.append(r)
            print(fmt(r))
        for mode, delta in [("none", 0.0), ("eta_post", 0.5)]:
            r = arm_eval(scale, mode, delta, 10.0, PROBES,
                         resid_override=resid2, base_override=base2)
            r["protocol"] = "oos_probe_lam10_refitLAB"
            results.append(r)
            print("  OOS" + fmt(r)[5:])

    # ---- (e2): FWL control — residualize vs per-curve degree-2 poly ------
    print("\n== (e2) FWL control: per-curve degree-2 poly removed from resid+feat (lam=10) ==")
    for scale in SCALES:
        D = data[scale]
        for mode, delta in [("none", 0.0), ("eta_post", 0.5)]:
            fo, ro, fo_sharp, ro_sharp = [], [], [], []
            for c in CURVES:
                cur = D["curves"][c]
                t = np.asarray(cur.step, dtype=np.float64)
                t = (t - t.min()) / max(t.max() - t.min(), 1.0)
                Pb = np.stack([np.ones_like(t), t, t * t], axis=1)
                Q, _ = np.linalg.qr(Pb)
                proj = lambda v: v - Q @ (Q.T @ v)
                f = proj(D["convs"][(mode, delta, c)].feature(10.0))
                r_ = proj(D["resid"][c])
                fo.append(f)
                ro.append(r_)
                if c in SHARP:
                    fo_sharp.append(f)
                    ro_sharp.append(r_)
            k = fit_kappa(fo, ro)
            r2p = pooled_r2(fo, ro, k)
            r2s = pooled_r2(fo_sharp, ro_sharp, k)
            print(f"  {scale:>4s} {mode:>8s} d={delta:.2f} k={k:.4f} "
                  f"R2pool_orth={r2p:+.3f} R2sharp_orth={r2s:+.3f}")
            results.append({"protocol": "fwl_deg2", "scale": scale, "mode": mode,
                            "delta": delta, "kappa": k, "r2_pooled_orth": r2p,
                            "r2_sharp_orth": r2s})

    out = REPRO.parent / "results" / "formula_lab" / "referee_eta_weight.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        return o

    out.write_text(json.dumps(clean(results), indent=1), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
