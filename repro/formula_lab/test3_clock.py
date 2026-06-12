#!/usr/bin/env python3
"""Adjudication test 3: merged-clock kernel.

Per-step kernel decay rate  lam_S * eta_u + 1/tau_0  (affine clock) vs the
shipped pure-S clock (lam_slow * eta_u) and a power clock.

(i)  AIC on existing tau measurements: public wsdcon transients (re-measured
     here per scale) + the 10.7M equal-S ladder taus.
(ii) Falsifier: per-family lam_S refit with tau_0 fixed -- does the
     probes(15-19) vs sharp(0.5-5) gap collapse?
(iii) Retrodiction: does the affine clock with measured constants reproduce
     the 10.7M deployment (lam*~1 grid hack) on probes->sharp transfer?
(iv) Non-regression on public probes-only (matched) and LOS.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit, minimize_scalar, minimize

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from reproduce_cosine_to_wsd import (  # noqa: E402
    load_curve, mpl_predict, metrics, MPL_PRECOMPUTED_INIT, SCALES, PEAK_LR,
)
from formula_lab.lab import fit_origin, DECAY, PROBES  # noqa: E402
from formula_lab.matched_probe import cal_probes_for  # noqa: E402

LADDER = [(0.5e-4, 171.0), (1.0e-4, 150.0), (2.0e-4, 131.0),
          (4.0e-4, 135.0), (8.0e-4, 98.0)]   # 10.7M equal-S ladder (r2>=0.98)


def feature_clock(curve, lam_S: float, inv_tau0: float, delta: float = 0.0):
    eta = curve.lrs.astype(np.float64)
    drop = np.zeros_like(eta)
    drop[1:] = np.maximum(eta[:-1] - eta[1:], 0.0)
    if delta != 0.0:
        drop = drop * np.power(np.maximum(eta / PEAK_LR, 1e-12), delta)
    drop = drop / PEAK_LR
    dec = np.exp(-(lam_S * eta + inv_tau0))
    s = np.empty_like(eta)
    acc = 0.0
    for t in range(len(eta)):
        acc = acc * dec[t] + drop[t]
        s[t] = acc
    return s[np.asarray(curve.step, dtype=np.int64)]


def measure_public_taus():
    """Residual transient tau per (scale, wsdcon)."""
    out = []
    for scale in SCALES:
        p = MPL_PRECOMPUTED_INIT[scale]
        for n, mul in [("wsdcon_3.csv", 3), ("wsdcon_9.csv", 9),
                       ("wsdcon_18.csv", 18)]:
            c = load_curve(scale, n)
            r = c.loss - mpl_predict(p, c)
            m = (c.step >= 8000) & (c.step <= 14000)
            t = (c.step[m] - 8000).astype(float)
            y = r[m]

            def mdl(t, A, tau, c0):
                return c0 + A * np.exp(-t / tau)
            try:
                po, _ = curve_fit(mdl, t, y, p0=[max(y[0], 1e-3), 800.0, 0.0],
                                  maxfev=40000,
                                  bounds=([0, 30, -0.05], [1, 30000, 0.05]))
                pred = mdl(t, *po)
                ss = np.sum((y - y.mean()) ** 2)
                r2 = 1 - np.sum((y - pred) ** 2) / max(ss, 1e-30)
                if po[0] > 1e-3 and r2 > 0.5:
                    out.append((scale, mul * 1e-5, float(po[1]), float(r2)))
            except Exception:
                pass
    return out


def aic(n, sse, k):
    return n * np.log(max(sse, 1e-30) / n) + 2 * k


def fit_clocks(pts, label):
    """pts: list of (eta, tau).  Fit log-tau with 3 clock models."""
    eta = np.array([e for e, _ in pts])
    tau = np.array([t for _, t in pts])
    n = len(pts)
    print(f"  [{label}] n={n}")

    # pure-S: tau = 1/(lamS * eta)
    def sse_pure(loglam):
        lam = np.exp(loglam)
        return float(np.sum((np.log(tau) - np.log(1.0 / (lam * eta))) ** 2))
    r = minimize_scalar(sse_pure, bounds=(-5, 8), method="bounded")
    lam_p = float(np.exp(r.x))
    a_pure = aic(n, r.fun, 1)
    print(f"    pure-S : lamS={lam_p:8.2f}                AIC={a_pure:7.2f}")

    # affine: tau = 1/(lamS*eta + 1/tau0)
    def sse_aff(th):
        lam, it0 = np.exp(th)
        return float(np.sum((np.log(tau) - np.log(1.0 / (lam * eta + it0))) ** 2))
    best = None
    for l0 in [1.0, 10.0]:
        for t0 in [200.0, 850.0, 5000.0]:
            r = minimize(sse_aff, [np.log(l0), np.log(1.0 / t0)],
                         method="Nelder-Mead", options={"xatol": 1e-4})
            if best is None or r.fun < best.fun:
                best = r
    lam_a, it0_a = np.exp(best.x)
    a_aff = aic(n, best.fun, 2)
    print(f"    affine : lamS={lam_a:8.2f} tau0={1/it0_a:8.0f}  AIC={a_aff:7.2f}")

    # power: tau = c * eta^-p
    A = np.vstack([np.ones(n), -np.log(eta)]).T
    coef, res, *_ = np.linalg.lstsq(A, np.log(tau), rcond=None)
    sse_pow = float(np.sum((np.log(tau) - A @ coef) ** 2))
    a_pow = aic(n, sse_pow, 2)
    print(f"    power  : p={coef[1]:6.2f} c={np.exp(coef[0]):8.2f}     AIC={a_pow:7.2f}")
    return {"pure": (lam_p, a_pure), "affine": (lam_a, it0_a, a_aff),
            "power": (coef[1], a_pow)}


def fit_lam_family(scale, curves, inv_tau0, delta=0.0):
    p = MPL_PRECOMPUTED_INIT[scale]
    data = [(load_curve(scale, n),) for n in curves]
    data = [(c[0], c[0].loss - mpl_predict(p, c[0])) for c in data]

    def sse(loglam):
        lam = float(np.exp(loglam))
        tot = 0.0
        xs = [feature_clock(c, lam, inv_tau0, delta) for c, _ in data]
        x = np.concatenate(xs)
        y = np.concatenate([r for _, r in data])
        k = max(0.0, fit_origin(x, y)[0])
        return float(np.sum((y - k * x) ** 2))
    r = minimize_scalar(sse, bounds=(np.log(0.05), np.log(80)), method="bounded")
    return float(np.exp(r.x))


def main():
    print("== (i) clock-model AIC on tau measurements ==")
    pub = measure_public_taus()
    for s, e, t, r2 in pub:
        print(f"   {s:>4}M eta2={e:.1e} tau={t:7.0f} r2={r2:.2f}")
    fits_pub = fit_clocks([(e, t) for _, e, t, _ in pub], "public wsdcon")
    fits_lad = fit_clocks(LADDER, "10.7M ladder")
    # joint affine: shared functional form, per-dataset params is trivial;
    # the unification claim needs ONE rule -- compare: does affine fit BOTH
    # datasets with sensible params while pure-S fails the ladder and
    # constant-tau fails the public set?
    print("\n== (ii) per-family lam_S with tau_0 fixed (public) ==")
    lam_aff, it0_aff, _ = fits_pub["affine"]
    for it0, tag in [(0.0, "pure-S (shipped)"), (it0_aff, f"affine tau0={1/max(it0_aff,1e-12):.0f}")]:
        cells = []
        for scale in SCALES:
            lp = fit_lam_family(scale, list(PROBES), it0)
            ls = fit_lam_family(scale, list(DECAY), it0)
            cells.append(f"{scale}M {lp:5.1f}/{ls:5.1f}")
        print(f"   {tag:28s} probe/sharp lam_S: " + "  ".join(cells))

    print("\n== (iii) retrodiction: 10.7M bed deployment with ladder constants ==")
    # affine kernel with the ladder-fit constants (lam_S=5.11, tau0=168),
    # pooled-probe kappa, d=0 -- analyze_gen protocol.  Shipped (lam*=1 grid):
    # sharp600 -18.1, wsd -7.6, wsdld -6.2.
    try:
        rep = Path(__file__).resolve().parents[2] / "represent" / "repro"
        sys.path.insert(0, str(rep))
        import analyze_curves as AC  # noqa: E402
        from analyze_gen import load_gen, PEAK as MPEAK  # noqa: E402
        cv = AC.load_scale("m")
        for k, v in load_gen().items():
            cv[k] = v
        params, _ = AC.fit_mpl(cv, ["constant", "cosine", "wsdcon_20"], n_starts=8)
        resid = {s: cv[s]["loss"] - AC.mpl_pred_at(cv[s]["lr"], cv[s]["step"], params)
                 for s in cv}
        lamS_l, tau0_l = 5.11, 168.0

        def feat_m(lr, lam_S, inv_t0):
            eta = np.asarray(lr, float)
            drop = np.zeros_like(eta)
            drop[1:] = np.maximum(eta[:-1] - eta[1:], 0.0)
            drop = drop / MPEAK
            dec = np.exp(-(lam_S * eta + inv_t0))
            s_ = np.empty_like(eta)
            acc = 0.0
            for t in range(len(eta)):
                acc = acc * dec[t] + drop[t]
                s_[t] = acc
            return s_

        probes_m = ["wsdcon_5", "wsdcon_10", "wsdcon_40", "wsdcon_80"]
        xs = [feat_m(cv[s]["lr"], lamS_l, 1.0 / tau0_l)[cv[s]["step"]]
              for s in probes_m]
        ys = [resid[s] for s in probes_m]
        kap = max(0.0, fit_origin(np.concatenate(xs), np.concatenate(ys))[0])
        for s, shipped in [("sharp600", -18.1), ("wsd", -7.6), ("wsdld", -6.2)]:
            if s not in cv:
                continue
            phi = feat_m(cv[s]["lr"], lamS_l, 1.0 / tau0_l)[cv[s]["step"]]
            m0 = float(np.mean(np.abs(resid[s])))
            m1 = float(np.mean(np.abs(resid[s] - kap * phi)))
            print(f"   {s:10s} affine-retrodiction {100*(m1/m0-1):+6.1f}% "
                  f"(shipped lam*-grid: {shipped:+.1f}%)")
        print("   -> retrodiction FAILS: tau0=168 caps kernel memory far below"
              " the deployed effective memory; lam-grid step RETAINED.")
    except Exception as ex:
        print(f"   retrodiction run failed: {ex}")

    print("\n== (iv) non-regression: probes-only (matched) with affine clock ==")
    for tag, it0 in [("pure-S", 0.0), ("affine", it0_aff)]:
        m0s, m1s, wins = [], [], 0
        for scale in SCALES:
            p = MPL_PRECOMPUTED_INIT[scale]
            spec_delta = 0.25
            for n in DECAY:
                cu = load_curve(scale, n)
                cal = cal_probes_for(cu)
                xs, ys = [], []
                for cn in cal:
                    c = load_curve(scale, cn)
                    xs.append(feature_clock(c, 10.0, it0, spec_delta))
                    ys.append(c.loss - mpl_predict(p, c))
                kappa = max(0.0, fit_origin(np.concatenate(xs),
                                            np.concatenate(ys))[0])
                bp = mpl_predict(p, cu)
                m0 = metrics(cu.loss, bp)["mae"]
                m1 = metrics(cu.loss, bp + kappa * feature_clock(
                    cu, 10.0, it0, spec_delta))["mae"]
                m0s.append(m0); m1s.append(m1); wins += int(m1 < m0)
        d = 100.0 * (np.mean(m1s) / np.mean(m0s) - 1.0)
        print(f"   {tag:8s} d=1/4 matched probes-only: {d:+.1f}% {wins}/6 "
              f"(shipped matched: -27.0)")


if __name__ == "__main__":
    main()
