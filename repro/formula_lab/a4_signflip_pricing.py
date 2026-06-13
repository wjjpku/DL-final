#!/usr/bin/env python3
"""ITEM A4: floor-level pricing of the large-B sign flip (zero-GPU).

Observation to price (AUDIT-A restated record, robust at common horizon
[6000,7000), s1337, eta2=1e-4): the drop-vs-control paired gap inverts with
batch: B12 -0.109, B24 -0.066, B48 -0.029, B96 +0.006, B192 +0.034.

Hypothesis to test (the item's stated form):
    gap(B) = -[floor(eta1,B) - floor(eta2,B)]   equilibrium floor benefit,
                                                shrinks with B (~ eta/B)
             + R_drop(B)                        B-BLIND remaining-transient
                                                shortfall at the horizon.

Estimators (per the item):
  floor(eta1,B)  = no-drop control's OWN tail = mean over the common horizon
                   [6000,7000) (controls at constant peak are still slowly
                   descending the backbone -- exp extrapolation is
                   misspecified for them; drift slope reported as the
                   systematic).  Horizon-indexed level, not an equilibrium
                   (AUDIT-C budget-indexed language).
  floor(eta2,B)  = F_inf from  loss(t) = F + A*exp(-(t-3000)/tau)  fitted to
                   the drop arm on the COMMON window [3000,7000)
                   (window-matched per AUDIT-A; cap sensitivity reported).
  R_drop(B)      = the fitted transient averaged over the arm's own sample
                   steps in [6000,7000) (like-for-like with the window mean).

Predictions per B vs the measured gap:
  pred_full   = (F_d + R_d) - ctrl_hz     fit-adequacy bound (near-circular:
                                          equals meas_gap + drop-fit residual
                                          at the horizon)
  pred_bblind = (F_d + R_BAR) - ctrl_hz   hypothesis-constrained: per-B
                                          transient replaced by the B-blind
                                          mean R_BAR
Functional pricing (the actual floor-level law test, 5 gaps):
  gap(B) = r0 - a * B^-s   with s free / s=1 (eta/B) / s=0.5; r0 = B-blind
  shortfall, a*B^-s = floor benefit.  Pass bar from the item: flip location
  ~ B 96 and per-B magnitudes within ~2e-3.

Cross-checks: b12/b24 far-tail means vs fitted F_inf (arms run to
11000/8600); s1338 drop-arm F_inf replicates (b12/b192; NO s1338 controls
exist, so no replicate paired gap).

G5: no MPL fit is touched anywhere in here.
Output: results/formula_lab/a4_signflip_pricing.json
"""
import json
import os
import sys

import numpy as np
from scipy.optimize import curve_fit, least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "represent", "repro"))
import analyze_curves as AC  # noqa: E402

CDIR = os.path.join(REPO, "represent", "results", "curves_bladder")
OUT = os.path.join(REPO, "results", "formula_lab", "a4_signflip_pricing.json")
BS = [12, 24, 48, 96, 192]
TRUNK = 3000
HZ = (6000, 7000)
CAP_MAIN = 7000           # common fit window cap (AUDIT-A window matching)
CAPS_SENS = [5000, None]  # sensitivity: short cap / full per-arm window


def load(tag):
    rows = np.genfromtxt(os.path.join(CDIR, tag + ".csv"), delimiter=",",
                         names=True)
    step = np.atleast_1d(rows["step"]).astype(int)
    loss = np.atleast_1d(rows["eval_loss"]).astype(float)
    return step, AC.smooth_by_step(step, loss)


def fit_relax(step, sm, cap):
    """F_inf + A*exp(-(t-3000)/tau) on [3000, cap) -- DROP arms only."""
    m = step >= TRUNK
    if cap is not None:
        m &= step < cap
    t = (step[m] - TRUNK).astype(float)
    y = sm[m]

    def mdl(t, F, A, tau):
        return F + A * np.exp(-t / tau)

    best = None
    for tau0 in (100.0, 300.0, 900.0, 2500.0, 8000.0):
        try:
            po, _ = curve_fit(mdl, t, y,
                              p0=[y[-1], max(y[0] - y[-1], 1e-3), tau0],
                              maxfev=60000, bounds=([0, 0, 5], [10, 2, 30000]))
            sse = float(np.sum((y - mdl(t, *po)) ** 2))
            if best is None or sse < best[0]:
                best = (sse, po)
        except Exception:
            pass
    sse, (F, A, tau) = best
    ss = float(np.sum((y - y.mean()) ** 2))
    return dict(F=float(F), A=float(A), tau=float(tau),
                r2=float(1 - sse / max(ss, 1e-30)))


def horizon_mean(step, sm):
    m = (step >= HZ[0]) & (step < HZ[1])
    return float(np.mean(sm[m])), m


def drift_per_1k(step, sm, lo=5000, hi=7000):
    """Linear slope of the smoothed curve over [lo,hi), per 1000 steps."""
    m = (step >= lo) & (step < hi)
    if m.sum() < 4:
        return None
    return float(np.polyfit(step[m].astype(float), sm[m], 1)[0] * 1000)


def crossing(bs, gaps):
    for i in range(len(bs) - 1):
        g0, g1 = gaps[i], gaps[i + 1]
        if g0 < 0 <= g1 or g0 >= 0 > g1:
            x0, x1 = np.log(bs[i]), np.log(bs[i + 1])
            return float(np.exp(x0 - g0 * (x1 - x0) / (g1 - g0)))
    return None


def fit_pricing_law(bs, gaps, s_fixed=None):
    """gap(B) = r0 - a * B^-s  (a>0).  Returns dict with params, per-B
    preds, maxerr, implied crossing."""
    b = np.asarray(bs, float)
    g = np.asarray(gaps, float)

    if s_fixed is None:
        def resid(th):
            r0, loga, s = th
            return g - (r0 - np.exp(loga) * b ** -s)
        best = None
        for s0 in (0.5, 1.0, 1.5):
            r = least_squares(resid, x0=[0.02, np.log(1.0), s0],
                              bounds=([-1, -15, 0.05], [1, 15, 4]))
            if best is None or r.cost < best.cost:
                best = r
        r0, loga, s = best.x
    else:
        s = float(s_fixed)
        X = np.vstack([np.ones_like(b), -(b ** -s)]).T
        th, *_ = np.linalg.lstsq(X, g, rcond=None)
        r0, a = th
        loga = np.log(max(a, 1e-12))
    a = float(np.exp(loga))
    pred = r0 - a * b ** -s
    err = np.abs(pred - g)
    bstar = (a / r0) ** (1 / s) if r0 > 0 else None
    return dict(r0=float(r0), a=a, s=float(s),
                pred={int(B): float(p) for B, p in zip(bs, pred)},
                maxerr=float(err.max()), meanerr=float(err.mean()),
                B_star=float(bstar) if bstar else None)


def price(etag, cap, data, verbose=True):
    rows = {}
    for B in BS:
        sd, smd = data[f"b{B}_{etag}_s1337"]
        sc, smc = data[f"b{B}_enodrop_s1337"]
        fd = fit_relax(sd, smd, cap)
        drop_hz, md = horizon_mean(sd, smd)
        ctrl_hz, _ = horizon_mean(sc, smc)
        R_d = float(np.mean(fd["A"] * np.exp(-(sd[md] - TRUNK) / fd["tau"])))
        rows[B] = dict(F_d=fd["F"], A_d=fd["A"], tau_d=fd["tau"],
                       r2_d=fd["r2"], R_d=R_d, ctrl_hz=ctrl_hz,
                       drop_hz=drop_hz, meas_gap=drop_hz - ctrl_hz,
                       benefit=ctrl_hz - fd["F"],
                       fit_resid_hz=(fd["F"] + R_d) - drop_hz,
                       ctrl_drift_1k=drift_per_1k(sc, smc),
                       drop_drift_1k=drift_per_1k(sd, smd))
    R_bar = float(np.mean([rows[B]["R_d"] for B in BS]))
    R_sd = float(np.std([rows[B]["R_d"] for B in BS]))
    for B in BS:
        r = rows[B]
        r["pred_full"] = (r["F_d"] + r["R_d"]) - r["ctrl_hz"]
        r["pred_bblind"] = (r["F_d"] + R_bar) - r["ctrl_hz"]
    gaps_m = [rows[B]["meas_gap"] for B in BS]
    out = dict(rows=rows, R_bar=R_bar, R_d_spread_sd=R_sd,
               cross_meas=crossing(BS, gaps_m))
    for k in ("pred_full", "pred_bblind"):
        pred = [rows[B][k] for B in BS]
        err = [abs(p - g) for p, g in zip(pred, gaps_m)]
        out[f"cross_{k}"] = crossing(BS, pred)
        out[f"maxerr_{k}"] = float(max(err))
    # functional pricing law on the measured gaps
    out["law_sfree"] = fit_pricing_law(BS, gaps_m)
    out["law_s1"] = fit_pricing_law(BS, gaps_m, s_fixed=1.0)
    out["law_s05"] = fit_pricing_law(BS, gaps_m, s_fixed=0.5)
    # same law on the benefit term alone (floor-level side of the ledger)
    ben = [rows[B]["benefit"] for B in BS]
    out["law_benefit_sfree"] = fit_pricing_law(BS, [-x for x in ben])
    if verbose:
        print(f"\n== {etag}  cap={cap}  R_bar={R_bar*1e3:+.2f}e-3 "
              f"(per-B sd {R_sd*1e3:.2f}e-3) ==")
        print("   B   meas_gap  pred_full pred_bblind  benefit    R_d     "
              "tau_d  r2_d  fitres_hz ctrl_drift/1k")
        for B in BS:
            r = rows[B]
            print(f"  {B:4d} {r['meas_gap']*1e3:+8.1f}e-3 "
                  f"{r['pred_full']*1e3:+7.1f}e-3 "
                  f"{r['pred_bblind']*1e3:+8.1f}e-3 "
                  f"{r['benefit']*1e3:+7.1f}e-3 {r['R_d']*1e3:+6.2f}e-3 "
                  f"{r['tau_d']:6.0f} {r['r2_d']:.3f} "
                  f"{r['fit_resid_hz']*1e3:+6.2f}e-3 "
                  f"{r['ctrl_drift_1k']*1e3:+6.2f}e-3")
        print(f"  sign flip:  measured B* = {fmt(out['cross_meas'])}   "
              f"pred_full B* = {fmt(out['cross_pred_full'])}   "
              f"pred_bblind B* = {fmt(out['cross_pred_bblind'])}")
        print(f"  |pred-meas| max: full {out['maxerr_pred_full']*1e3:.2f}e-3"
              f"  bblind {out['maxerr_pred_bblind']*1e3:.2f}e-3  (bar ~2)")
        for k in ("law_sfree", "law_s1", "law_s05"):
            L = out[k]
            print(f"  {k:12s}: gap = {L['r0']*1e3:+.1f}e-3 - "
                  f"{L['a']:.3f}*B^-{L['s']:.2f}  maxerr "
                  f"{L['maxerr']*1e3:.2f}e-3  B* = {fmt(L['B_star'])}")
    return out


def fmt(x):
    return f"{x:.0f}" if x is not None else "none"


def main():
    data = {}
    for B in BS:
        for tag in [f"b{B}_e10_s1337", f"b{B}_e40_s1337",
                    f"b{B}_enodrop_s1337"]:
            data[tag] = load(tag)

    report = {"design": dict(horizon=HZ, cap_main=CAP_MAIN, trunk=TRUNK,
                             eta1=1.5e-3)}
    report["e10_main"] = price("e10", CAP_MAIN, data)
    report["e40_secondary"] = price("e40", CAP_MAIN, data)
    for cap in CAPS_SENS:
        report[f"e10_cap{cap if cap else 'full'}"] = price("e10", cap, data)

    # -- cross-checks ------------------------------------------------------
    print("\n== cross-checks ==")
    xc = {}
    for B, lo, hi in [(12, 10000, 11000), (24, 7600, 8600)]:
        sd, smd = data[f"b{B}_e10_s1337"]
        m = (sd >= lo) & (sd < hi)
        ft = float(np.mean(smd[m]))
        fi = report["e10_main"]["rows"][B]["F_d"]
        xc[f"fartail_b{B}_e10"] = dict(tail=ft, F_inf=fi, diff=ft - fi)
        print(f"  b{B} e10 far tail [{lo},{hi}) = {ft:.4f} vs F_inf {fi:.4f}"
              f"  (diff {1e3*(ft-fi):+.1f}e-3 -> still descending past F_inf)")
    for B in [12, 192]:
        s2, sm2 = load(f"b{B}_e10_s1338")
        f2 = fit_relax(s2, sm2, CAP_MAIN)
        hz2, _ = horizon_mean(s2, sm2)
        r1 = report["e10_main"]["rows"][B]
        xc[f"seed_b{B}_e10"] = dict(F_s1337=r1["F_d"], F_s1338=f2["F"],
                                    hz_s1337=r1["drop_hz"], hz_s1338=hz2)
        print(f"  b{B} e10 s1337 vs s1338: F_inf {r1['F_d']:.4f}/{f2['F']:.4f}"
              f" (d {1e3*(f2['F']-r1['F_d']):+.1f}e-3), horizon mean "
              f"{r1['drop_hz']:.4f}/{hz2:.4f} (d {1e3*(hz2-r1['drop_hz']):+.1f}e-3)")
    report["cross_checks"] = xc

    # -- E5 forecast (drift-based, falsifiable) ----------------------------
    # Both arms are still descending at every B; controls FASTER (lower
    # noise floor at large B exposes backbone descent; control accumulates
    # S 15x faster than the e10 drop arm).  Linear extrapolation of the
    # [5000,7000) drifts predicts the gap at a +2k-extended horizon
    # [8000,9000): the flip should STRENGTHEN and B* move DOWN -- the
    # settled-floor picture (gap frozen/decaying) is the alternative E5
    # discriminates against.
    print("\n== E5 drift forecast (e10, horizon extended to [8000,9000)) ==")
    e5 = {}
    for B in BS:
        r = report["e10_main"]["rows"][B]
        grow = r["drop_drift_1k"] - r["ctrl_drift_1k"]
        g2 = r["meas_gap"] + 2.0 * grow
        e5[B] = dict(gap_growth_per_1k=grow, gap_now=r["meas_gap"],
                     gap_pred_8k9k=g2)
        print(f"  B={B:3d}: growth {grow*1e3:+.2f}e-3/1k -> gap "
              f"{r['meas_gap']*1e3:+.1f}e-3 now, {g2*1e3:+.1f}e-3 at +2k")
    report["e5_forecast"] = e5

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(report, open(OUT, "w"), indent=1, default=float)
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
