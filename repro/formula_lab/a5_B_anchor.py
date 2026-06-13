#!/usr/bin/env python3
"""ITEM A5: split-free B_m anchor for the kappa-transfer ratio.

The V5 kappa transfer (kappa_l = kappa_m * B_l/B_m) was left "closure-
dependent; not separated", partly because B_l/B_m is split-unstable at m
(B_m = 581 on the wsdcon_20 split vs 277 on wsdcon_40 -> committed ratios
1.18 vs 2.42 in optsched_predictions_l.json).

The B-identity itself is dL_eq/deta ~ B.  Here we measure dL_eq/deta
DIRECTLY from the equal-S ladder floors (corrected design-window floors of
represent/results/curves_floor*, via analyze_floor2.collect/tail_floor --
the AUDIT-C protocol, NO MPL fit anywhere) at both scales, and form the
measurement-anchored ratio

    R_anchor = (dL_eq/deta)_l / (dL_eq/deta)_m   at matched eta.

Per scale we fit the ladder power law F(eta) = L + a*(eta/peak)^p exactly
as analyze_floor2.fit_p, then take the local slope dF/deta =
a*p*x^(p-1)/peak at the geometric mid-rung eta (and across the rung grid);
plus a model-free adjacent-rung finite-difference cross-check.  Bootstrap
CIs: residual bootstrap (primary, same protocol as fit_p) and rung-pair
bootstrap (sensitivity).

Then (SENSITIVITY-ONLY, G5: the backbone refits below are the existing
derby_likeforlike like-for-like machinery, identical eval grid, nothing
verdict-bearing is refit) we restate the V5 like-for-like chi2 with the
anchored ratio in place of the fitted-B ratio.

Output: results/formula_lab/a5_B_anchor.json
"""
import json
import os
import sys

import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "represent", "repro"))
sys.path.insert(0, HERE)
import analyze_floor2 as AF  # noqa: E402  (collect/tail_floor/RUNG_ETA/PEAK)
import derby_likeforlike as D  # noqa: E402

PEAK = AF.PEAK
N_BOOT = 1000
COMMITTED = {"wsdcon_20 (seeds 0/1)": [1.1769053491853123, 1.1920315034396807],
             "wsdcon_40 (seeds 0/1)": [2.4168409931353123, 2.4168409931353123]}


def ladder_floors(scale):
    """Pooled-seed design-window floors per rung (analyze_floor2 protocol)."""
    files = AF.collect(scale)
    per_rung = {}
    for (r, seed), f in sorted(files.items()):
        fl, _, _ = AF.tail_floor(f, AF.RUNG_ETA[r])
        per_rung.setdefault(r, {})[seed] = fl
    rungs = sorted(per_rung, key=lambda r: AF.RUNG_ETA[r])
    etas = np.array([AF.RUNG_ETA[r] for r in rungs])
    pooled = np.array([np.mean(list(per_rung[r].values())) for r in rungs])
    return rungs, etas, pooled, per_rung


def fit_pow(etas, fls):
    """F = L + exp(loga)*x^p, x = eta/PEAK -- identical model/bounds to
    analyze_floor2.fit_p; returns (L, a, p, residuals)."""
    x = etas / PEAK

    def resid(th):
        L, loga, p = th
        return fls - (L + np.exp(loga) * x ** p)

    best = None
    for p0 in (0.7, 1.0, 1.4, 2.0):
        r = least_squares(resid, x0=[fls.min() - 0.02, np.log(0.05), p0],
                          bounds=([0.5, -10, 0.2], [fls.min(), 3, 3.5]))
        if best is None or r.cost < best.cost:
            best = r
    L, loga, p = best.x
    return float(L), float(np.exp(loga)), float(p), resid(best.x)


def slope(a, p, eta):
    """dF/deta = a*p*x^(p-1)/PEAK at eta."""
    x = eta / PEAK
    return a * p * x ** (p - 1) / PEAK


def boot_ratio(etas_m, fls_m, etas_l, fls_l, eta_eval, n=N_BOOT, mode="resid"):
    """Bootstrap R_anchor = slope_l/slope_m at eta_eval.
    mode 'resid': residual bootstrap per scale (fit_p protocol).
    mode 'pair' : resample rungs with replacement (>=5 unique etas)."""
    rng = np.random.default_rng(0)
    fits = {"m": fit_pow(etas_m, fls_m), "l": fit_pow(etas_l, fls_l)}
    data = {"m": (etas_m, fls_m), "l": (etas_l, fls_l)}
    out = []
    for _ in range(n):
        sl = {}
        try:
            for sc in ("m", "l"):
                e, f = data[sc]
                L, a, p, res = fits[sc]
                if mode == "resid":
                    fb = (f - res) + rng.choice(res, len(res), replace=True)
                    eb = e
                else:
                    idx = rng.integers(0, len(e), len(e))
                    if len(np.unique(e[idx])) < 5:
                        raise ValueError
                    eb, fb = e[idx], f[idx]
                Lb, ab, pb, _ = fit_pow(eb, fb)
                sl[sc] = slope(ab, pb, eta_eval)
            out.append(sl["l"] / sl["m"])
        except Exception:
            pass
    lo, hi = np.percentile(out, [5, 95])
    return float(lo), float(hi), len(out)


def finite_diff(etas, fls):
    """Model-free adjacent-rung slopes at geometric midpoints."""
    mids = np.sqrt(etas[1:] * etas[:-1])
    sl = np.diff(fls) / np.diff(etas)
    return mids, sl


def main():
    report = {"protocol": "analyze_floor2 collect/tail_floor design-window "
                          "floors (AUDIT-C corrected); no MPL fit in the "
                          "anchor measurement"}
    lad = {}
    for sc in ("m", "l"):
        rungs, etas, pooled, per_rung = ladder_floors(sc)
        L, a, p, res = fit_pow(etas, pooled)
        lad[sc] = dict(rungs=rungs, etas=etas, pooled=pooled,
                       per_rung=per_rung, L=L, a=a, p=p, res=res)
        print(f"scale {sc}: {len(rungs)} rungs, F = {L:.4f} + {a:.4f}*x^"
              f"{p:.3f}  (rms resid {np.sqrt(np.mean(res**2)):.4f})")
        report[f"fit_{sc}"] = dict(L=L, a=a, p=p,
                                   rms=float(np.sqrt(np.mean(res ** 2))))

    # shared eta grid + geometric mid-rung
    e_m, e_l = lad["m"]["etas"], lad["l"]["etas"]
    shared = sorted(set(e_m) & set(e_l))
    eta_mid = float(np.sqrt(shared[0] * shared[-1]))
    print(f"\nshared rung etas: {[f'{e:.0e}' for e in shared]}; geometric "
          f"mid eta = {eta_mid:.3e}")

    sm = slope(lad["m"]["a"], lad["m"]["p"], eta_mid)
    sl_ = slope(lad["l"]["a"], lad["l"]["p"], eta_mid)
    R = sl_ / sm
    print(f"(dL/deta)_m(mid) = {sm:.1f}   (dL/deta)_l(mid) = {sl_:.1f}"
          f"   ->  R_anchor = {R:.3f}")
    grid = {f"{e:.1e}": float(slope(lad['l']['a'], lad['l']['p'], e)
                              / slope(lad['m']['a'], lad['m']['p'], e))
            for e in shared}
    print("R_anchor across rung etas:",
          {k: round(v, 3) for k, v in grid.items()})

    lo, hi, nb = boot_ratio(e_m, lad["m"]["pooled"], e_l, lad["l"]["pooled"],
                            eta_mid, mode="resid")
    lo2, hi2, nb2 = boot_ratio(e_m, lad["m"]["pooled"], e_l,
                               lad["l"]["pooled"], eta_mid, mode="pair")
    print(f"residual bootstrap 90% CI [{lo:.3f}, {hi:.3f}] (n={nb})")
    print(f"rung-pair bootstrap 90% CI [{lo2:.3f}, {hi2:.3f}] (n={nb2})")

    # per-seed sensitivity
    seed_R = {}
    for seed in (1337, 1338):
        try:
            fm = np.array([lad["m"]["per_rung"][r][seed]
                           for r in lad["m"]["rungs"]])
            fl = np.array([lad["l"]["per_rung"][r][seed]
                           for r in lad["l"]["rungs"]])
            _, am, pm, _ = fit_pow(e_m, fm)
            _, al, plw, _ = fit_pow(e_l, fl)
            seed_R[seed] = float(slope(al, plw, eta_mid)
                                 / slope(am, pm, eta_mid))
        except Exception:
            pass
    print(f"per-seed R_anchor: {seed_R}")

    # model-free finite differences at matched adjacent-rung midpoints
    mm, sm_fd = finite_diff(e_m, lad["m"]["pooled"])
    ml, sl_fd = finite_diff(e_l, lad["l"]["pooled"])
    fd = {}
    for i, (em_, el_) in enumerate(zip(mm, ml)):
        if abs(em_ - el_) < 1e-12:
            fd[f"{em_:.2e}"] = float(sl_fd[i] / sm_fd[i])
    fdv = np.array(list(fd.values()))
    print(f"finite-difference R per adjacent-rung midpoint: "
          f"{ {k: round(v, 3) for k, v in fd.items()} }")
    print(f"  median {np.median(fdv):.3f}, range [{fdv.min():.3f}, "
          f"{fdv.max():.3f}]")

    verdict = {}
    for name, vals in COMMITTED.items():
        v = float(np.mean(vals))
        verdict[name] = dict(ratio=v, inside_resid_CI=bool(lo <= v <= hi),
                             inside_pair_CI=bool(lo2 <= v <= hi2))
        print(f"committed {name}: ratio {v:.3f} -> inside resid-CI: "
              f"{lo <= v <= hi}; inside pair-CI: {lo2 <= v <= hi2}")
    report.update(eta_mid=eta_mid, dLdeta_m_mid=float(sm),
                  dLdeta_l_mid=float(sl_), R_anchor=float(R),
                  R_grid=grid, ci_resid=[lo, hi], ci_pair=[lo2, hi2],
                  per_seed_R=seed_R, finite_diff_R=fd,
                  fd_median=float(np.median(fdv)),
                  committed_vs_anchor=verdict,
                  B_committed=dict(B_m=[581.5, 574.1, 276.6],
                                   B_l=[684.4, 684.4, 668.6]))

    # ---- V5 like-for-like sensitivity under the anchored ratio ----------
    print("\n==== V5 like-for-like SENSITIVITY (G5: derby_likeforlike "
          "machinery, anchored ratio; not verdict-bearing) ====")
    pl = json.load(open(os.path.join(REPO, "results", "formula_lab",
                                     "optsched_predictions_l.json")))
    rat_committed = float(np.mean([e["kappa_ratio"] for e in pl["ensemble"]]))
    print(f"anchored ratio {R:.3f} vs committed-ensemble mean "
          f"{rat_committed:.3f}")

    def mk(r):
        return lambda p: {t: D.KAP_M[t] * r for t in D.KAP_M}

    cv_l = D.load_suite_l()
    l_dirs = os.path.join(REPO, "represent", "results", "curves_optsched_l")
    l_seeds = {1300: [1337, 1338, 1339],
               3000: [1337, 1338, 1339, 1340, 1341],
               5000: [1337, 1338, 1339],
               5700: [1337, 1338, 1339, 1340, 1341]}
    res_l = D.run_bed("l/A5", cv_l,
                      [("B_anchor", mk(R)),
                       ("B_id_committed", mk(rat_committed)),
                       ("naive", mk(1.0))],
                      l_dirs, l_seeds, "SENSITIVITY-ONLY (G5)")
    ch = res_l["chi2"]
    print("\n  V5 deltas (positive = B_anchor better; >=6 separates):")
    v5 = {}
    for arm in [t for t, _, _ in D.ARMS]:
        dn = ch["naive"][arm] - ch["B_anchor"][arm]
        db = ch["B_id_committed"][arm] - ch["B_anchor"][arm]
        v5[arm] = dict(chi2_anchor=ch["B_anchor"][arm],
                       chi2_naive=ch["naive"][arm],
                       chi2_Bid_committed=ch["B_id_committed"][arm],
                       d_naive_minus_anchor=float(dn),
                       d_Bid_minus_anchor=float(db))
        print(f"    {arm:7s}: chi2 anchor={ch['B_anchor'][arm]:7.1f} "
              f"naive={ch['naive'][arm]:7.1f} "
              f"Bid={ch['B_id_committed'][arm]:7.1f} | "
              f"naive-anchor {dn:+7.1f} "
              f"({'B_anchor' if dn >= 6 else 'naive' if dn <= -6 else 'NOT SEP'}) "
              f"Bid-anchor {db:+7.1f}")
    report["v5_sensitivity"] = dict(label="SENSITIVITY-ONLY (G5)",
                                    rat_committed_mean=rat_committed,
                                    chi2=v5,
                                    gaps_measured=res_l["gaps"])

    op = os.path.join(REPO, "results", "formula_lab", "a5_B_anchor.json")
    json.dump(report, open(op, "w"), indent=1, default=float)
    print("\nwrote", op)


if __name__ == "__main__":
    main()
