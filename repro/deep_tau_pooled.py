#!/usr/bin/env python3
"""Backing artifact for the headline tau-exponent (tau ~ eta^-p) from the
committed deep_tau measurements (results/deep_tau.log).  Computes per-scale
slopes, the balanced fixed-effects pooled slope (per-scale intercept, common
slope), the single-intercept pooled OLS, and the naive mean-of-slopes, each
with an SE -- so the headline number is referee-reproducible.

tau(steps) at lr_b in {3,9,18}e-5, per scale, from deep_tau.log:"""
import json
import os

import numpy as np

TAU = {  # scale -> list of (lr_b, tau)
    "25M":  [(3e-5, 2999), (9e-5, 2185), (1.8e-4, 1161)],
    "100M": [(3e-5, 2906), (9e-5, 1616), (1.8e-4, 394)],
    "400M": [(3e-5, 2884), (9e-5, 1598), (1.8e-4, 496)],
}
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    ".."))


def main():
    per_scale = {}
    for s, pts in TAU.items():
        e = np.array([p[0] for p in pts]); t = np.array([p[1] for p in pts])
        slope, icpt = np.polyfit(np.log(e), np.log(t), 1)
        # SE of slope from the 3-point regression residuals
        pred = slope * np.log(e) + icpt
        n = len(e); dof = n - 2
        sxx = np.sum((np.log(e) - np.log(e).mean()) ** 2)
        se = np.sqrt(np.sum((np.log(t) - pred) ** 2) / dof / sxx)
        per_scale[s] = (-slope, se)   # p = -slope
        print(f"  {s:>5}: p = {-slope:.3f} +/- {se:.3f}")

    ps = np.array([per_scale[s][0] for s in TAU])
    naive_mean = float(ps.mean()); naive_sd = float(ps.std(ddof=1))
    naive_se = naive_sd / np.sqrt(len(ps))
    print(f"\n  naive mean-of-per-scale p = {naive_mean:.3f} "
          f"(SD {naive_sd:.3f}, SE {naive_se:.3f})")

    # balanced fixed-effects: per-scale intercept, ONE common slope
    # (identical x per group -> equals mean of per-scale slopes; SE from pooled resid)
    lx, ly, grp = [], [], []
    for gi, s in enumerate(TAU):
        for e, t in TAU[s]:
            lx.append(np.log(e)); ly.append(np.log(t)); grp.append(gi)
    lx = np.array(lx); ly = np.array(ly); grp = np.array(grp)
    # design: common slope + 3 intercept dummies
    X = np.column_stack([lx] + [(grp == g).astype(float) for g in range(3)])
    beta, *_ = np.linalg.lstsq(X, ly, rcond=None)
    resid = ly - X @ beta
    dof = len(ly) - X.shape[1]
    XtXinv = np.linalg.inv(X.T @ X)
    se_slope = np.sqrt(np.sum(resid ** 2) / dof * XtXinv[0, 0])
    fe_p = -beta[0]
    print(f"  fixed-effects pooled p (per-scale intercept) = {fe_p:.3f} "
          f"+/- {se_slope:.3f}")

    # single-intercept pooled OLS (ignores scale)
    s2, _ = np.polyfit(lx, ly, 1)[:2]
    print(f"  single-intercept pooled OLS p = {-s2:.3f}")

    print(f"\n  PAPER currently claims p = 1.00 +/- 0.18 -- NOT reproduced by "
          f"any of the above; honest pooled = {fe_p:.2f} +/- {se_slope:.2f} "
          f"(consistent with tau~1/eta at 100M/400M; shallow at 25M).")
    json.dump({"per_scale_p": {s: per_scale[s][0] for s in TAU},
               "per_scale_se": {s: per_scale[s][1] for s in TAU},
               "naive_mean_p": naive_mean, "naive_sd": naive_sd,
               "naive_se": naive_se, "fixed_effects_p": float(fe_p),
               "fixed_effects_se": float(se_slope),
               "single_intercept_p": float(-s2),
               "note": "backs the tau-exponent headline; supersedes the "
                       "unbacked 1.00+/-0.18"},
              open(os.path.join(REPO, "results", "DEEP_TAU_POOLED.json"), "w"),
              indent=1)
    print("\nsaved results/DEEP_TAU_POOLED.json")


if __name__ == "__main__":
    main()
