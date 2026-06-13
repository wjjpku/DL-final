#!/usr/bin/env python3
"""Audit remediation 1A (finding fatal: fit-window length confounded with
B2).  Re-fit every bladder arm on COMMON post-drop windows and re-run the
pre-registered regression; evaluate the prereg buckets honestly.

Also restates paired gaps at a common post-drop horizon and re-examines the
q fit WITHOUT the un-preregistered sign-censoring.
"""
import os
import sys

import numpy as np
from scipy.optimize import curve_fit

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "represent", "repro"))
import analyze_curves as AC  # noqa: E402

CDIR = os.path.join(REPO, "represent", "results", "curves_bladder")
ARMS = []
for B in [12, 24, 48, 96, 192]:
    for e, tag in [(1e-4, "e10"), (4e-4, "e40")]:
        ARMS.append((B, e, f"b{B}_{tag}_s1337"))
for B, e, t in [(12, 1e-4, "b12_e10_s1338"), (12, 4e-4, "b12_e40_s1338"),
                (192, 1e-4, "b192_e10_s1338"), (192, 4e-4, "b192_e40_s1338")]:
    ARMS.append((B, e, t))


def load(tag):
    rows = np.genfromtxt(os.path.join(CDIR, tag + ".csv"), delimiter=",",
                         names=True)
    step = np.atleast_1d(rows["step"]).astype(int)
    loss = np.atleast_1d(rows["eval_loss"]).astype(float)
    return step, AC.smooth_by_step(step, loss)


def fit_tau(step, sm, w):
    m = (step >= 3000) & (step < 3000 + w)
    t = (step[m] - 3000).astype(float)
    y = sm[m]
    if len(t) < 10:
        return None

    def mdl(t, F, A, tau):
        return F + A * np.exp(-t / tau)
    try:
        po, _ = curve_fit(mdl, t, y, p0=[y[-1], max(y[0] - y[-1], 1e-3), 500],
                          maxfev=50000,
                          bounds=([0, 1e-4, 10], [5, 2, 30000]))
        pred = mdl(t, *po)
        r2 = 1 - np.sum((y - pred) ** 2) / max(np.sum((y - y.mean()) ** 2),
                                               1e-30)
        return po[2], r2
    except Exception:
        return None


def regress(taus):
    X = np.array([[1.0, np.log(B), np.log(e)] for B, e, _ in taus])
    y = np.log(np.array([t for _, _, t in taus]))
    th, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ th
    rng = np.random.default_rng(0)
    bs = []
    n = len(y)
    for _ in range(2000):
        i = rng.integers(0, n, n)
        try:
            b, *_ = np.linalg.lstsq(X[i], y[i], rcond=None)
            bs.append(b[1])
        except Exception:
            pass
    lo, hi = np.percentile(bs, [5, 95])
    return th[1], lo, hi, th[2]


def bucket(b, lo, hi):
    if -0.65 <= b <= -0.35 and not (lo <= 0 <= hi):
        return "C2 noise clock"
    if -0.15 <= b <= 0.15 and not (lo <= -0.5 <= hi):
        return "C1/C3 B-BLIND"
    if (lo <= 0 <= hi) and (lo <= -0.5 <= hi):
        return "UNMEASURABLE"
    return "out-of-band (no prereg bucket)"


def main():
    data = {t: load(t) for _, _, t in ARMS}
    print("== common-window regressions (log tau = a + b_B log B + b_eta "
          "log eta) ==")
    for w in [1000, 2000, 4000]:
        taus = []
        for B, e, tag in ARMS:
            r = fit_tau(*data[tag], w)
            if r and r[1] >= 0.6:
                taus.append((B, e, r[0]))
        if len(taus) < 8:
            print(f"  W={w}: only {len(taus)} arms pass r2 gate -- skip")
            continue
        b, lo, hi, be = regress(taus)
        print(f"  W={w:5d}: b_B = {b:+.3f} (90% CI [{lo:+.3f},{hi:+.3f}]) "
              f"b_eta={be:+.2f} n={len(taus)} -> {bucket(b, lo, hi)}")

    print("\n== paired gaps at common horizon [6000,7000) (drop - control, "
          "s1337) ==")
    for e2, tag in [(1e-4, "e10"), (4e-4, "e40")]:
        pts = []
        for B in [12, 24, 48, 96, 192]:
            sd, smd = data[f"b{B}_{tag}_s1337"]
            stc, smc = load(f"b{B}_enodrop_s1337")
            md = (sd >= 6000) & (sd < 7000)
            mc = (stc >= 6000) & (stc < 7000)
            if md.sum() and mc.sum():
                g = float(np.mean(smd[md]) - np.mean(smc[mc]))
                pts.append((B, g))
                print(f"  B={B:3d} eta2={e2:.0e}: gap = {g:+.4f}")
        neg = [(B, g) for B, g in pts if g < 0]
        print(f"  eta2={e2:.0e}: {len(pts)-len(neg)}/{len(pts)} gaps "
              f"non-negative; log-log q fit on sign-censored subset is NOT "
              f"reported (audit: misspecified, gaps cross zero)")


if __name__ == "__main__":
    main()
