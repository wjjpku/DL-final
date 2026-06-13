"""Attempt 1A analysis -- executes results/formula_lab/bladder_prereg.json
verbatim.

Per arm: fit r(t) = F_inf + A*exp(-(t-3000)/tau) on dense post-drop evals
(r2 >= 0.6 gate).  Regression: log tau = a + b_B log B2 + b_eta log eta2
(drop arms only), bootstrap 90% CI on b_B.  Paired floor gaps (drop minus
control, same seed/B2, bitwise-shared batches): gap = G(eta2) * B2^q.

Verdicts (pre-registered):
  C2 noise clock : b_B = -0.5 +/- 0.15, CI excludes 0; q in [-0.65, -0.35]
  C1/C3 B-blind  : b_B in [-0.15, 0.15], CI excludes -0.5
  UNMEASURABLE   : CI covers both
"""
import glob
import json
import os
import sys

import numpy as np
from scipy.optimize import curve_fit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CDIR = os.path.join(ROOT, "results", "curves_bladder")
TRUNK = 3000


def load(tag):
    rows = np.genfromtxt(os.path.join(CDIR, tag + ".csv"), delimiter=",",
                         names=True)
    return (np.atleast_1d(rows["step"]).astype(int),
            np.atleast_1d(rows["eval_loss"]).astype(float))


def fit_tau(tag):
    st, ev = load(tag)
    m = st >= TRUNK
    t = (st[m] - TRUNK).astype(float)
    y = ev[m]

    def mdl(t, F, A, tau):
        return F + A * np.exp(-t / tau)
    best = None
    for tau0 in (100.0, 300.0, 900.0, 2500.0):
        try:
            po, _ = curve_fit(mdl, t, y, p0=[y[-1], max(y[0] - y[-1], 1e-3),
                                             tau0],
                              maxfev=60000,
                              bounds=([0, 0, 5], [10, 5, 30000]))
            pred = mdl(t, *po)
            ss = float(np.sum((y - y.mean()) ** 2))
            r2 = 1 - float(np.sum((y - pred) ** 2)) / max(ss, 1e-30)
            if best is None or r2 > best[3]:
                best = (po[0], po[1], po[2], r2)
        except Exception:
            pass
    return best  # (F_inf, A, tau, r2)


def tail_mean(tag, last=1000):
    st, ev = load(tag)
    end = st.max()
    m = st >= end - last
    return float(np.mean(ev[m]))


def main():
    tags = [os.path.basename(p)[:-4]
            for p in glob.glob(os.path.join(CDIR, "*.csv"))]
    drops, ctrls = {}, {}
    print("== per-arm exponential fits ==")
    for tag in sorted(tags):
        parts = tag.split("_")
        B2 = int(parts[0][1:])
        et = parts[1][1:]
        seed = int(parts[2][1:])
        if et == "nodrop":
            ctrls[(B2, seed)] = tag
            continue
        eta2 = int(et) * 1e-5
        fit = fit_tau(tag)
        if fit is None:
            print(f"  {tag:18s} FIT FAILED")
            continue
        F, A, tau, r2 = fit
        used = r2 >= 0.6 and A > 1e-3
        print(f"  {tag:18s} tau={tau:7.0f} A={A:+.4f} F={F:.4f} r2={r2:.3f} "
              f"used={used}")
        if used:
            drops[(B2, eta2, seed)] = (tau, tag)

    # regression log tau = a + b_B log B2 + b_eta log eta2
    keys = list(drops.keys())
    if len(keys) >= 6:
        X = np.column_stack([np.ones(len(keys)),
                             [np.log(k[0]) for k in keys],
                             [np.log(k[1]) for k in keys]])
        Y = np.array([np.log(drops[k][0]) for k in keys])
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        resid = Y - X @ beta
        rng = np.random.default_rng(0)
        bs = []
        for _ in range(2000):
            idx = rng.integers(0, len(keys), len(keys))
            try:
                b, *_ = np.linalg.lstsq(X[idx], Y[idx], rcond=None)
                bs.append(b[1])
            except Exception:
                pass
        lo, hi = np.percentile(bs, [5, 95])
        print(f"\n== regression ==\n  b_B = {beta[1]:+.3f}  (90% CI "
              f"[{lo:+.3f}, {hi:+.3f}])   b_eta = {beta[2]:+.3f}")
        if -0.65 <= beta[1] <= -0.35 and hi < 0:
            verdict = "C2 NOISE CLOCK"
        elif -0.15 <= beta[1] <= 0.15 and (lo > -0.5 or hi < -0.5) and not (lo <= -0.5 <= hi):
            verdict = "C1/C3 B-BLIND"
        else:
            verdict = "UNMEASURABLE" if (lo <= 0 <= hi and lo <= -0.5 <= hi) \
                else "OUT-OF-BAND (report as measured)"
        print(f"  tau-verdict: {verdict}")

    # paired floor gaps
    print("\n== paired floor gaps (drop - control, last-1000 tail) ==")
    gaps = {}
    for (B2, eta2, seed), (tau, tag) in sorted(drops.items()):
        c = ctrls.get((B2, seed))
        if not c:
            continue
        g = tail_mean(tag) - tail_mean(c)
        gaps[(B2, eta2, seed)] = g
        print(f"  B2={B2:>3} eta2={eta2:.0e} s{seed}: gap={g:+.4f}")
    by = {}
    for (B2, eta2, seed), g in gaps.items():
        by.setdefault(eta2, []).append((B2, g))
    for eta2, pts in by.items():
        pts = [(b, g) for b, g in pts if g < 0]
        if len(pts) >= 3:
            q = np.polyfit([np.log(b) for b, _ in pts],
                           [np.log(-g) for _, g in pts], 1)[0]
            print(f"  q(eta2={eta2:.0e}) = {q:+.3f}  (C2 expects q in "
                  f"[-0.65,-0.35])")

    out = {"taus": {f"{k[0]}_{k[1]:.0e}_{k[2]}": drops[k][0] for k in drops},
           "gaps": {f"{k[0]}_{k[1]:.0e}_{k[2]}": gaps[k] for k in gaps}}
    json.dump(out, open(os.path.join(ROOT, "results", "BLADDER_REPORT.json"),
                        "w"), indent=1)
    print("\nsaved results/BLADDER_REPORT.json")


if __name__ == "__main__":
    main()
