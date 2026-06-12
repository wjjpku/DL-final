"""Multi-seed paired deposit-ratio measurement A2/A1.

For each seed s in {1337 (no suffix), 1338, 1339}:
  d1_s(t) = onedrop_s - constant_s  (post-2500 window)
  d2_s(t) = twodrop_s - onedrop_s   (post-4500 window)
Average the difference curves across seeds on a common step grid (noise
drops ~sqrt(3)), then fit c0 + c1*t + A*exp(-t/tau) on the averaged raw
difference; report A2/A1 across a specification grid (window x tau-mode)
with min/max spread.  Predictions: d=0: 0.700 | p=1.25: 0.564 |
d=0.5: 0.383 | p=2: 0.303.
"""
import os, sys, glob
import numpy as np
from scipy.optimize import curve_fit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_curves as AC

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GENDIR = os.path.join(ROOT, "results", "curves_gen")
CURVEDIR = os.path.join(ROOT, "results", "curves")
SEEDS = ["", "_s1338", "_s1339"]


def load_csv(path):
    rows = np.genfromtxt(path, delimiter=",", names=True)
    return (np.atleast_1d(rows["step"]).astype(int),
            np.atleast_1d(rows["eval_loss"]).astype(float))


def get(name, suffix):
    # constant for seed 1337 lives in results/curves/m_constant.csv
    cands = [os.path.join(GENDIR, f"{name}{suffix}.csv")]
    if name == "constant" and suffix == "":
        cands.append(os.path.join(CURVEDIR, "m_constant.csv"))
    for p in cands:
        if os.path.exists(p):
            return load_csv(p)
    return None


def diff_curve(a, b, drop, window):
    """Raw difference (a-b) on a's grid restricted to [drop, drop+window]."""
    sa, la = a
    sb, lb = b
    m = (sa >= drop) & (sa <= drop + window)
    t = sa[m] - drop
    d = la[m] - np.interp(sa[m], sb, lb)
    return t.astype(float), d


def fit_amp(t, d, tau_fix=None):
    if tau_fix:
        def mdl(t, c0, c1, A):
            return c0 + c1 * t + A * np.exp(-t / tau_fix)
        p0 = [d[-1], 0.0, max(d[0] - d[-1], 1e-3)]
        po, _ = curve_fit(mdl, t, d, p0=p0, maxfev=60000,
                          bounds=([-1, -1e-2, -1], [1, 1e-2, 1]))
        A, tau = po[2], tau_fix
        pred = mdl(t, *po)
    else:
        def mdl(t, c0, c1, A, tau):
            return c0 + c1 * t + A * np.exp(-t / tau)
        p0 = [d[-1], 0.0, max(d[0] - d[-1], 1e-3), 300.0]
        po, _ = curve_fit(mdl, t, d, p0=p0, maxfev=60000,
                          bounds=([-1, -1e-2, 0, 20], [1, 1e-2, 1, 4000]))
        A, tau = po[2], po[3]
        pred = mdl(t, *po)
    ss = float(np.sum((d - d.mean()) ** 2))
    r2 = 1 - float(np.sum((d - pred) ** 2)) / ss if ss > 0 else float("nan")
    return A, tau, r2


def main():
    # collect per-seed difference curves on each seed's own grid, then
    # average via interpolation onto the first seed's grid
    pairs = {"d1": ("onedrop", "constant", 2500), "d2": ("twodrop", "onedrop", 4500)}
    avg = {}
    for key, (na, nb, drop) in pairs.items():
        grids, vals = [], []
        for sfx in SEEDS:
            a = get(na, sfx); b = get(nb, sfx)
            if a is None or b is None:
                print(f"  [missing] {na}{sfx} or {nb}{sfx}")
                continue
            t, d = diff_curve(a, b, drop, 1500)
            grids.append(t); vals.append(d)
        if not vals:
            return
        t0 = grids[0]
        stack = [vals[0]] + [np.interp(t0, g, v) for g, v in zip(grids[1:], vals[1:])]
        avg[key] = (t0, np.mean(stack, axis=0), len(stack))
        print(f"{key}: averaged over {len(stack)} seeds, n={len(t0)} points")

    print(f"\n{'spec':24s} {'A1':>8s} {'tau1':>6s} {'r2':>6s} {'A2':>8s} "
          f"{'tau2':>6s} {'r2':>6s} {'A2/A1':>7s}")
    ratios = []
    for W in [1000, 1200, 1500]:
        for tf in [None, 850.0]:
            try:
                t1, d1, _ = avg["d1"]; m1 = t1 <= W
                t2, d2, _ = avg["d2"]; m2 = t2 <= W
                A1, tau1, r21 = fit_amp(t1[m1], d1[m1], tf)
                A2, tau2, r22 = fit_amp(t2[m2], d2[m2], tf)
                r = A2 / A1 if abs(A1) > 1e-9 else float("nan")
                ratios.append(r)
                lbl = f"w={W} " + ("tau-free" if tf is None else "tau=850")
                print(f"{lbl:24s} {A1:+8.4f} {tau1:6.0f} {r21:6.2f} "
                      f"{A2:+8.4f} {tau2:6.0f} {r22:6.2f} {r:7.3f}")
            except Exception as e:
                print(f"w={W} tf={tf}: fit failed {e}")
    if ratios:
        print(f"\nA2/A1 spec range: [{np.min(ratios):.3f}, {np.max(ratios):.3f}]  "
              f"median {np.median(ratios):.3f}")
        print("predictions: d=0: 0.700 | p=1.25: 0.564 | d=0.5: 0.383 | p=2: 0.303")


if __name__ == "__main__":
    main()
