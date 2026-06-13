"""Adversarial audit of analyze_floor2.py Stage-1 verdict.

Recomputes floors under multiple tail definitions and refits p:
  A) shipped: mean of smoothed eval over last max(4, n//4) ROWS  (analyzer)
  B) steps25: mean over rows with step >= 3000 + 0.75*T2   (prereg wording)
  C) steps10: mean over rows with step >= 3000 + 0.90*T2
  D) last5:   mean of final 5 rows (end state)
Also reports the actual step at which each shipped tail window starts, and
the settledness slope (smoothed eval vs step, last 25% of stage 2).
"""
import glob
import os
import sys

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_curves as AC

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PEAK = 1.5e-3
RUNG_ETA = {"5": 0.5e-4, "10": 1e-4, "20": 2e-4, "30": 3e-4, "40": 4e-4,
            "60": 6e-4, "80": 8e-4, "150": 1.5e-3}
N_BOOT = 1000


def collect(scale):
    out = {}
    if scale == "m":
        for f in glob.glob(os.path.join(ROOT, "results", "curves_floor",
                                        "floor_*.csv")):
            r = os.path.basename(f)[:-4].split("_")[1]
            out[(r, 1337)] = f
    for f in glob.glob(os.path.join(ROOT, "results", f"curves_floor_{scale}",
                                    "floor_*.csv")):
        parts = os.path.basename(f)[:-4].split("_")
        seed = int(parts[2][1:]) if len(parts) == 3 else 1337
        out[(parts[1], seed)] = f
    return out


def measures(path, rung):
    rows = np.genfromtxt(path, delimiter=",", names=True)
    step = np.atleast_1d(rows["step"]).astype(int)
    loss = np.atleast_1d(rows["eval_loss"]).astype(float)
    sm = AC.smooth_by_step(step, loss)
    n = len(step)
    k = max(4, n // 4)
    shipped = float(np.mean(sm[-k:]))
    start_shipped = int(step[-k])
    eta2 = RUNG_ETA[rung]
    T2 = int(round(1.2 / eta2))
    cut25 = 3000 + 0.75 * T2
    cut10 = 3000 + 0.90 * T2
    m25 = step >= cut25
    m10 = step >= cut10
    f25 = float(np.mean(sm[m25]))
    f10 = float(np.mean(sm[m10]))
    last5 = float(np.mean(sm[-5:]))
    # settledness: OLS slope of smoothed eval on step over last 25% of stage2
    if m25.sum() >= 4:
        A = np.vstack([np.ones(m25.sum()), step[m25] / 1000.0]).T
        th, *_ = np.linalg.lstsq(A, sm[m25], rcond=None)
        slope = float(th[1])  # loss per 1000 steps
    else:
        slope = np.nan
    return dict(shipped=shipped, f25=f25, f10=f10, last5=last5,
                start_shipped=start_shipped, n=n, T2=T2, slope=slope)


def fit_p(etas, fls):
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
    res = resid(best.x)
    rng = np.random.default_rng(0)
    ps = []
    for _ in range(N_BOOT):
        fb = (L + np.exp(loga) * x ** p) + rng.choice(res, len(res),
                                                      replace=True)
        try:
            r = least_squares(
                lambda th: fb - (th[0] + np.exp(th[1]) * x ** th[2]),
                x0=best.x, bounds=([0.5, -10, 0.2], [fb.min(), 3, 3.5]))
            ps.append(r.x[2])
        except Exception:
            pass
    lo, hi = np.percentile(ps, [5, 95])
    return float(p), float(lo), float(hi), float(L)


def fit_report(label, etas, fls):
    p, lo, hi, L = fit_p(np.array(etas), np.array(fls))
    print(f"    {label:34s} p={p:.3f}  90%CI[{lo:.3f},{hi:.3f}]  L={L:.4f}")
    return p, lo, hi


def main():
    all_meas = {}
    for scale in ["m", "l"]:
        files = collect(scale)
        print(f"\n===== scale {scale} =====")
        per = {}
        for (r, seed), f in sorted(files.items(),
                                   key=lambda kv: (RUNG_ETA[kv[0][0]],
                                                   kv[0][1])):
            ms = measures(f, r)
            per[(r, seed)] = ms
            print(f"  floor_{r:<4s} s{seed} n={ms['n']:4d} T2={ms['T2']:6d} "
                  f"shipped-tail-start={ms['start_shipped']:6d} "
                  f"(25%cut={3000+0.75*ms['T2']:.0f}) "
                  f"shipped={ms['shipped']:.4f} f25={ms['f25']:.4f} "
                  f"f10={ms['f10']:.4f} last5={ms['last5']:.4f} "
                  f"slope/1k={ms['slope']:+.5f}")
        all_meas[scale] = per

    print("\n===== p refits =====")
    res = {}
    for scale in ["m", "l"]:
        per = all_meas[scale]
        rungs = sorted({r for r, s in per}, key=lambda r: RUNG_ETA[r])
        print(f"\n  scale {scale}:")
        for key in ["shipped", "f25", "f10", "last5"]:
            etas, fls = [], []
            for r in rungs:
                vals = [per[(r, s)][key] for s in (1337, 1338)
                        if (r, s) in per]
                etas.append(RUNG_ETA[r])
                fls.append(np.mean(vals))
            res[(scale, key)] = fit_report(f"pooled {key}", etas, fls)
        # per-seed, shipped + f25
        for seed in (1337, 1338):
            for key in ["shipped", "f25"]:
                sel = [r for r in rungs if (r, seed) in per]
                if len(sel) >= 6:
                    e = [RUNG_ETA[r] for r in sel]
                    f_ = [per[(r, seed)][key] for r in sel]
                    fit_report(f"seed {seed} {key}", e, f_)
        # drop the no-drop anchor rung 150
        for key in ["shipped", "f25"]:
            etas, fls = [], []
            for r in rungs:
                if r == "150":
                    continue
                vals = [per[(r, s)][key] for s in (1337, 1338)
                        if (r, s) in per]
                etas.append(RUNG_ETA[r])
                fls.append(np.mean(vals))
            fit_report(f"pooled {key} NO-150", etas, fls)

    print("\n===== verdict re-eval per tail definition =====")
    for key in ["shipped", "f25", "f10", "last5"]:
        pm, lm, hm = res[("m", key)]
        pl, ll, hl = res[("l", key)]
        hw = float(np.sqrt((((hm - lm) / 2) ** 2 + ((hl - ll) / 2) ** 2) / 2))
        dp = pl - pm
        primary = dp > 0 and dp > 2 * hw
        disjoint = ll > hm or lm > hl
        gate = disjoint or pl >= 0.9
        if ll > 1:
            harden = "fire"
        elif hl <= 1:
            harden = "pool"
        else:
            harden = "NOT SCOREABLE"
        print(f"  {key:8s} dp={dp:+.3f} 2hw={2*hw:.3f} "
              f"primary={'FIRE' if primary else 'no'} "
              f"gate={'OPEN' if gate else 'CLOSED'} "
              f"m_CI_excl1={hm < 1} harden_l={harden}")


if __name__ == "__main__":
    main()
