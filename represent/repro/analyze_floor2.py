"""Attempt 2 verdicts: p(N) / p_tau(N) across the single-recipe scale ladder.
Executes scaleladder_prereg.json verbatim (NLS + residual bootstrap exactly
as analyze_floor.py / scaleladder_rehearsal.py).

m floors come from the original 6-rung bed (results/curves_floor, s1337,
bs=48) + gap rungs floor_30/60 (curves_floor_m, s1337) + the full s1338
replicate; l (and gated ml/xl) from curves_floor_<scale>.
"""
import glob
import json
import os
import sys

import numpy as np
from scipy.optimize import curve_fit, least_squares

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_curves as AC

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PEAK = 1.5e-3
RUNG_ETA = {"5": 0.5e-4, "10": 1e-4, "20": 2e-4, "30": 3e-4, "40": 4e-4,
            "60": 6e-4, "80": 8e-4, "150": 1.5e-3}
N_BOOT = 1000


def tail_floor(path):
    rows = np.genfromtxt(path, delimiter=",", names=True)
    step = np.atleast_1d(rows["step"]).astype(int)
    loss = np.atleast_1d(rows["eval_loss"]).astype(float)
    sm = AC.smooth_by_step(step, loss)
    n = len(step)
    return float(np.mean(sm[-max(4, n // 4):])), step, sm


def collect(scale):
    """{(rung, seed): path} for one scale."""
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
    return float(p), float(lo), float(hi), float(L), float(np.exp(loga))


def tau_ladder(scale, files):
    """tau(eta2) from the dense post-drop window (analyze_floor protocol)."""
    taus = {}
    for (r, seed), f in sorted(files.items()):
        eta2 = RUNG_ETA[r]
        if eta2 >= PEAK:
            continue
        _, step, sm = tail_floor(f)
        m = (step >= 3000) & (step <= 5500)
        t = (step[m] - 3000).astype(float)
        y = sm[m]
        if len(t) < 10:
            continue

        def mdl(t, a, b, amp, tau):
            return a + b * t + amp * np.exp(-t / tau)
        try:
            po, _ = curve_fit(mdl, t, y,
                              p0=[y[-1], 0.0, max(y[0] - y[-1], 1e-3), 300.0],
                              maxfev=50000,
                              bounds=([0, -1e-3, 0, 10], [5, 1e-3, 2, 6000]))
            pred = mdl(t, *po)
            r2 = 1 - np.sum((y - pred) ** 2) / max(
                np.sum((y - y.mean()) ** 2), 1e-30)
            if po[2] > 1e-3 and r2 > 0.6 and 10 < po[3] < 5900:
                taus.setdefault(eta2, []).append(float(po[3]))
        except Exception:
            pass
    return {e: float(np.mean(v)) for e, v in sorted(taus.items())}


def main():
    report = {}
    pooled = {}
    for scale in ["m", "ml", "l", "xl"]:
        files = collect(scale)
        if len(files) < 6:
            continue
        print(f"\n== scale {scale} ({len(files)} rung files) ==")
        per_rung = {}
        for (r, seed), f in sorted(files.items(),
                                   key=lambda kv: RUNG_ETA[kv[0][0]]):
            fl, _, _ = tail_floor(f)
            per_rung.setdefault(r, {})[seed] = fl
            print(f"  floor_{r:<4s} s{seed}: {fl:.4f}")
        rungs = sorted(per_rung, key=lambda r: RUNG_ETA[r])
        etas = np.array([RUNG_ETA[r] for r in rungs])
        fls = np.array([np.mean(list(per_rung[r].values())) for r in rungs])
        mono = bool(np.all(np.diff(fls) > 0))
        p, lo, hi, L0, a0 = fit_p(etas, fls)
        print(f"  pooled-seed fit: floor = {L0:.4f} + {a0:.4f}*x^{p:.3f}"
              f"   90% CI [{lo:.3f},{hi:.3f}]  monotone={mono}")
        per_seed_p = {}
        for seed in [1337, 1338]:
            sel = [r for r in rungs if seed in per_rung[r]]
            if len(sel) >= 6:
                e = np.array([RUNG_ETA[r] for r in sel])
                f_ = np.array([per_rung[r][seed] for r in sel])
                ps, plo, phi, _, _ = fit_p(e, f_)
                per_seed_p[seed] = ps
                print(f"  seed {seed} only: p={ps:.3f} [{plo:.3f},{phi:.3f}]")
        taus = tau_ladder(scale, files)
        p_tau, aic = None, None
        if len(taus) >= 3:
            e = np.array(list(taus)); tv = np.array([taus[k] for k in taus])
            sl, ic = np.polyfit(np.log(e), np.log(tv), 1)
            pred = np.exp(ic) * e ** sl
            r2 = 1 - np.sum((tv - pred) ** 2) / np.sum((tv - tv.mean()) ** 2)
            p_tau = float(-sl)
            # clock rescope: tau = lamS/eta (pure, k=1) vs tau0 + lamS/eta (k=2)
            X = 1.0 / e
            lam_pure = float(np.sum(X * tv) / np.sum(X * X))
            rss_p = float(np.sum((tv - lam_pure * X) ** 2))
            A = np.vstack([np.ones_like(X), X]).T
            th, *_ = np.linalg.lstsq(A, tv, rcond=None)
            rss_a = float(np.sum((tv - A @ th) ** 2))
            n = len(tv)
            aic = {"pure_S": n * np.log(rss_p / n) + 2,
                   "affine": n * np.log(rss_a / n) + 4,
                   "tau0": float(th[0]), "lam_pure": lam_pure,
                   "lam_affine": float(th[1])}
            print(f"  tau ladder ({len(taus)} rungs): p_tau={p_tau:.2f} "
                  f"(r2={r2:.2f}); AIC pure={aic['pure_S']:.1f} "
                  f"affine={aic['affine']:.1f} tau0={th[0]:.0f}")
        report[scale] = dict(floors={r: per_rung[r] for r in rungs},
                             monotone=mono, p=p, p_ci=[lo, hi],
                             per_seed_p=per_seed_p, taus=taus, p_tau=p_tau,
                             aic=aic)
        pooled[scale] = (p, lo, hi)

    print("\n== verdicts (scaleladder_prereg.json) ==")
    if "m" in pooled and "l" in pooled:
        pm, lm, hm = pooled["m"]
        pl, ll, hl = pooled["l"]
        wm, wl = (hm - lm) / 2, (hl - ll) / 2
        pooled_hw = float(np.sqrt((wm ** 2 + wl ** 2) / 2))
        dp = pl - pm
        primary = dp > 0 and dp > 2 * pooled_hw
        disjoint = ll > hm or lm > hl
        bonus = (hm < 1 < ll) or (hl < 1 < lm)
        print(f"  dp(m->l) = {dp:+.3f}; 2x pooled half-width = "
              f"{2*pooled_hw:.3f} -> PRIMARY emergence: "
              f"{'FIRES' if primary else 'no'}")
        print(f"  CIs m[{lm:.3f},{hm:.3f}] l[{ll:.3f},{hl:.3f}] "
              f"disjoint={disjoint}; BONUS straddle-1: "
              f"{'FIRES' if bonus else 'no'}")
        gate = disjoint or pl >= 0.9
        print(f"  STAGING GATE (extend ml+xl): {'OPEN' if gate else 'CLOSED'}"
              f"  (disjoint={disjoint} or p_l={pl:.3f}>=0.9)")
        if ll > 1:
            harden = "fire (p>1)"
        elif hl <= 1:
            harden = "pool (p<=1)"
        else:
            harden = "NOT SCOREABLE (CI straddles 1)"
        print(f"  GATE-HARDENING COMMIT at l (before target residuals): "
              f"{harden}")
        cpl = report["l"].get("p_tau")
        if ll > 1 and cpl is not None:
            print(f"  coupling criterion: CI_lower(p_l)>1 requires "
                  f"p_tau>0.5 -> p_tau={cpl:.2f}: "
                  f"{'OK' if cpl > 0.5 else 'FALSIFIES single-edge account'}")
        report["verdicts"] = dict(dp=dp, pooled_hw=pooled_hw,
                                  primary=bool(primary), bonus=bool(bonus),
                                  staging_gate=bool(gate), harden=harden)
    out = os.path.join(ROOT, "results", "FLOOR2_REPORT.json")
    json.dump(report, open(out, "w"), indent=2, default=float)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
