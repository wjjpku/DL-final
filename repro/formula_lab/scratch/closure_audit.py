#!/usr/bin/env python3
"""Closure audit: cheap decisive tests attacking three 'closed' verdicts.

T1  C7 two-channel: memoryless superlinear floor surplus g(eta_t) + UNWEIGHTED
    one-pole lag, both amplitudes fit on the 3 wsdcon probes only (frozen MPL),
    evaluated on held-out wsd+wsdld.  Baselines: d=0 -17.1%, d=0.5 -28.6%.
T2  Nonlinear ODE LOO transfer with the dimensionless invariant rho = r*/kappa
    transferred instead of raw r* (kappa then solved self-consistently on the
    target's probes).  Baseline: raw-r* LOO -23.1%, shipped d=0.5 -28.6%.
T3  Two-pole kernel with rates PINNED at the independently measured family
    values (probe ~15, sharp ~1-5), only mixture weight scanned; kappa from
    probes -> sharp.  Oracle grid: if no cell beats the matched one-pole
    optimum, the kernel-shape closure survives a non-joint-SSE protocol.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from reproduce_cosine_to_wsd import (  # noqa: E402
    load_curve, mpl_predict, metrics, MPL_PRECOMPUTED_INIT, SCALES, PEAK_LR,
)
from formula_lab.lab import (  # noqa: E402
    feature, fit_origin, probe_floor_powerlaw, DECAY, PROBES,
)
from formula_lab.kernels import exp2_feature  # noqa: E402
from formula_lab.nonlinear_ode import fit_ode, ode_response  # noqa: E402


# --------------------------------------------------------------- T1: C7
def memoryless_feature(curve, p, surplus=True):
    x = np.maximum(curve.lrs.astype(np.float64) / PEAK_LR, 1e-12)
    g = np.power(x, p) - (x if surplus else 0.0)
    return g[np.asarray(curve.step, dtype=np.int64)]


def t1_c7():
    print("=" * 78)
    print("T1  C7 two-channel (memoryless floor surplus + unweighted lag@10)")
    print("    probes-only -> sharp; baselines d=0 -17.1%, d=0.5 -28.6%")
    print("=" * 78)
    spec = {"form": "lr", "lam": 10.0}
    for surplus in [True, False]:
        tag = "g=x^p - x" if surplus else "g=x^p"
        deltas, wins = [], 0
        diag = []
        for scale in SCALES:
            pmpl = MPL_PRECOMPUTED_INIT[scale]
            _, p = probe_floor_powerlaw(scale)  # probes-only, leakage-clean
            xs1, xs2, ys = [], [], []
            for n in PROBES:
                c = load_curve(scale, n)
                xs1.append(feature(c, spec))
                xs2.append(memoryless_feature(c, p, surplus))
                ys.append(c.loss - mpl_predict(pmpl, c))
            X = np.stack([np.concatenate(xs1), np.concatenate(xs2)], axis=1)
            y = np.concatenate(ys)
            k, *_ = np.linalg.lstsq(X, y, rcond=None)
            k1, k2 = float(k[0]), float(k[1])
            if k1 < 0:  # lag amplitude must be nonnegative
                k1 = 0.0
                k2 = float(np.dot(X[:, 1], y) / max(np.dot(X[:, 1], X[:, 1]), 1e-18))
            # single-channel reference kappa and in-sample sharp kappa
            k_single = max(0.0, fit_origin(X[:, 0], y)[0])
            xs_sh, ys_sh = [], []
            for n in DECAY:
                c = load_curve(scale, n)
                xs_sh.append(feature(c, spec))
                ys_sh.append(c.loss - mpl_predict(pmpl, c))
            k_sharp = fit_origin(np.concatenate(xs_sh), np.concatenate(ys_sh))[0]
            diag.append(f"  {scale:>4}M p={p:.2f} k1={k1:.4f} k2={k2:+.4f} "
                        f"(single-ch probe k={k_single:.4f}, sharp k={k_sharp:.4f})")
            for n in DECAY:
                c = load_curve(scale, n)
                base = mpl_predict(pmpl, c)
                pred = base + k1 * feature(c, spec) + k2 * memoryless_feature(c, p, surplus)
                m0 = metrics(c.loss, base)["mae"]
                m1 = metrics(c.loss, pred)["mae"]
                deltas.append(100.0 * (m1 / m0 - 1.0))
                wins += int(m1 < m0)
        print(f"[{tag:10s}] probes->sharp: {np.mean(deltas):+6.1f}%  {wins}/6")
        for d in diag:
            print(d)


# ------------------------------------------------- T2: ODE r*/kappa rescale
def t2_ode():
    print()
    print("=" * 78)
    print("T2  nonlinear ODE LOO: raw r* vs dimensionless rho=r*/kappa transfer")
    print("    baselines: raw-r* LOO -23.1%, linear -17.1%, shipped d=0.5 -28.6%")
    print("=" * 78)
    fam = DECAY + PROBES
    shape = {}
    for scale in SCALES:
        th, _ = fit_ode(scale, fam, linear=False)
        shape[scale] = th
        print(f"  {scale:>4}M joint: lam0={th[0]:7.2f} r*={th[1]:.4f} "
              f"kappa={th[2]:.4f}  rho=r*/kappa={th[1]/th[2]:.3f}")
    lr = np.log([shape[s][1] for s in SCALES])
    lrho = np.log([shape[s][1] / shape[s][2] for s in SCALES])
    print(f"  cross-scale spread (std of log): raw r* {np.std(lr):.3f} "
          f"vs rho {np.std(lrho):.3f}")

    for mode in ["raw", "rho"]:
        m0s, m1s, wins = [], [], 0
        for tgt in SCALES:
            others = [s for s in SCALES if s != tgt]
            lam0 = float(np.exp(np.mean([np.log(shape[s][0]) for s in others])))
            if mode == "raw":
                rstar_of = lambda k: float(np.exp(np.mean(
                    [np.log(shape[s][1]) for s in others])))
            else:
                rho = float(np.exp(np.mean(
                    [np.log(shape[s][1] / shape[s][2]) for s in others])))
                rstar_of = lambda k: rho * k
            pmpl = MPL_PRECOMPUTED_INIT[tgt]
            resids = []
            for n in PROBES:
                c = load_curve(tgt, n)
                resids.append((c, c.loss - mpl_predict(pmpl, c)))

            def sse(logk):
                k = float(np.exp(logk))
                rs = rstar_of(k)
                tot = 0.0
                for c, r in resids:
                    tot += float(np.sum((r - ode_response(c, lam0, rs, k)) ** 2))
                return tot

            res = minimize_scalar(sse, bounds=(np.log(1e-4), np.log(1.0)),
                                  method="bounded", options={"xatol": 1e-4})
            kap = float(np.exp(res.x))
            rs = rstar_of(kap)
            row = []
            for n in DECAY:
                c = load_curve(tgt, n)
                base = mpl_predict(pmpl, c)
                pred = base + ode_response(c, lam0, rs, kap)
                m0 = metrics(c.loss, base)["mae"]
                m1 = metrics(c.loss, pred)["mae"]
                m0s.append(m0); m1s.append(m1); wins += int(m1 < m0)
                row.append(f"{n.split('_')[0]} {100 * (m1 / m0 - 1):+.1f}%")
            print(f"  [{mode}] {tgt:>4}M lam0={lam0:5.2f} r*={rs:.4f} "
                  f"kappa={kap:.4f}: " + "  ".join(row))
        d = 100.0 * (np.mean(m1s) / np.mean(m0s) - 1.0)
        print(f"  [{mode}] OVERALL: {d:+.1f}% {wins}/6")


# ------------------------------------------------ T3: pinned two-pole grid
def t3_twopole():
    print()
    print("=" * 78)
    print("T3  two-pole kernel, rates pinned at family-measured values;")
    print("    probes->sharp (oracle grid).  One-pole refs: lam=10 d=0 -17.1%,")
    print("    d=0.5 -28.6%, (delta,lam) grid best -34.5%")
    print("=" * 78)
    rows = []
    for d in [0.0, 0.25, 0.5]:
        for lam_f in [10.0, 15.0, 19.0]:
            for lam_s in [1.0, 2.0, 5.0]:
                for w in [0.3, 0.5, 0.7, 0.85]:
                    deltas, wins = [], 0
                    for scale in SCALES:
                        pmpl = MPL_PRECOMPUTED_INIT[scale]
                        xs, ys = [], []
                        for n in PROBES:
                            c = load_curve(scale, n)
                            xs.append(exp2_feature(c, lam_f, lam_s, w,
                                                   eta_weight_delta=d))
                            ys.append(c.loss - mpl_predict(pmpl, c))
                        kap = max(0.0, fit_origin(np.concatenate(xs),
                                                  np.concatenate(ys))[0])
                        for n in DECAY:
                            c = load_curve(scale, n)
                            base = mpl_predict(pmpl, c)
                            f = exp2_feature(c, lam_f, lam_s, w,
                                             eta_weight_delta=d)
                            m0 = metrics(c.loss, base)["mae"]
                            m1 = metrics(c.loss, base + kap * f)["mae"]
                            deltas.append(100.0 * (m1 / m0 - 1.0))
                            wins += int(m1 < m0)
                    rows.append((float(np.mean(deltas)), wins, d, lam_f, lam_s, w))
    rows.sort()
    print("  best 8 cells:")
    for v, wins, d, lf, ls, w in rows[:8]:
        print(f"    d={d:4.2f} lam_f={lf:4.1f} lam_s={ls:3.1f} w={w:.2f}: "
              f"{v:+6.1f}% ({wins}/6)")
    n_beat = sum(1 for r in rows if r[0] < -28.6)
    print(f"  cells beating shipped d=0.5 (-28.6%): {n_beat}/{len(rows)}")
    n_beat2 = sum(1 for r in rows if r[0] < -34.5)
    print(f"  cells beating one-pole grid best (-34.5%): {n_beat2}/{len(rows)}")


if __name__ == "__main__":
    t1_c7()
    t3_twopole()
    t2_ode()
