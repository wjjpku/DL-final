#!/usr/bin/env python3
"""Quick residual-structure hunt:
A) per-scale spec variants (integral floor-gap w/ measured p; shrunk delta) on
   probes-only, dilution, LOS, T1(mpl-B) protocols.
B) residual-of-the-corrected-law shape analysis on sharp curves.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from reproduce_cosine_to_wsd import (  # noqa: E402
    load_curve, mpl_predict, metrics, MPL_PRECOMPUTED_INIT, SCALES,
)
from formula_lab import lab  # noqa: E402
from formula_lab.lab import (  # noqa: E402
    feature, fit_origin, DECAY, PROBES, probe_floor_powerlaw,
)

# measured per-scale floor exponents (from probes, leakage-clean)
P_MEAS = {s: probe_floor_powerlaw(s)[1] for s in SCALES}
print("measured p per scale:", {s: round(p, 3) for s, p in P_MEAS.items()})


def spec_for(scale, variant):
    p = P_MEAS[scale]
    if variant == "lr":
        return {"form": "lr", "lam": 10}
    if variant == "pow25":
        return {"form": "pow", "delta": 0.25, "lam": 10}
    if variant == "pow50":
        return {"form": "pow", "delta": 0.5, "lam": 10}
    if variant == "pow_meas":
        return {"form": "pow", "delta": max(p - 1.0, 0.0), "lam": 10}
    if variant == "floor_meas":
        return {"form": "floor", "p": p, "lam": 10}
    if variant == "floor_meas_clip":  # p clipped to >=1 (superlinear only)
        return {"form": "floor", "p": max(p, 1.0), "lam": 10}
    if variant == "shrunk":  # combine default 1/4 with measured (p-1)+
        return {"form": "pow", "delta": 0.5 * (0.25 + max(p - 1.0, 0.0)), "lam": 10}
    if variant == "maxcomb":  # max(1/4, (p-1)+)
        return {"form": "pow", "delta": max(0.25, max(p - 1.0, 0.0)), "lam": 10}
    raise ValueError(variant)


def probes_only_v(variant):
    deltas, wins = [], 0
    for scale in SCALES:
        sp = spec_for(scale, variant)
        p = MPL_PRECOMPUTED_INIT[scale]
        xs, ys = [], []
        for n in PROBES:
            c = load_curve(scale, n)
            xs.append(feature(c, sp))
            ys.append(c.loss - mpl_predict(p, c))
        kappa = max(0.0, fit_origin(np.concatenate(xs), np.concatenate(ys))[0])
        for n in DECAY:
            cu = load_curve(scale, n)
            bp = mpl_predict(p, cu)
            m0 = metrics(cu.loss, bp)["mae"]
            m1 = metrics(cu.loss, bp + kappa * feature(cu, sp))["mae"]
            deltas.append(100.0 * (m1 / m0 - 1.0))
            wins += int(m1 < m0)
    return float(np.mean(deltas)), wins


def dilution_v(variant):
    """kappa from [other sharp + 3 probes] -> held-out sharp."""
    deltas, wins = [], 0
    for scale in SCALES:
        sp = spec_for(scale, variant)
        p = MPL_PRECOMPUTED_INIT[scale]
        for held in DECAY:
            cal = [n for n in DECAY if n != held] + PROBES
            xs, ys = [], []
            for n in cal:
                c = load_curve(scale, n)
                xs.append(feature(c, sp))
                ys.append(c.loss - mpl_predict(p, c))
            kappa = max(0.0, fit_origin(np.concatenate(xs),
                                        np.concatenate(ys))[0])
            cu = load_curve(scale, held)
            bp = mpl_predict(p, cu)
            m0 = metrics(cu.loss, bp)["mae"]
            m1 = metrics(cu.loss, bp + kappa * feature(cu, sp))["mae"]
            deltas.append(100.0 * (m1 / m0 - 1.0))
            wins += int(m1 < m0)
    return float(np.mean(deltas)), wins


def los_v(variant):
    deltas, wins = [], 0
    for scale in SCALES:
        sp = spec_for(scale, variant)
        p = MPL_PRECOMPUTED_INIT[scale]
        for held in DECAY:
            cal = [n for n in DECAY if n != held][0]
            cc = load_curve(scale, cal)
            kappa = max(0.0, fit_origin(feature(cc, sp),
                                        cc.loss - mpl_predict(p, cc))[0])
            cu = load_curve(scale, held)
            bp = mpl_predict(p, cu)
            m0 = metrics(cu.loss, bp)["mae"]
            m1 = metrics(cu.loss, bp + kappa * feature(cu, sp))["mae"]
            deltas.append(100.0 * (m1 / m0 - 1.0))
            wins += int(m1 < m0)
    return float(np.mean(deltas)), wins


def t1_v(variant, chain="mpl-B"):
    """Table-1 chain with per-scale specs (generalizes lab.table1_protocol)."""
    specs = {s: spec_for(s, variant) for s in SCALES}
    ratio = {s: lab.kappa_fit_sharp(s, specs[s]) /
             lab.kappa_pred(s, specs[s], chain) for s in SCALES}
    deltas, wins = [], 0
    for tgt in SCALES:
        c_loo = float(np.mean([ratio[s] for s in SCALES if s != tgt]))
        kappa = c_loo * lab.kappa_pred(tgt, specs[tgt], chain)
        p = MPL_PRECOMPUTED_INIT[tgt]
        for n in DECAY:
            cu = load_curve(tgt, n)
            bp = mpl_predict(p, cu)
            m0 = metrics(cu.loss, bp)["mae"]
            m1 = metrics(cu.loss, bp + kappa * feature(cu, specs[tgt]))["mae"]
            deltas.append(100.0 * (m1 / m0 - 1.0))
            wins += int(m1 < m0)
    return float(np.mean(deltas)), wins


def main():
    variants = ["lr", "pow25", "pow50", "pow_meas", "floor_meas",
                "floor_meas_clip", "shrunk", "maxcomb"]
    print(f"\n{'variant':16s} {'probes':>10s} {'dilut':>10s} {'LOS':>10s} "
          f"{'T1-B':>10s}")
    for v in variants:
        po, pw = probes_only_v(v)
        di, dw = dilution_v(v)
        lo, lw = los_v(v)
        t1, tw = t1_v(v)
        print(f"{v:16s} {po:+7.1f}/{pw} {di:+7.1f}/{dw} {lo:+7.1f}/{lw} "
              f"{t1:+7.1f}/{tw}")

    # ---------- B: residual-of-corrected-law shape ----------
    print("\n== residual after probes-calibrated pow d=0.25 correction ==")
    for scale in SCALES:
        sp = spec_for(scale, "pow25")
        p = MPL_PRECOMPUTED_INIT[scale]
        xs, ys = [], []
        for n in PROBES:
            c = load_curve(scale, n)
            xs.append(feature(c, sp))
            ys.append(c.loss - mpl_predict(p, c))
        kappa = max(0.0, fit_origin(np.concatenate(xs), np.concatenate(ys))[0])
        for n in DECAY:
            cu = load_curve(scale, n)
            r = cu.loss - mpl_predict(p, cu) - kappa * feature(cu, sp)
            st = np.asarray(cu.step)
            # decay window is steps 20000-24000
            zones = [(20000, 21000), (21000, 22500), (22500, 24001)]
            zr = [float(np.mean(r[(st >= a) & (st < b)])) for a, b in zones]
            pre = float(np.mean(r[st < 20000]))
            # remaining-residual R2 vs a 2nd fast pole and vs unweighted feat
            best_fast = (np.nan, 0.0)
            mwin = st >= 20000
            rw = r[mwin]
            ssw = float(np.sum((rw - rw.mean()) ** 2))
            for lam2 in [20, 30, 50, 100, 200]:
                f2 = feature(cu, {"form": "lr", "lam": lam2})[mwin]
                k2, r22 = fit_origin(f2, rw)
                if np.isfinite(r22) and r22 > best_fast[1]:
                    best_fast = (lam2, r22)
            print(f"  {scale:>4} {n:20s} pre={pre:+.4f} "
                  f"zones={zr[0]:+.4f}/{zr[1]:+.4f}/{zr[2]:+.4f} "
                  f"bestFastPole lam={best_fast[0]} R2={best_fast[1]:.3f}")


if __name__ == "__main__":
    main()
