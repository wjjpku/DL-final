#!/usr/bin/env python3
"""Hunt 2:
A) matched-amplitude (LOS-calibrated) residual shape on sharp curves;
   which extra pole does the remaining residual prefer (lam grid down to 0.5)?
B) rising kernel K(s)=exp(-lam_s s)-exp(-lam_f s) (delayed onset, NOT a
   positive mixture -- never tested) in probes-only / dilution / LOS / T1.
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
from formula_lab.lab import feature, fit_origin, DECAY, PROBES  # noqa: E402


def rise_feature(curve, delta, lam_s, lam_f):
    base = {"form": "pow", "delta": delta} if delta > 0 else {"form": "lr"}
    fs = feature(curve, {**base, "lam": lam_s})
    ff = feature(curve, {**base, "lam": lam_f})
    return fs - ff


def proto_eval(featfn):
    """Return (probes, dilut, LOS) mean dMAE% / wins for a feature function."""
    out = {}
    for proto in ["probes", "dilut", "LOS"]:
        deltas, wins = [], 0
        for scale in SCALES:
            p = MPL_PRECOMPUTED_INIT[scale]
            if proto == "probes":
                cal_sets = {n: PROBES for n in DECAY}
            elif proto == "dilut":
                cal_sets = {n: [m for m in DECAY if m != n] + PROBES
                            for n in DECAY}
            else:
                cal_sets = {n: [m for m in DECAY if m != n] for n in DECAY}
            for held in DECAY:
                xs, ys = [], []
                for n in cal_sets[held]:
                    c = load_curve(scale, n)
                    xs.append(featfn(c))
                    ys.append(c.loss - mpl_predict(p, c))
                kappa = max(0.0, fit_origin(np.concatenate(xs),
                                            np.concatenate(ys))[0])
                cu = load_curve(scale, held)
                bp = mpl_predict(p, cu)
                m0 = metrics(cu.loss, bp)["mae"]
                m1 = metrics(cu.loss, bp + kappa * featfn(cu))["mae"]
                deltas.append(100.0 * (m1 / m0 - 1.0))
                wins += int(m1 < m0)
        out[proto] = (float(np.mean(deltas)), wins)
    return out


def main():
    # ---------- A: matched-amplitude residual shape ----------
    print("== residual after LOS-calibrated (matched) pow d=0.25 ==")
    sp = {"form": "pow", "delta": 0.25, "lam": 10}
    for scale in SCALES:
        p = MPL_PRECOMPUTED_INIT[scale]
        for held in DECAY:
            cal = [n for n in DECAY if n != held][0]
            cc = load_curve(scale, cal)
            kappa = max(0.0, fit_origin(feature(cc, sp),
                                        cc.loss - mpl_predict(p, cc))[0])
            cu = load_curve(scale, held)
            r = cu.loss - mpl_predict(p, cu) - kappa * feature(cu, sp)
            st = np.asarray(cu.step)
            zones = [(20000, 21000), (21000, 22500), (22500, 24001)]
            zr = [float(np.mean(r[(st >= a) & (st < b)])) for a, b in zones]
            mwin = st >= 20000
            rw = r[mwin]
            best = (np.nan, -np.inf, 0.0)
            for lam2 in [0.5, 1, 2, 5, 10, 20, 50, 100]:
                f2 = feature(cu, {"form": "lr", "lam": lam2})[mwin]
                k2, r22 = fit_origin(f2, rw)
                if np.isfinite(r22) and r22 > best[1]:
                    best = (lam2, r22, k2)
            print(f"  {scale:>4} {held:20s} zones={zr[0]:+.4f}/{zr[1]:+.4f}/"
                  f"{zr[2]:+.4f} bestPole lam={best[0]} R2={best[1]:.3f} "
                  f"k={best[2]:+.4f}")

    # ---------- B: rising kernel ----------
    print("\n== rising kernel exp(-lam_s)-exp(-lam_f), pow d=0.25 base ==")
    print(f"{'variant':28s} {'probes':>10s} {'dilut':>10s} {'LOS':>10s}")
    base = proto_eval(lambda c: feature(c, sp))
    print(f"{'monotone d=.25@10 [ship]':28s} "
          f"{base['probes'][0]:+7.1f}/{base['probes'][1]} "
          f"{base['dilut'][0]:+7.1f}/{base['dilut'][1]} "
          f"{base['LOS'][0]:+7.1f}/{base['LOS'][1]}")
    for lam_s, lam_f in [(10, 30), (10, 50), (10, 100), (5, 30), (7, 50),
                         (10, 200)]:
        r = proto_eval(lambda c, a=lam_s, b=lam_f: rise_feature(c, 0.25, a, b))
        print(f"rise d=.25 ls={lam_s} lf={lam_f:<8} "
              f"{r['probes'][0]:+7.1f}/{r['probes'][1]} "
              f"{r['dilut'][0]:+7.1f}/{r['dilut'][1]} "
              f"{r['LOS'][0]:+7.1f}/{r['LOS'][1]}")
    # mixture: monotone + small rising piece is 2 kappas -> skip (capacity).

if __name__ == "__main__":
    main()
