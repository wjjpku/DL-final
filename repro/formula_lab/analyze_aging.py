#!/usr/bin/env python3
"""g3d2 T1 (zero-GPU): does an AGING two-clock floor form reconcile the
sublinear endpoint exponent (p~0.65) with the superlinear asymptote
(A3 p_Finf~1.0) across the E4 horizon ladder?

Model: F_eq(eta, H) = L0 + a*eta^zeta - b_sec*log(1 + H/(tau0*eta))
fit JOINTLY to the per-rung floors at horizons H in {3000,12000,24000}
(curves_floor_m base + _t12000 + _t24000), using the corrected design
window.  Reports zeta (asymptotic exponent), the secular term, fit R2, and
the implied finite-horizon p(H)=dlogF/dlog eta vs the measured 0.65/0.77/0.79.
ZERO GPU (re-analysis of committed E4 curves)."""
import os
import sys

import numpy as np
from scipy.optimize import curve_fit

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", ".."))
sys.path.insert(0, os.path.join(REPO, "represent", "repro"))
import analyze_curves as AC  # noqa: E402

CDIR = os.path.join(REPO, "represent", "results", "curves_floor_m")
RUNG_ETA = {"5": 0.5e-4, "10": 1e-4, "20": 2e-4, "30": 3e-4, "40": 4e-4,
            "60": 6e-4, "80": 8e-4, "150": 1.5e-3}
HORIZONS = {3000: "", 12000: "_t12000", 24000: "_t24000"}


def floor_at(rung, H, sfx):
    # base 3000 uses curves_floor (s1337) + curves_floor_m gap rungs; for the
    # horizon test we use curves_floor_m for all (consistent bed/seed 1337).
    if H == 3000:
        cand = [os.path.join(REPO, "represent", "results", "curves_floor",
                             f"floor_{rung}.csv"),
                os.path.join(CDIR, f"floor_{rung}.csv")]
    else:
        cand = [os.path.join(CDIR, f"floor_{rung}{sfx}.csv")]
    for f in cand:
        if os.path.exists(f):
            rows = np.genfromtxt(f, delimiter=",", names=True)
            step = np.atleast_1d(rows["step"]).astype(int)
            loss = np.atleast_1d(rows["eval_loss"]).astype(float)
            sm = AC.smooth_by_step(step, loss)
            eta2 = RUNG_ETA[rung]
            cut = H + 0.75 * int(round(1.2 / eta2))
            m = step >= cut
            if m.sum() >= 4:
                return float(np.mean(sm[m]))
    return None


def main():
    pts = []   # (eta, H, F)
    for H, sfx in HORIZONS.items():
        for rung, eta in RUNG_ETA.items():
            F = floor_at(rung, H, sfx)
            if F is not None:
                pts.append((eta, float(H), F))
    pts = np.array(pts)
    eta, H, F = pts[:, 0], pts[:, 1], pts[:, 2]
    print(f"loaded {len(pts)} floor points across horizons "
          f"{sorted(set(H.astype(int)))}")

    # aging model: F = L0 + a*eta^zeta - b_sec*log(1 + H/(tau0*eta))
    def aging(X, L0, a, zeta, b_sec, tau0):
        e, h = X
        return L0 + a * e ** zeta - b_sec * np.log1p(h / (tau0 * e))
    p0 = [F.min(), 30.0, 1.0, 0.01, 1e6]
    try:
        po, _ = curve_fit(aging, (eta, H), F, p0=p0, maxfev=200000,
                          bounds=([0.5, 0.1, 0.3, 0, 1e3],
                                  [1.5, 1e4, 3.0, 1.0, 1e12]))
        pred = aging((eta, H), *po)
        r2 = 1 - np.sum((F - pred) ** 2) / np.sum((F - F.mean()) ** 2)
        L0, a, zeta, b_sec, tau0 = po
        print(f"\nAGING fit: F = {L0:.4f} + {a:.3f}*eta^{zeta:.3f} "
              f"- {b_sec:.4f}*log(1+H/({tau0:.2e}*eta))   R2={r2:.4f}")
        print(f"  asymptotic exponent zeta = {zeta:.3f}  "
              f"(>1 = superlinear equilibrium floor)")
        # implied finite-horizon p(H) = dlogF_excess/dlog eta near a mid rung
        for Htest in [3000, 12000, 24000, 1e9]:
            em = np.geomspace(0.5e-4, 1.5e-3, 200)
            Fex = a * em ** zeta - b_sec * np.log1p(Htest / (tau0 * em))
            # local slope dlog(Fex-min)/dlog eta at eta=2e-4
            i = np.argmin(np.abs(em - 2e-4))
            dlogF = np.gradient(np.log(np.clip(Fex - Fex.min() + 1e-6, 1e-6,
                                               None)), np.log(em))
            tag = "inf" if Htest > 1e8 else str(int(Htest))
            print(f"  implied p(H={tag:>6}) ~ {dlogF[i]:+.2f}")
    except Exception as e:
        print("aging fit failed:", e); r2 = 0; zeta = 0

    print("\nmeasured endpoint p(H): 0.65 (3k) / 0.77 (12k) / 0.79 (24k); "
          "A3 p_Finf ~ 1.04 [0.80,1.28]")
    if r2 >= 0.9 and zeta > 1.05:
        v = ("AGING_SUPPORTED: single closed form fits all horizons (R2>=0.9) "
             "with superlinear asymptote zeta>1 -> reconciles sublinear "
             "endpoint with superlinear equilibrium; candidate formula term")
    elif r2 >= 0.9:
        v = (f"AGING_FITS_zeta={zeta:.2f}: closed form fits but asymptote not "
             f"clearly superlinear -> consistent with sublinear, no new claim")
    else:
        v = "AGING_WEAK: no clean closed-form fit across horizons"
    print("\nT1 VERDICT:", v)


if __name__ == "__main__":
    main()
