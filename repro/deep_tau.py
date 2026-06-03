#!/usr/bin/env python3
"""Directly measure the loss-relaxation time tau from the wsdcon post-step transient.

Non-adiabatic theory: after an LR change the residual ΔL = L - L_eq decays
exponentially with rate r = 1/tau. The two-stage wsdcon curves step the LR down at
step 8000 from eta_peak=3e-4 to lr_b in {3,9,18}e-5 and then hold it constant for
~8000 steps -- a clean exponential-relaxation window.

We fit, on the cosine-fit-MPL residual r(t) for t>8000,
    r(t) = floor + amp * exp(-(step-8000)/tau),
and read off tau for each (scale, lr_b). Tests:
  * tau magnitude vs the tau~1200 preferred by the few-shot fit (independent check);
  * tau vs lr_b: r ~ eta * lambda  =>  tau ~ 1/eta  (step-time relaxation rate that
    scales with the LR) would give tau ∝ 1/lr_b; a constant tau means step-time.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from reproduce_cosine_to_wsd import (  # noqa: E402
    load_curve, mpl_predict, MPL_PRECOMPUTED_INIT, SCALES,
)

STEP_DOWN = 8000
WSDCON = [("wsdcon_3.csv", 3e-5), ("wsdcon_9.csv", 9e-5), ("wsdcon_18.csv", 18e-5)]


def relax(t, floor, amp, tau):
    return floor + amp * np.exp(-(t - STEP_DOWN) / tau)


def main():
    print("=" * 76)
    print("Direct measurement of loss-relaxation time tau (wsdcon post-step transient)")
    print("=" * 76)
    rows = {}
    for scale in SCALES:
        p = MPL_PRECOMPUTED_INIT[scale]
        print(f"\n[{scale}M]   {'curve':14s} {'lr_b':>8s} {'tau(steps)':>11s} "
              f"{'amp':>8s} {'tau*lr_b':>9s}")
        for name, lrb in WSDCON:
            c = load_curve(scale, name)
            r = c.loss - mpl_predict(p, c)
            m = c.step > STEP_DOWN + 50           # skip the 1-2 step jump itself
            t, y = c.step[m].astype(float), r[m]
            try:
                popt, _ = curve_fit(relax, t, y,
                                    p0=[0.0, max(y.max(), 1e-3), 1500.0],
                                    bounds=([-0.05, 0.0, 50.0], [0.05, 1.0, 20000.0]),
                                    maxfev=20000)
                floor, amp, tau = popt
                rows.setdefault(scale, []).append((lrb, tau))
                print(f"        {name:14s} {lrb:8.1e} {tau:11.0f} {amp:8.4f} "
                      f"{tau*lrb:9.4f}")
            except Exception as e:
                print(f"        {name:14s} fit failed: {e}")

    # tau vs lr_b scaling, per scale:  log tau = a*log(lr_b) + b ; slope -1 => tau~1/eta
    print("\n" + "=" * 76)
    print("tau vs lr_b scaling (slope of log tau vs log lr_b; -1 => tau∝1/eta = S-time)")
    print("=" * 76)
    for scale in SCALES:
        if scale not in rows or len(rows[scale]) < 2:
            continue
        lrbs = np.array([x[0] for x in rows[scale]])
        taus = np.array([x[1] for x in rows[scale]])
        slope = np.polyfit(np.log(lrbs), np.log(taus), 1)[0]
        prod = taus * lrbs
        print(f"   {scale:>4s}M  tau={taus.astype(int)}  slope={slope:+.2f}  "
              f"tau*lr_b={np.round(prod,3)} (const? CV={np.std(prod)/np.mean(prod)*100:.0f}%)")
    print("\nInterpretation: slope≈-1 and tau*lr_b≈const => relaxation rate r∝eta")
    print("(loss relaxes in cumulative-LR / S-time); slope≈0 => constant step-time tau.")


if __name__ == "__main__":
    main()
