#!/usr/bin/env python3
"""E3 pre-launch predictions (committed before GPU per G4).

Concentration-graded drop: 1.5e-3 -> 4e-4 ramped linearly over
k in {1, 50, 200, 800} steps from step 3000, then held so every arm ends
stage 2 at the same cumulative S2 = 1.2 (equal-S measurement point).

Two committed prediction families for the paired equal-S final-excess
differences D(k) = excess(k) - excess(k=1):
  LINEAR : the shipped one-pole S-clock kernel -- deposits (eta_{t-1} -
           eta_t)+ decay as exp(-lam_slow * dS); spreading the drop only
           re-times deposits.
  NONLIN r*: excess relaxes as d(ex)/dS = -lam_slow * ex^{r*} / A0^{r*-1}
           (normalized so the k=1 instant arm matches the linear arm's
           deposit scale A0; zero new DOF beyond r*).
Verdict (committed): GLS chi2 of measured D(k) (seed-paired, tail means
over the last 25% of the hold) vs LINEAR and vs best r* on the grid
{1.25, 1.5, 2, 3}; delta-chi2 >= 6 separates; interior-CI r* = identified.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
ETA1, ETA2, LAM, S2 = 1.5e-3, 4e-4, 10.0, 1.2
KS = [1, 50, 200, 800]
KAPPA = 0.0027610792134732053   # shipped d=0 closure kappa, m bed (ARMS)


def schedule(k):
    ramp = np.linspace(ETA1, ETA2, k + 1)[1:]
    s_ramp = float(ramp.sum())
    hold = int(round((S2 - s_ramp) / ETA2))
    return np.concatenate([ramp, np.full(hold, ETA2)])


def linear_excess(etas):
    acc = 0.0
    prev = ETA1
    for e in etas:
        acc *= np.exp(-LAM * e)
        acc += max(prev - e, 0.0)
        prev = e
    return KAPPA * acc / ETA1


def nonlin_excess(etas, rstar):
    """dE/dS = -LAM * E^r / A0^{r-1}, deposits as in linear; A0 = total
    drop scale so k=1 matches the linear normalization at deposit time."""
    A0 = ETA1 - ETA2
    E, prev = 0.0, ETA1
    for e in etas:
        E = E - LAM * (E ** rstar) / (A0 ** (rstar - 1)) * e if E > 0 else 0.0
        E = max(E, 0.0)
        E += max(prev - e, 0.0)
        prev = e
    return KAPPA * E / ETA1


def tail_window_mean(fn, etas):
    """Average the excess over the last 25% of the hold (the measurement
    window), stepping the accumulator to each window position."""
    n = len(etas)
    w0 = n - max(int(0.25 * (n - np.argmax(etas <= ETA2 + 1e-12))), 50)
    vals = []
    for cut in range(w0, n, max((n - w0) // 40, 1)):
        vals.append(fn(etas[:cut]))
    return float(np.mean(vals))


def main():
    out = {"committed_before_launch": True, "kappa_source": "shipped d=0 m",
           "ks": KS, "linear": {}, "nonlinear": {}}
    es = {k: schedule(k) for k in KS}
    print("arm lengths:", {k: len(es[k]) for k in KS},
          " S2 check:", {k: round(float(es[k].sum()), 4) for k in KS})
    lin = {k: tail_window_mean(linear_excess, es[k]) for k in KS}
    out["linear"] = {str(k): lin[k] for k in KS}
    print("LINEAR excess e-3:", {k: round(v * 1e3, 3) for k, v in lin.items()})
    print("LINEAR D(k) vs k=1 e-3:",
          {k: round((lin[k] - lin[1]) * 1e3, 3) for k in KS})
    for r in [1.25, 1.5, 2.0, 3.0]:
        nl = {k: tail_window_mean(lambda e: nonlin_excess(e, r), es[k])
              for k in KS}
        out["nonlinear"][str(r)] = {str(k): nl[k] for k in KS}
        print(f"NONLIN r*={r}: D(k) e-3:",
              {k: round((nl[k] - nl[1]) * 1e3, 3) for k in KS})
    op = os.path.join(REPO, "results", "formula_lab", "e3_predictions.json")
    json.dump(out, open(op, "w"), indent=1)
    print("wrote", op)


if __name__ == "__main__":
    main()
