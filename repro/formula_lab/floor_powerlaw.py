#!/usr/bin/env python3
"""Is the equilibrium noise floor a power law in eta?

The eta-weighted drops (eta_k/eta_peak)^delta with delta~0.5 imply
F_eq(eta) ~ eta^(1+delta) ~ eta^1.5, i.e. chi = dF/deta ~ eta^0.5.
Independent test: the wsdcon probes settle at floor(eta) for
eta in {3,9,18}e-5; fit log floor ~ p log eta.  Also test the real ~10M
transformer constant-stage finals if available (5 LRs, 16x range).
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from reproduce_cosine_to_wsd import SCALES  # noqa: E402
from nonadiabatic_theory import estimate_dLeq_deta  # noqa: E402


def main():
    print("== public wsdcon probes: floor(eta) power-law exponent p ==")
    for s in SCALES:
        _, etas, floors = estimate_dLeq_deta(s)
        ok = floors > 0
        if ok.sum() >= 2:
            p = np.polyfit(np.log(etas[ok]), np.log(floors[ok]), 1)[0]
        else:
            p = float("nan")
        print(f"  {s:>4}M floors={np.array2string(floors, precision=5)} "
              f"etas={etas}  p={p:.3f}")

    # real ~10M transformer: wsdcon_{5,10,20,40,80} stage-2 finals
    cdir = REPO.parent / "represent" / "results" / "curves"
    files = sorted(cdir.glob("m_wsdcon_*.csv"))
    if files:
        print("\n== real ~10M transformer: stage-2 final eval losses ==")
        etas, finals = [], []
        for f in files:
            raw = np.genfromtxt(f, delimiter=",", skip_header=1)
            tag = int(f.stem.split("_")[-1])
            etas.append(tag * 1e-5)
            finals.append(float(np.mean(raw[-5:, 3])))  # eval_loss tail
        etas = np.array(etas); finals = np.array(finals)
        order = np.argsort(etas)
        etas, finals = etas[order], finals[order]
        for e, fl in zip(etas, finals):
            print(f"  eta2={e:.1e} final={fl:.4f}")
        # floor = final - backbone; backbone unknown -> fit L_inf + C eta^p
        from scipy.optimize import least_squares

        def resid(theta):
            L_inf, logC, p = theta
            return finals - (L_inf + np.exp(logC) * etas ** p)

        best = None
        for p0 in [0.5, 1.0, 1.5, 2.0]:
            r = least_squares(resid, x0=[finals.min() - 0.05, np.log(1.0), p0],
                              bounds=([0, -20, 0.1], [finals.min(), 20, 3.0]))
            if best is None or r.cost < best.cost:
                best = r
        L_inf, logC, p = best.x
        print(f"  fit: final = {L_inf:.4f} + {np.exp(logC):.3g} * eta^{p:.3f}")
        print(f"  -> real-model floor power-law exponent p = {p:.3f}")


if __name__ == "__main__":
    main()
