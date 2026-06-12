"""NextGen target-identifiability gate on the gen schedules:
R_target = ||M_G phi||^2 / ||phi||^2 with G = span{1, DCT_1..4} (paper's
schedule-agnostic nuisance basis).  Gate: abstain if R_target < 0.01."""
import os, sys, math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_curves as AC
from analyze_gen import load_gen, wfeature


def dct_basis(n, modes):
    idx = np.arange(n, dtype=float)
    cols = [np.ones(n)]
    for k in range(1, modes + 1):
        cols.append(np.cos(math.pi * (idx + 0.5) * k / max(n, 1)))
    z = np.column_stack(cols)
    return z / np.maximum(np.linalg.norm(z, axis=0), 1e-12)


def retention(phi, modes=4):
    z = dct_basis(len(phi), modes)
    coef, *_ = np.linalg.lstsq(z, phi, rcond=None)
    perp = phi - z @ coef
    return float(np.dot(perp, perp) / max(np.dot(phi, phi), 1e-18))


def main():
    cv = AC.load_scale("m")
    for k, v in load_gen().items():
        cv[k] = v
    for spec, tag in [({"form": "lr"}, "lr"),
                      ({"form": "pow", "delta": 0.5}, "pow0.5")]:
        print(f"== feature: {tag}, lam=1 ==")
        for name in ["invsqrt", "sharp600", "twodrop", "cyclic", "wsd",
                     "wsdld", "wsdcon_40", "cosine"]:
            if name not in cv:
                continue
            phi = wfeature(cv[name]["lr"], spec, 1.0)[cv[name]["step"]]
            r = retention(phi)
            print(f"  {name:10s} R_target={r:7.4f}  gate(>=0.01): "
                  f"{'PASS' if r >= 0.01 else 'ABSTAIN'}")


if __name__ == "__main__":
    main()
