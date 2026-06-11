"""Does the paper's smooth identifiability gate switch the correction off on
the diffuse invsqrt schedule (and keep it on for sharp600/twodrop)?

w_id = sigmoid(3 log(6000/drop_eff_steps)) * sigmoid(3 log(feature_max/0.05))
     * sigmoid(3 log(total_drop/0.05)),   thresholds in eta_peak-normalized units.
"""
import os, sys, json, math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_curves as AC
from analyze_gen import load_gen, wfeature, PEAK


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))


def main():
    cv = AC.load_scale("m")
    for k, v in load_gen().items():
        cv[k] = v
    for name in ["invsqrt", "sharp600", "twodrop", "cyclic", "wsd", "wsdld",
                 "wsdcon_40", "cosine"]:
        if name not in cv:
            continue
        eta = np.asarray(cv[name]["lr"], float)
        drop = np.zeros_like(eta)
        drop[1:] = np.maximum(eta[:-1] - eta[1:], 0.0)
        dn = drop / PEAK
        total = float(dn.sum())
        l2 = float(np.dot(dn, dn))
        eff = total * total / l2 if l2 > 1e-18 else float("inf")
        phi = wfeature(eta, {"form": "lr"}, 1.0)[cv[name]["step"]]
        fmax = float(np.max(phi))
        w = (sigmoid(3 * math.log(6000 / max(eff, 1e-9)))
             * sigmoid(3 * math.log(max(fmax, 1e-12) / 0.05))
             * sigmoid(3 * math.log(max(total, 1e-12) / 0.05)))
        print(f"{name:10s} total_drop={total:6.2f} eff_steps={eff:9.0f} "
              f"feat_max={fmax:6.3f} -> w_id={w:.3f}")


if __name__ == "__main__":
    main()
