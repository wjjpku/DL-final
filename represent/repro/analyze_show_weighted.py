"""E6 never-scanned bed: weighted-closure arms on the showcase suite
(results/curves_show/, peak 2.5e-3).  MPL on [constant, cosine, wsdcon_20];
kappa+lam from probes wsdcon_{5,10,40,80}; targets wsd_sharp / wsd_grad."""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_curves as AC
from analyze_show import load

PEAK = 2.5e-3
PROBES = ["wsdcon_5", "wsdcon_10", "wsdcon_40", "wsdcon_80"]
TARGETS = ["wsd_sharp", "wsd_grad"]
LAM_GRID = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0]

ARMS = [
    ("lr (paper)", {"form": "lr"}),
    ("pow d=0.25", {"form": "pow", "delta": 0.25}),
    ("pow d=0.5", {"form": "pow", "delta": 0.5}),
    ("pow d=0.75", {"form": "pow", "delta": 0.75}),
]


def wfeature(lr, spec, lam):
    eta = np.asarray(lr, float)
    drop = np.zeros_like(eta)
    drop[1:] = np.maximum(eta[:-1] - eta[1:], 0.0)
    if spec.get("form") == "pow":
        drop = drop * np.power(np.maximum(eta / PEAK, 1e-12), spec["delta"])
    dec = np.exp(-lam * eta)
    s = np.empty_like(eta)
    acc = 0.0
    for t in range(len(eta)):
        acc = acc * dec[t] + drop[t]
        s[t] = acc
    return s / PEAK


def fit_origin(x, y):
    xx = float(np.dot(x, x))
    return 0.0 if xx <= 1e-18 else max(0.0, float(np.dot(x, y) / xx))


def main():
    cv = load()
    train = [s for s in ["constant", "cosine", "wsdcon_20"] if s in cv]
    params, fobj = AC.fit_mpl(cv, train, n_starts=8)
    print(f"MPL fit on {train}: obj={fobj:.5f}; params={np.round(params,4)}")
    resid = {s: cv[s]["loss"] - AC.mpl_pred_at(cv[s]["lr"], cv[s]["step"], params)
             for s in cv}
    for tag, spec in ARMS:
        best = None
        for lam in LAM_GRID:
            xs = [wfeature(cv[s]["lr"], spec, lam)[cv[s]["step"]] for s in PROBES]
            ys = [resid[s] for s in PROBES]
            x, y = np.concatenate(xs), np.concatenate(ys)
            k = fit_origin(x, y)
            sse = float(np.sum((y - k * x) ** 2))
            if best is None or sse < best[2]:
                best = (lam, k, sse)
        lam, kappa, _ = best
        out = []
        for s in TARGETS:
            if s not in cv:
                continue
            phi = wfeature(cv[s]["lr"], spec, lam)[cv[s]["step"]]
            m0 = float(np.mean(np.abs(resid[s])))
            m1 = float(np.mean(np.abs(resid[s] - kappa * phi)))
            out.append(f"{s} {m0:.4f}->{m1:.4f} ({100*(m1/m0-1):+.1f}%)")
        print(f"{tag:12s} lam*={lam:g} kappa={kappa:.4f}  " + "  ".join(out))


if __name__ == "__main__":
    main()
