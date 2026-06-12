"""Concentration-matched probe calibration on the real ~10.7M OOF bed.
Identical to analyze_gen.py protocol (backbone on [constant,cosine,wsdcon_20],
lam per arm from pooled probes) EXCEPT kappa is taken from the single probe
whose stage-2 LR is log-nearest to the target's terminal LR (schedule-only
information; leakage-clean).
"""
import os, sys, json, glob
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_curves as AC
import analyze_gen as AG

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PEAK = 1.5e-3
PROBES = AG.PROBES
HELD = AG.HELD
LAM_GRID = AG.LAM_GRID
PROBE_LR = {"wsdcon_5": 5e-5, "wsdcon_10": 1e-4, "wsdcon_40": 4e-4,
            "wsdcon_80": 8e-4}

ARMS = [
    ("lr (paper)", {"form": "lr"}),
    ("pow d=0.25", {"form": "pow", "delta": 0.25}),
    ("pow d=0.5", {"form": "pow", "delta": 0.5}),
    ("pow d=0.75", {"form": "pow", "delta": 0.75}),
]


def main():
    cv = AC.load_scale("m")
    gen = AG.load_gen()
    for k, v in gen.items():
        cv[k] = v
    train = ["constant", "cosine", "wsdcon_20"]
    params, fobj = AC.fit_mpl(cv, train, n_starts=8)
    print(f"MPL fit obj={fobj:.5f}")
    resid = {}
    for s in cv:
        st = cv[s]["step"]
        pred = AC.mpl_pred_at(cv[s]["lr"], st, params)
        resid[s] = cv[s]["loss"] - pred

    for tag, spec in ARMS:
        # lam from pooled probes (audited convention)
        best = None
        for lam in LAM_GRID:
            xs = [AG.wfeature(cv[s]["lr"], spec, lam)[cv[s]["step"]]
                  for s in PROBES]
            ys = [resid[s] for s in PROBES]
            x, y = np.concatenate(xs), np.concatenate(ys)
            k, _ = AG.fit_origin(x, y)
            sse = float(np.sum((y - k * x) ** 2))
            if best is None or sse < best[1]:
                best = (lam, sse)
        lam = best[0]

        # per-probe kappas at this lam
        kap = {}
        for s in PROBES:
            x = AG.wfeature(cv[s]["lr"], spec, lam)[cv[s]["step"]]
            kap[s], _ = AG.fit_origin(x, resid[s])
        # pooled kappa (baseline)
        xs = [AG.wfeature(cv[s]["lr"], spec, lam)[cv[s]["step"]] for s in PROBES]
        kpool, _ = AG.fit_origin(np.concatenate(xs),
                                 np.concatenate([resid[s] for s in PROBES]))

        print(f"\n== {tag}: lam*={lam:g} kpool={kpool:.4f} "
              f"per-probe k={ {s: round(kap[s],4) for s in PROBES} } ==")
        rows = {}
        for s in HELD:
            if s not in cv:
                continue
            term = float(np.asarray(cv[s]["lr"], float)[-1])
            match = min(PROBES, key=lambda q: abs(np.log(PROBE_LR[q] / term)))
            kappa = kap[match]
            phi = AG.wfeature(cv[s]["lr"], spec, lam)[cv[s]["step"]]
            y = cv[s]["loss"]
            base = y - resid[s]
            mae0 = float(np.mean(np.abs(resid[s])))
            mae_m = float(np.mean(np.abs(y - (base + kappa * phi))))
            mae_p = float(np.mean(np.abs(y - (base + kpool * phi))))
            rows[s] = (mae0, mae_p, mae_m, match)
            ex = "  [excl]" if s == "cyclic" else ""
            print(f"  {s:9s} term={term:.1e} match={match:10s} "
                  f"MPL {mae0:.4f} pooled {mae_p:.4f} "
                  f"({100*(mae_p/mae0-1):+6.1f}%) matched {mae_m:.4f} "
                  f"({100*(mae_m/mae0-1):+6.1f}%){ex}")
        clean = {s: r for s, r in rows.items() if s != "cyclic"}
        mp = float(np.mean([100*(r[1]/r[0]-1) for r in clean.values()]))
        mm = float(np.mean([100*(r[2]/r[0]-1) for r in clean.values()]))
        print(f"  mean (excl cyclic): pooled {mp:+.1f}%  matched {mm:+.1f}%")


if __name__ == "__main__":
    main()
