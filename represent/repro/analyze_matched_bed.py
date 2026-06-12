"""Adjudication T-C: matched-probe rule on the 10.7M bed.

With m_wsdcon_15 trained (stage-2 1.5e-4 = terminal LR of sharp600/wsd/wsdld),
the shipped matched rule fires on this bed: calibrate kappa on wsdcon_15
alone for those targets; pooling is the status-quo fallback (cannot regress).
Compare to the shipped pooled numbers at d=0 (-18.1/-7.6/-6.2) and d=0.75
(-71.2/-36.1/-32.7).  lam* selection unchanged (probe-pooled grid).
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_curves as AC
from analyze_gen import load_gen, wfeature, fit_origin, LAM_GRID, PEAK

PROBES_POOL = ["wsdcon_5", "wsdcon_10", "wsdcon_40", "wsdcon_80"]
MATCH_PROBE = "wsdcon_15"
TARGETS = {"sharp600": (-18.1, -71.2), "wsd": (-7.6, -36.1),
           "wsdld": (-6.2, -32.7)}


def main():
    cv = AC.load_scale("m")
    for k, v in load_gen().items():
        cv[k] = v
    assert MATCH_PROBE in cv, "wsdcon_15 not trained yet"
    params, _ = AC.fit_mpl(cv, ["constant", "cosine", "wsdcon_20"], n_starts=8)
    resid = {s: cv[s]["loss"] - AC.mpl_pred_at(cv[s]["lr"], cv[s]["step"], params)
             for s in cv}

    for tag, delta in [("d=0", 0.0), ("d=0.75", 0.75)]:
        spec = ({"form": "pow", "delta": delta} if delta > 0 else {"form": "lr"})
        # lam* from pooled probes (unchanged protocol)
        best = None
        for lam in LAM_GRID:
            xs = [wfeature(cv[s]["lr"], spec, lam)[cv[s]["step"]] for s in PROBES_POOL]
            ys = [resid[s] for s in PROBES_POOL]
            x, y = np.concatenate(xs), np.concatenate(ys)
            k, _ = fit_origin(x, y)
            sse = float(np.sum((y - k * x) ** 2))
            if best is None or sse < best[2]:
                best = (lam, k, sse)
        lam, k_pool, _ = best
        # matched kappa from wsdcon_15 alone (same lam)
        xm = wfeature(cv[MATCH_PROBE]["lr"], spec, lam)[cv[MATCH_PROBE]["step"]]
        k_match, _ = fit_origin(xm, resid[MATCH_PROBE])
        k_match = max(0.0, k_match)
        print(f"\n== {tag}: lam*={lam:g} kappa pooled={k_pool:.4f} "
              f"matched(wsdcon_15)={k_match:.4f} ==")
        for s, (ship0, ship75) in TARGETS.items():
            if s not in cv:
                continue
            shipped = ship0 if delta == 0 else ship75
            phi = wfeature(cv[s]["lr"], spec, lam)[cv[s]["step"]]
            m0 = float(np.mean(np.abs(resid[s])))
            mp = float(np.mean(np.abs(resid[s] - k_pool * phi)))
            mm = float(np.mean(np.abs(resid[s] - k_match * phi)))
            print(f"  {s:10s} pooled {100*(mp/m0-1):+6.1f}%  "
                  f"matched {100*(mm/m0-1):+6.1f}%  (shipped {shipped:+.1f}%)")


if __name__ == "__main__":
    main()
