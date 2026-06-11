#!/usr/bin/env python3
"""W2 strengthening: is MPL's B stable when refit WITHOUT any two-stage curve?

Official split = [cosine_24000, constant_24000, wsdcon_9] (includes a two-stage
probe).  Refit MPL on smooth-only splits and compare B, and the resulting
zero-probe Table-1 chain (kappa = c * eta_peak * B, c by LOO across scales).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from reproduce_cosine_to_wsd import (  # noqa: E402
    load_curve, mpl_predict, metrics, MPL_PRECOMPUTED_INIT, SCALES, PEAK_LR,
    subsample_curve,
)
import validate_theory as V  # noqa: E402
from formula_lab.lab import feature, fit_origin, DECAY  # noqa: E402

SPLITS = {
    "official(+wsdcon9)": None,  # use precomputed
    "cos24+const24": ["cosine_24000.csv", "constant_24000.csv"],
    "cos24+cos72": ["cosine_24000.csv", "cosine_72000.csv"],
    "cos24+cos72+const24": ["cosine_24000.csv", "cosine_72000.csv", "constant_24000.csv"],
}


def main():
    spec = {"form": "lr", "lam": 10}
    B_table = {}
    for split, curves in SPLITS.items():
        Bs = {}
        params_by_scale = {}
        for scale in SCALES:
            if curves is None:
                p = MPL_PRECOMPUTED_INIT[scale]
            else:
                train = [load_curve(scale, n) for n in curves]
                p = V.fit_mpl(train, np.array(MPL_PRECOMPUTED_INIT[scale], float),
                              V.F_MPL)
            params_by_scale[scale] = np.asarray(p, float)
            Bs[scale] = float(np.asarray(p)[3])
        B_table[split] = (Bs, params_by_scale)
        print(f"{split:22s} B = " + " ".join(f"{Bs[s]:8.1f}" for s in SCALES))

    print("\n== zero-probe Table-1 chain per split (kappa = c_loo * eta_peak * B) ==")
    for split, (Bs, params_by_scale) in B_table.items():
        # kappa_fit per scale on sharp residuals of the PRECOMPUTED backbone
        # (the deployed predictor keeps the official backbone; B only feeds the
        # amplitude rule)
        ratio = {}
        for s in SCALES:
            p0 = MPL_PRECOMPUTED_INIT[s]
            xs, ys = [], []
            for n in DECAY:
                c = load_curve(s, n)
                ys.append(c.loss - mpl_predict(p0, c))
                xs.append(feature(c, spec))
            kf = fit_origin(np.concatenate(xs), np.concatenate(ys))[0]
            ratio[s] = kf / (Bs[s] * PEAK_LR)
        m0s, m1s, wins = [], [], 0
        for tgt in SCALES:
            c_loo = float(np.mean([ratio[s] for s in SCALES if s != tgt]))
            kappa = c_loo * Bs[tgt] * PEAK_LR
            p0 = MPL_PRECOMPUTED_INIT[tgt]
            for n in DECAY:
                cu = load_curve(tgt, n)
                base = mpl_predict(p0, cu)
                m0 = metrics(cu.loss, base)["mae"]
                m1 = metrics(cu.loss, base + kappa * feature(cu, spec))["mae"]
                m0s.append(m0)
                m1s.append(m1)
                wins += int(m1 < m0)
        rs = np.array(list(ratio.values()))
        # ratio-of-means aggregation, matching deep_predict.py / Table 1
        delta = 100.0 * (np.mean(m1s) / np.mean(m0s) - 1.0)
        print(f"{split:22s} T1-B {delta:+6.1f}% {wins}/6  "
              f"cCV={rs.std()/abs(rs.mean())*100:.0f}%")


if __name__ == "__main__":
    main()
