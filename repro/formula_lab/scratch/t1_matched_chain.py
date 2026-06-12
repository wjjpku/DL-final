#!/usr/bin/env python3
"""Last-chance candidate: Table-1 cross-scale chain with the matched-probe
(wsdcon_3) kappa as the per-scale amplitude predictor, and its hybrid
(equal-weight average) with the mpl-B chain.

Budget identical to T1lin/T1-B: target scale sees ONLY probes + frozen MPL
params; c_loo from other scales' sharp fits.  Number to beat: -44.0 (T1lin).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from reproduce_cosine_to_wsd import (  # noqa: E402
    load_curve, mpl_predict, metrics, MPL_PRECOMPUTED_INIT, SCALES, PEAK_LR,
)
from formula_lab.lab import (  # noqa: E402
    feature, fit_origin, kappa_fit_sharp, probe_floor_powerlaw, DECAY,
)

MATCH = "wsdcon_3.csv"  # stage-2 LR 3e-5 == sharp targets' terminal LR


def kappa_matched(scale: str, spec: dict) -> float:
    p = MPL_PRECOMPUTED_INIT[scale]
    c = load_curve(scale, MATCH)
    return fit_origin(feature(c, spec), c.loss - mpl_predict(p, c))[0]


def kappa_B(scale: str) -> float:
    return float(MPL_PRECOMPUTED_INIT[scale][3]) * PEAK_LR


def run_chain(spec_for_scale, pred_fn, tag: str):
    specs = {s: spec_for_scale(s) for s in SCALES}
    preds = {s: pred_fn(s, specs[s]) for s in SCALES}
    fits = {s: kappa_fit_sharp(s, specs[s]) for s in SCALES}
    ratio = {s: fits[s] / preds[s] for s in SCALES}
    rv = np.array(list(ratio.values()))
    cv = float(rv.std() / abs(rv.mean()) * 100)
    m0s, m1s, wins = [], [], 0
    for tgt in SCALES:
        c_loo = float(np.mean([ratio[s] for s in SCALES if s != tgt]))
        kappa = c_loo * preds[tgt]
        p = MPL_PRECOMPUTED_INIT[tgt]
        for n in DECAY:
            cu = load_curve(tgt, n)
            base = mpl_predict(p, cu)
            m0 = metrics(cu.loss, base)["mae"]
            m1 = metrics(cu.loss, base + kappa * feature(cu, specs[tgt]))["mae"]
            m0s.append(m0); m1s.append(m1); wins += int(m1 < m0)
    d = 100.0 * (np.mean(m1s) / np.mean(m0s) - 1.0)
    print(f"{tag:42s} {d:+7.1f}% {wins}/6  cCV={cv:5.1f}%  "
          f"ratios=" + "/".join(f"{ratio[s]:.3f}" for s in SCALES))
    return d


def main():
    def fixed(spec):
        return lambda s: dict(spec)

    def measured(s):
        _, p = probe_floor_powerlaw(s)
        return {"form": "pow", "delta": max(p - 1.0, 0.0), "lam": 10}

    arms = [
        ("lr d=0", fixed({"form": "lr", "lam": 10})),
        ("pow d=0.25", fixed({"form": "pow", "delta": 0.25, "lam": 10})),
        ("pow d=measured(p-1)+", measured),
        ("pow d=0.5", fixed({"form": "pow", "delta": 0.5, "lam": 10})),
    ]
    print("== T1 chain, kappa_pred = matched-probe (wsdcon_3) fit ==")
    print("   beat: T1lin -44.0 / T1-B -43.1 (paper, 6/6)")
    for tag, sf in arms:
        run_chain(sf, kappa_matched, f"matched-chain {tag}")
    print("\n== hybrid: kappa_pred = mean(matched-probe, B-chain) ==")
    for tag, sf in arms:
        run_chain(sf, lambda s, sp: 0.5 * (kappa_matched(s, sp) + kappa_B(s)),
                  f"hybrid(B+matched) {tag}")
    print("\n== reference reruns (sanity) ==")
    run_chain(fixed({"form": "lr", "lam": 10}),
              lambda s, sp: kappa_B(s), "T1-B lr d=0 (expect -43.1)")


if __name__ == "__main__":
    main()
