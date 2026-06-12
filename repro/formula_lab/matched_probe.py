#!/usr/bin/env python3
"""Matched-probe calibration (adjudication test 1; SHIPPED rule).

Rule (strict exact-match-else-pool): when calibrating kappa for a target
schedule, if some probe's stage-2 LR EXACTLY equals the target's terminal LR,
fit kappa by origin-LS on that probe alone; otherwise pool all probes.
Leakage-clean: uses probe losses + the target's *schedule* (known a priori),
never target losses.  Audit note: the exact-match gate was identified after
observing the 10.7M-bed mismatch (post-hoc origin, not leakage; on that bed
no exact match exists, so the rule pools everywhere and the shipped bed
numbers are unchanged by construction).

On the public curves wsd/wsdld terminate at 3.00e-5 = wsdcon_3's stage-2
exactly, so the rule selects wsdcon_3 for the sharp targets.
LOS / T1 / matrix protocols are untouched by construction.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from reproduce_cosine_to_wsd import (  # noqa: E402
    load_curve, mpl_predict, metrics, MPL_PRECOMPUTED_INIT, SCALES,
)
from formula_lab.lab import (  # noqa: E402
    feature, fit_origin, probe_floor_powerlaw, DECAY, PROBES,
)

STAGE2 = {"wsdcon_3.csv": 3e-5, "wsdcon_9.csv": 9e-5, "wsdcon_18.csv": 18e-5}


def cal_probes_for(target_curve) -> list[str]:
    """Strict exact-match-else-pool probe selection from schedules only.

    'Exact' at the schedule-specification level: the public sharp decays are
    specified to end at END_LR=3e-5; the per-step discretization leaves the
    last implemented step within 0.23% of it (wsd 0.06%, wsdld 0.225%), so we match at 1% relative
    tolerance (next-nearest probe is 3x away)."""
    terminal = float(target_curve.lrs[-1])
    for name, lr2 in STAGE2.items():
        if abs(lr2 - terminal) <= 0.01 * lr2:
            return [name]
    return list(PROBES)


def kappa_from(scale: str, spec: dict, cal: list[str]) -> float:
    p = MPL_PRECOMPUTED_INIT[scale]
    xs, ys = [], []
    for n in cal:
        c = load_curve(scale, n)
        xs.append(feature(c, spec))
        ys.append(c.loss - mpl_predict(p, c))
    return max(0.0, fit_origin(np.concatenate(xs), np.concatenate(ys))[0])


def probes_only_matched(spec_for_scale) -> tuple[float, int, list]:
    m0s, m1s, wins, rows = [], [], 0, []
    for scale in SCALES:
        spec = spec_for_scale(scale)
        p = MPL_PRECOMPUTED_INIT[scale]
        for n in DECAY:
            cu = load_curve(scale, n)
            cal = cal_probes_for(cu)
            kappa = kappa_from(scale, spec, cal)
            bp = mpl_predict(p, cu)
            m0 = metrics(cu.loss, bp)["mae"]
            m1 = metrics(cu.loss, bp + kappa * feature(cu, spec))["mae"]
            m0s.append(m0); m1s.append(m1); wins += int(m1 < m0)
            rows.append((scale, n, ",".join(c.split(".")[0] for c in cal),
                         kappa, 100.0 * (m1 / m0 - 1.0)))
    return 100.0 * (np.mean(m1s) / np.mean(m0s) - 1.0), wins, rows


def dilution_matched(spec_for_scale) -> tuple[float, int]:
    m0s, m1s, wins = [], [], 0
    for scale in SCALES:
        spec = spec_for_scale(scale)
        p = MPL_PRECOMPUTED_INIT[scale]
        for held in DECAY:
            other = [n for n in DECAY if n != held][0]
            cu = load_curve(scale, held)
            cal_pr = cal_probes_for(cu)
            xs, ys = [], []
            for n in [other] + cal_pr:
                c = load_curve(scale, n)
                xs.append(feature(c, spec))
                ys.append(c.loss - mpl_predict(p, c))
            kappa = max(0.0, fit_origin(np.concatenate(xs),
                                        np.concatenate(ys))[0])
            bp = mpl_predict(p, cu)
            m0 = metrics(cu.loss, bp)["mae"]
            m1 = metrics(cu.loss, bp + kappa * feature(cu, spec))["mae"]
            m0s.append(m0); m1s.append(m1); wins += int(m1 < m0)
    return 100.0 * (np.mean(m1s) / np.mean(m0s) - 1.0), wins


def main():
    def fixed(spec):
        return lambda scale: spec

    def measured(scale):
        _, p = probe_floor_powerlaw(scale)
        return {"form": "pow", "delta": max(p - 1.0, 0.0), "lam": 10}

    ARMS = [
        ("d=0 (Eq.law)", fixed({"form": "lr", "lam": 10})),
        ("d=1/4 default", fixed({"form": "pow", "delta": 0.25, "lam": 10})),
        ("d=(p-1)+ measured", measured),
        ("d=1/2 frontier", fixed({"form": "pow", "delta": 0.5, "lam": 10})),
    ]
    print("== matched-probe calibration (strict exact-match-else-pool) ==")
    print(f"{'arm':>20s} {'probes-only':>12s} {'dilution':>10s}   "
          f"(shipped pooled: d=1/4 -23.0/-25.8, d=1/2 -28.6/-31.6)")
    for tag, sf in ARMS:
        po, w1, rows = probes_only_matched(sf)
        dl, w2 = dilution_matched(sf)
        print(f"{tag:>20s} {po:+10.1f} {w1}/6 {dl:+8.1f} {w2}/6")
    print("\nper-cell detail (d=1/4):")
    _, _, rows = probes_only_matched(fixed({"form": "pow", "delta": 0.25,
                                            "lam": 10}))
    for scale, n, cal, k, d in rows:
        print(f"  {scale:>4}M {n:22s} cal={cal:10s} kappa={k:.4f} {d:+.1f}%")


if __name__ == "__main__":
    main()
