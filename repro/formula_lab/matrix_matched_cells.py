#!/usr/bin/env python3
"""Adjudication T-A (SHIPPED): 6x6 matrix with the matched-probe override.

Rule: for a (train, test) cell where the TRAIN curve's stage-2 constant LR
exactly matches the TEST curve's terminal LR (schedule-level, 1% rel tol),
replace the shrunk final_no_cap kappa with the raw origin-LS kappa fitted on
the train curve (matched-probe calibration, same as matched_probe.py).
Leakage-clean: uses only train losses + both schedules.
Affected cells on the public 6x6: wsdcon_3 -> wsd, wsdcon_3 -> wsdld.
Numbers to beat (pow d=0.25@10): worst -2.85 / mean -13.51.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import current_law_amplitude_kappa_search as amp  # noqa: E402
import current_law_continuous_kappa_search as base  # noqa: E402
import current_law_eb_kappa_search as eb  # noqa: E402
import current_law_orthogonal_kappa_search as orth  # noqa: E402
from current_law_final_kappa import final_kappa  # noqa: E402
from formula_lab.lab import feature, fit_origin  # noqa: E402
from formula_lab.matrix_protocol import feats_for  # noqa: E402

STAGE2 = {"wsdcon_3.csv": 3e-5, "wsdcon_9.csv": 9e-5, "wsdcon_18.csv": 18e-5}


def is_matched(train_curve: str, test_curve: str, scale: str) -> bool:
    lr2 = STAGE2.get(train_curve)
    if lr2 is None:
        return False
    cu = base.load_curve(scale, test_curve)
    term = float(cu.lrs[-1])
    return abs(lr2 - term) <= 0.01 * lr2 and train_curve != test_curve


def raw_kappa(scale: str, train_curve: str, spec: dict) -> float:
    c = base.load_curve(scale, train_curve)
    resid = c.loss - base.mpl_predict(base.MPL_PRECOMPUTED_INIT[scale], c)
    return max(0.0, fit_origin(feature(c, spec), resid)[0])


def run_matrix_matched(spec: dict) -> dict:
    feats = feats_for(spec)
    base_rows = []
    for curve, label in base.CURVES:
        for scale in base.SCALES:
            stats = amp.enriched_stats(scale, curve, feats)
            base_rows.append({"scale": scale, "train_curve": curve,
                              "train_label": label, **stats})
    orth_stats = {(scale, curve): orth.orthogonal_stats(scale, curve, feats, 2)
                  for curve, _ in base.CURVES for scale in base.SCALES}
    details = []
    n_override = 0
    for train_curve, _ in base.CURVES:
        pool = [r for r in base_rows if r["train_curve"] != train_curve]
        tau = eb.estimate_tau(pool, "q75")["tau"]
        for scale in base.SCALES:
            kappa_std = final_kappa(orth_stats[(scale, train_curve)], tau,
                                    cap=None)
            for test_curve, _ in base.CURVES:
                if is_matched(train_curve, test_curve, scale):
                    kappa = raw_kappa(scale, train_curve, spec)
                    n_override += 1
                else:
                    kappa = kappa_std
                scored = base.score(scale, test_curve, kappa, feats)
                details.append({"train_curve": train_curve,
                                "test_curve": test_curve, **scored})
    summary = {}
    for tr, _ in base.CURVES:
        for te, _ in base.CURVES:
            sub = [r for r in details if r["train_curve"] == tr
                   and r["test_curve"] == te]
            summary[(tr, te)] = float(np.mean([float(r["delta_pct"])
                                               for r in sub]))
    off = [v for (tr, te), v in summary.items() if tr != te]
    return {"worst": float(np.max(off)), "mean": float(np.mean(off)),
            "median": float(np.median(off)), "n_override": n_override,
            "w3_wsd": summary[("wsdcon_3.csv", "wsd_20000_24000.csv")],
            "w3_wsdld": summary[("wsdcon_3.csv", "wsdld_20000_24000.csv")]}


def main():
    for tag, spec in [("lr@10", {"form": "lr", "lam": 10}),
                      ("pow d=0.25@10", {"form": "pow", "delta": 0.25,
                                         "lam": 10})]:
        r = run_matrix_matched(spec)
        print(f"{tag:15s} worst {r['worst']:+.2f} median {r['median']:+.2f} "
              f"mean {r['mean']:+.2f}  w3->wsd {r['w3_wsd']:+.2f} "
              f"w3->wsdld {r['w3_wsdld']:+.2f}  overrides={r['n_override']}")
    print("beat: lr@10 worst -2.72/mean -12.08 | d=1/4 worst -2.85/mean -13.51")
    print("baseline matched cells d=1/4: w3->wsd -21.33, w3->wsdld -18.32")


if __name__ == "__main__":
    main()
