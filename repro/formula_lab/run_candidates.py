#!/usr/bin/env python3
"""Sweep candidate formula variants through the accuracy protocols.

Baselines to beat (paper / engineering map):
  Table-1 chain (probe-linear, lr form): -44.0%, 6/6 wins
  leave-one-sharp (lr form):             -49.0%, 6/6 wins
  in-sample sharp R^2 (lr@10):           ~0.29/0.61/0.74 pooled wsd+wsdld
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from formula_lab import lab  # noqa: E402

OUT = REPO.parent / "results" / "formula_lab"


def show(tag: str, spec: dict, chain: str | None, use_probe_p=False):
    r2 = lab.insample_r2(spec if not use_probe_p else
                         _probe_spec(spec))
    row = {"tag": tag, "spec": spec, "chain": chain, "use_probe_p": use_probe_p,
           "r2": r2}
    los = lab.leave_one_sharp_protocol(spec if not use_probe_p else _probe_spec(spec))
    row["los_delta"] = los["delta_pct"]
    row["los_wins"] = los["wins"]
    if chain:
        t1 = lab.table1_protocol(spec, chain, use_probe_p=use_probe_p)
        row["t1_delta"] = t1["delta_pct"]
        row["t1_wins"] = t1["wins"]
        row["t1_ratios"] = t1["ratios"]
        rcv = np.array(list(t1["ratios"].values()))
        row["t1_ratio_cv"] = float(rcv.std() / abs(rcv.mean()) * 100)
    else:
        row["t1_delta"] = float("nan")
        row["t1_wins"] = -1
        row["t1_ratio_cv"] = float("nan")
    r2s = " ".join(f"{r2[s]:.3f}" for s in ["25", "100", "400"])
    print(f"{tag:34s} R2[{r2s}] LOS {row['los_delta']:+6.1f}% {row['los_wins']}/6  "
          f"T1 {row['t1_delta']:+6.1f}% {row['t1_wins']}/6 cCV={row['t1_ratio_cv']:.0f}%")
    return row


def _probe_spec(spec):
    # representative: 100M probe p (only used for the R2/LOS display of probe-p
    # variants; table1 handles per-scale specs internally)
    sp = dict(spec)
    _, p = lab.probe_floor_powerlaw("100")
    if sp.get("form") == "floor":
        sp["p"] = p
    elif sp.get("form") == "pow":
        sp["delta"] = max(p - 1.0, 0.0)
    return sp


def main():
    rows = []
    print("== baselines (paper law, form=lr) ==")
    rows.append(show("lr@10 / probe-linear  [PAPER]", {"form": "lr", "lam": 10}, "probe-linear"))
    rows.append(show("lr@10 / mpl-B", {"form": "lr", "lam": 10}, "mpl-B"))

    print("\n== power-weighted drops ==")
    for d in [0.25, 0.5, 0.75, 1.0]:
        rows.append(show(f"pow d={d}@10 / probe-power", {"form": "pow", "delta": d, "lam": 10}, "probe-power"))
    rows.append(show("pow d=probe@10 / probe-power", {"form": "pow", "delta": 0.5, "lam": 10}, "probe-power", use_probe_p=True))

    print("\n== floor-drop form ==")
    for p in [1.25, 1.5, 1.75, 2.0]:
        rows.append(show(f"floor p={p}@10 / probe-power", {"form": "floor", "p": p, "lam": 10}, "probe-power"))
    rows.append(show("floor p=probe@10 / probe-power", {"form": "floor", "p": 1.5, "lam": 10}, "probe-power", use_probe_p=True))

    print("\n== affine weight ==")
    for rho in [0.5, 1.0]:
        rows.append(show(f"affine rho={rho}@10 / probe-linear", {"form": "affine", "rho": rho, "lam": 10}, "probe-linear"))

    print("\n== lambda sensitivity for the best forms ==")
    for lam in [5.0, 7.0, 14.0, 20.0]:
        rows.append(show(f"pow d=0.5@{lam:g} / probe-power", {"form": "pow", "delta": 0.5, "lam": lam}, "probe-power"))
        rows.append(show(f"floor p=1.5@{lam:g} / probe-power", {"form": "floor", "p": 1.5, "lam": lam}, "probe-power"))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "run_candidates.json").write_text(
        json.dumps(rows, indent=1, default=str), encoding="utf-8")
    print(f"\nwrote {OUT / 'run_candidates.json'}")


if __name__ == "__main__":
    main()
