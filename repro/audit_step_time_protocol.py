#!/usr/bin/env python3
"""Audit the shape-routed step-time protocol for target-loss blindness.

The shape-routed estimator is meant to be a deployment-style target-holdout
head: the target LR schedule may choose a route, but the target loss residual
must not choose the calibration source, tau, nuisance projection, or kappa.

This script checks that property dynamically by scrambling each target
residual and confirming that the correction assigned to that same target is
unchanged.
"""
from __future__ import annotations

import csv
import os
import zlib
from pathlib import Path

import numpy as np

TMP_CACHE = Path(os.environ.get("TMPDIR", "/tmp")) / "dl-final-cache"
TMP_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(TMP_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(TMP_CACHE / "xdg"))

from reproduce_cosine_to_wsd import SCALES
from step_time_shape_routed_estimator import (
    CORE_CURVES,
    EXTENDED_CURVES,
    OUT_DIR,
    build_cache,
    fit_kappa,
    response_feature,
    route_for_target,
    schedule_stats,
)


def train_text(train_curves: tuple[str, ...]) -> str:
    if not train_curves:
        return "none"
    return "+".join(curve.replace(".csv", "") for curve in train_curves)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def route_signature(row: dict[str, object]) -> dict[str, object]:
    return {
        "route": str(row["route"]),
        "train_curves": str(row["train_curves"]),
        "tau": float(row["tau"]),
        "nuisance": str(row["nuisance"]),
    }


def compute_route_rows(curve_defs: tuple[tuple[str, str], ...] | list[tuple[str, str]]) -> list[dict[str, object]]:
    cache = build_cache(curve_defs)
    stats = schedule_stats(cache, curve_defs)
    rows: list[dict[str, object]] = []
    for target_curve, target_label in curve_defs:
        route = route_for_target(stats, target_curve)
        train_curves = tuple(route["train_curves"])
        rows.append(
            {
                "target_curve": target_curve,
                "target_label": target_label,
                "route": route["route"],
                "train_curves": train_text(train_curves),
                "tau": float(route["tau"]),
                "nuisance": route["nuisance"],
            }
        )
    return rows


def audit_route_table(stored_path: Path, curve_defs: tuple[tuple[str, str], ...] | list[tuple[str, str]]) -> int:
    stored = {row["target_curve"]: route_signature(row) for row in read_csv(stored_path)}
    current = {str(row["target_curve"]): route_signature(row) for row in compute_route_rows(curve_defs)}
    require(set(stored) == set(current), f"{stored_path} route targets do not match recomputed targets")
    matched = 0
    for target_curve in sorted(stored):
        got = current[target_curve]
        want = stored[target_curve]
        require(got["route"] == want["route"], f"{target_curve}: route drifted")
        require(got["train_curves"] == want["train_curves"], f"{target_curve}: train_curves drifted")
        require(abs(float(got["tau"]) - float(want["tau"])) <= 1e-12, f"{target_curve}: tau drifted")
        require(got["nuisance"] == want["nuisance"], f"{target_curve}: nuisance drifted")
        matched += 1
    return matched


def stable_seed(text: str) -> int:
    return zlib.adler32(text.encode("utf-8")) & 0xFFFFFFFF


def audit_target_loss_blindness() -> list[dict[str, object]]:
    cache = build_cache(EXTENDED_CURVES)
    stats = schedule_stats(cache, EXTENDED_CURVES)
    rows: list[dict[str, object]] = []
    for target_curve, target_label in EXTENDED_CURVES:
        route = route_for_target(stats, target_curve)
        train_curves = tuple(route["train_curves"])
        tau = float(route["tau"])
        nuisance = str(route["nuisance"])
        target_not_in_train = target_curve not in train_curves

        for scale in SCALES:
            if train_curves and tau > 0.0:
                kappa = fit_kappa(cache, scale, train_curves, tau, nuisance)
            else:
                kappa = 0.0

            mutated_cache = cache.copy()
            mutated_row = dict(cache[(scale, target_curve)])
            rng = np.random.default_rng(stable_seed(f"{scale}:{target_curve}"))
            mutated_row["residual"] = 1000.0 * rng.standard_normal(len(mutated_row["residual"]))
            mutated_cache[(scale, target_curve)] = mutated_row

            if train_curves and tau > 0.0:
                scrambled_kappa = fit_kappa(mutated_cache, scale, train_curves, tau, nuisance)
            else:
                scrambled_kappa = 0.0

            target_curve_obj = cache[(scale, target_curve)]["curve"]
            if tau > 0.0:
                correction = kappa * response_feature(target_curve_obj, tau)
                scrambled_correction = scrambled_kappa * response_feature(target_curve_obj, tau)
                max_correction_abs_diff = float(np.max(np.abs(correction - scrambled_correction)))
            else:
                max_correction_abs_diff = 0.0
            kappa_abs_diff = abs(kappa - scrambled_kappa)
            status = (
                target_not_in_train
                and kappa_abs_diff <= 1e-12
                and max_correction_abs_diff <= 1e-12
            )
            rows.append(
                {
                    "scale": scale,
                    "target_curve": target_curve,
                    "target_label": target_label,
                    "route": route["route"],
                    "train_curves": train_text(train_curves),
                    "tau": tau,
                    "nuisance": nuisance,
                    "target_not_in_train": int(target_not_in_train),
                    "kappa": kappa,
                    "scrambled_target_residual_kappa": scrambled_kappa,
                    "kappa_abs_diff": kappa_abs_diff,
                    "max_correction_abs_diff": max_correction_abs_diff,
                    "status": "pass" if status else "fail",
                }
            )
    return rows


def write_report(rows: list[dict[str, object]], core_route_matches: int, extended_route_matches: int) -> None:
    total = len(rows)
    target_excluded = sum(int(row["target_not_in_train"]) for row in rows)
    pass_count = sum(1 for row in rows if row["status"] == "pass")
    max_kappa_diff = max(float(row["kappa_abs_diff"]) for row in rows)
    max_correction_diff = max(float(row["max_correction_abs_diff"]) for row in rows)
    nonzero_routes = sum(float(row["tau"]) > 0.0 for row in rows)
    lines = [
        "# Shape-Routed Protocol Audit\n\n",
        "This audit checks the deployment protocol behind the shape-routed step-time estimator.  The target LR schedule is allowed to choose a route, but the target loss residual is not allowed to choose the calibration source, tau, nuisance basis, kappa, or predicted correction.\n\n",
        "## Checks\n\n",
        f"- Core route-table lock: `{core_route_matches}/6` committed routes match the LR-schedule-only recomputation.\n",
        f"- Extended route-table lock: `{extended_route_matches}/9` committed routes match the LR-schedule-only recomputation.\n",
        f"- Target exclusion: `{target_excluded}/{total}` target-scale predictions exclude the target curve from the calibration set.\n",
        f"- Nonzero correction routes audited: `{nonzero_routes}/{total}`.\n",
        f"- Target residual scramble: max `|delta kappa| = {max_kappa_diff:.3e}`, max `|delta correction| = {max_correction_diff:.3e}`.\n",
        f"- Overall protocol status: `{pass_count}/{total}` rows pass.\n\n",
        "## Interpretation\n\n",
        "The audit does not claim that the route thresholds were chosen prospectively.  It verifies the narrower but essential deployment property: after the rule is fixed, a target's own loss residual cannot affect its assigned correction.  Any measured target-holdout gain therefore comes from LR-shape routing plus source-curve calibration, not from fitting the target residual.\n",
    ]
    (OUT_DIR / "PROTOCOL_AUDIT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    core_route_matches = audit_route_table(OUT_DIR / "route_table.csv", CORE_CURVES)
    extended_route_matches = audit_route_table(OUT_DIR / "extended_route_table.csv", EXTENDED_CURVES)
    rows = audit_target_loss_blindness()
    write_csv(OUT_DIR / "protocol_audit.csv", rows)
    write_report(rows, core_route_matches, extended_route_matches)

    failed = [row for row in rows if row["status"] != "pass"]
    require(not failed, f"protocol audit failed for {len(failed)} rows")
    print("shape-routed protocol audit passed")


if __name__ == "__main__":
    main()
