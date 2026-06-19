#!/usr/bin/env python3
"""Frozen geometry-tau one-kappa step-time model.

This is the reusable implementation of the current primary transferable rule:

    L_hat(t) = L_MPL(t) + kappa_hat * phi_tau(t)

Only kappa is fitted from calibration loss residuals.  The target loss residual
is never used by the target-holdout path.  Tau is computed from LR schedule
geometry, and the primary rule does not fit or transfer nuisance coefficients.
"""
from __future__ import annotations

import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path

TMP_CACHE = Path(os.environ.get("TMPDIR", "/tmp")) / "dl-final-cache"
TMP_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(TMP_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(TMP_CACHE / "xdg"))

import numpy as np

REPO = Path(__file__).resolve().parent
ROOT = REPO.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from reproduce_cosine_to_wsd import SCALES  # noqa: E402
from step_time_decomposed_estimator import summarize  # noqa: E402
from step_time_shape_routed_estimator import (  # noqa: E402
    CORE_CURVES,
    EXTENDED_CURVES,
    build_cache,
    fit_kappa,
    route_for_target,
    schedule_stats,
    score_target,
)


OUT_DIR = ROOT / "results" / "frozen_step_time_model"

STEP_TAU_BASE = 512.0
STEP_DROP_WEAK = 0.40
STEP_DROP_FULL = 0.90
TAIL_TAU_PER_STEP = 1.25
MAX_TAU = 8192.0


@dataclass(frozen=True)
class FrozenRoute:
    target_curve: str
    target_label: str
    route: str
    train_curves: tuple[str, ...]
    geometry_tau: float
    table_tau: float
    nuisance: str
    target_drop_norm: float
    target_decay_span: float
    target_schedule_len: float
    rationale: str


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt_pct(value: float) -> str:
    return f"{value:+.1f}%"


def source_text(train_curves: tuple[str, ...]) -> str:
    return "+".join(curve.replace(".csv", "") for curve in train_curves) if train_curves else "none"


def geometry_tau(stats: dict[str, dict[str, float]], target_curve: str) -> float:
    row = stats[target_curve]
    drop = float(row["drop_norm"])
    span = float(row["decay_span"])
    length = float(row["schedule_len"])
    if drop <= 0.05:
        return 0.0
    if span > 16000.0 and length <= 30000.0:
        return 0.0
    if span > 100.0:
        return min(MAX_TAU, TAIL_TAU_PER_STEP * span)
    q = np.clip((drop - STEP_DROP_WEAK) / (STEP_DROP_FULL - STEP_DROP_WEAK), 0.0, 1.0)
    return STEP_TAU_BASE * (1.0 + 2.0 * float(q) ** 3)


def frozen_route(
    stats: dict[str, dict[str, float]],
    target_curve: str,
    target_label: str,
    *,
    self_fit: bool = False,
) -> FrozenRoute:
    base_route = route_for_target(stats, target_curve)
    train_curves = (target_curve,) if self_fit else tuple(base_route["train_curves"])
    tau = geometry_tau(stats, target_curve)
    return FrozenRoute(
        target_curve=target_curve,
        target_label=target_label,
        route=str(base_route["route"]),
        train_curves=train_curves,
        geometry_tau=tau,
        table_tau=float(base_route["tau"]),
        nuisance="none",
        target_drop_norm=float(stats[target_curve]["drop_norm"]),
        target_decay_span=float(stats[target_curve]["decay_span"]),
        target_schedule_len=float(stats[target_curve]["schedule_len"]),
        rationale="primary frozen rule: geometry tau, one fitted source kappa, no nuisance projection",
    )


def route_to_row(route: FrozenRoute, mode: str, self_fit: bool) -> dict[str, object]:
    return {
        "mode": mode,
        "target_curve": route.target_curve,
        "target_label": route.target_label,
        "route": route.route,
        "train_curves": source_text(route.train_curves),
        "table_tau": route.table_tau,
        "geometry_tau": route.geometry_tau,
        "tau_ratio_vs_table": route.geometry_tau / route.table_tau if route.table_tau > 0.0 else 0.0,
        "nuisance": route.nuisance,
        "target_drop_norm": route.target_drop_norm,
        "target_decay_span": route.target_decay_span,
        "target_schedule_len": route.target_schedule_len,
        "self_fit": int(self_fit),
        "target_residual_used_for_kappa": int(self_fit),
        "rationale": route.rationale,
    }


def evaluate(
    curve_defs: tuple[tuple[str, str], ...] | list[tuple[str, str]] = CORE_CURVES,
    *,
    mode: str = "frozen_target_holdout",
    self_fit: bool = False,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    cache = build_cache(curve_defs)
    stats = schedule_stats(cache, curve_defs)
    route_rows: list[dict[str, object]] = []
    details: list[dict[str, object]] = []

    for target_curve, target_label in curve_defs:
        route = frozen_route(stats, target_curve, target_label, self_fit=self_fit)
        route_rows.append(route_to_row(route, mode, self_fit))
        for scale in SCALES:
            if route.train_curves and route.geometry_tau > 0.0:
                kappa = fit_kappa(cache, scale, route.train_curves, route.geometry_tau, route.nuisance)
            else:
                kappa = 0.0
            scored = score_target(cache, scale, target_curve, kappa, route.geometry_tau)
            details.append(
                {
                    "mode": mode,
                    "scale": scale,
                    "target_curve": target_curve,
                    "target_label": target_label,
                    "route": route.route,
                    "train_curves": source_text(route.train_curves),
                    "table_tau": route.table_tau,
                    "geometry_tau": route.geometry_tau,
                    "nuisance": route.nuisance,
                    "kappa": kappa,
                    "self_fit": int(self_fit),
                    "target_residual_used_for_kappa": int(self_fit),
                    **scored,
                }
            )

    summary: list[dict[str, object]] = []
    for target_curve, target_label in curve_defs:
        sub = [row for row in details if row["target_curve"] == target_curve]
        summary.append({"target_curve": target_curve, "target_label": target_label, **summarize(sub)})
    return route_rows, details, summary


def write_report(
    target_rows: list[dict[str, object]],
    self_rows: list[dict[str, object]],
    extended_rows: list[dict[str, object]],
    routes: list[dict[str, object]],
) -> None:
    target = summarize(target_rows)
    self_fit = summarize(self_rows)
    extended = summarize(extended_rows)
    safety_rows = [
        row
        for row in extended_rows
        if row["target_curve"] in {"cosine_24000.csv", "constant_24000.csv", "constant_72000.csv"}
    ]
    safety = summarize(safety_rows)
    lines = [
        "# Frozen Geometry-Tau One-Kappa Model\n\n",
        "This is the executable implementation of the frozen primary transferable rule.  It fits only `kappa` from source residuals; target residuals are excluded in target-holdout mode.\n\n",
        "## Formula\n\n",
        "```text\n",
        "L_hat(t) = L_MPL(t) + kappa_hat * phi_tau(t)\n",
        "phi_tau(t) = sum_{u <= t} exp(-(t-u)/tau) * max(lr_{u-1} - lr_u, 0) / lr_peak\n",
        "```\n\n",
        "Geometry tau:\n\n",
        "```text\n",
        "if positive_drop_span > 100:\n",
        "    tau = min(8192, 1.25 * positive_drop_span)\n",
        "else:\n",
        "    q = clip((total_positive_drop - 0.40) / (0.90 - 0.40), 0, 1)\n",
        "    tau = 512 * (1 + 2 q^3)\n",
        "```\n\n",
        "Safety gates set `tau=0` for zero-positive-drop and short-smooth controls.\n\n",
        "## Metrics\n\n",
        f"- Target-holdout primary rule: mean `{fmt_pct(float(target['mean_delta']))}`, worst `{fmt_pct(float(target['worst_delta']))}`, non-harm `{int(target['nonharm'])}/{int(target['rows'])}`.\n",
        f"- Same-curve one-kappa diagnostic: mean `{fmt_pct(float(self_fit['mean_delta']))}`, worst `{fmt_pct(float(self_fit['worst_delta']))}`, non-harm `{int(self_fit['nonharm'])}/{int(self_fit['rows'])}`.\n",
        f"- Extended safety audit: mean `{fmt_pct(float(extended['mean_delta']))}`, worst `{fmt_pct(float(extended['worst_delta']))}`, non-harm `{int(extended['nonharm'])}/{int(extended['rows'])}`.\n",
        f"- Safety controls only: mean `{fmt_pct(float(safety['mean_delta']))}`, worst `{fmt_pct(float(safety['worst_delta']))}`, non-harm `{int(safety['nonharm'])}/{int(safety['rows'])}`.\n\n",
        "## Route Table\n\n",
        "| target | route | source | geometry tau | table tau | target residual used for kappa? |\n",
        "|---|---|---|---:|---:|---:|\n",
    ]
    for row in routes:
        lines.append(
            f"| {row['target_label']} | {row['route']} | `{row['train_curves']}` | "
            f"{float(row['geometry_tau']):.1f} | {float(row['table_tau']):.1f} | "
            f"{int(row['target_residual_used_for_kappa'])} |\n"
        )
    lines += [
        "\n## Decision\n\n",
        "- Use this module as the source of truth for the frozen primary transferable rule.\n",
        "- Use decomposed self-fit only as a residual explanation diagnostic, not as the deployment rule.\n",
        "- Use residualized and cross-family results as audits around this primary rule.\n",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    routes, details, summary = evaluate(CORE_CURVES, mode="frozen_target_holdout", self_fit=False)
    self_routes, self_details, self_summary = evaluate(CORE_CURVES, mode="frozen_self_fit", self_fit=True)
    extended_routes, extended_details, extended_summary = evaluate(
        EXTENDED_CURVES,
        mode="frozen_extended_safety",
        self_fit=False,
    )
    write_csv(OUT_DIR / "route_table.csv", routes)
    write_csv(OUT_DIR / "target_holdout_details.csv", details)
    write_csv(OUT_DIR / "target_holdout_summary.csv", summary)
    write_csv(OUT_DIR / "self_fit_route_table.csv", self_routes)
    write_csv(OUT_DIR / "self_fit_details.csv", self_details)
    write_csv(OUT_DIR / "self_fit_summary.csv", self_summary)
    write_csv(OUT_DIR / "extended_route_table.csv", extended_routes)
    write_csv(OUT_DIR / "extended_safety_details.csv", extended_details)
    write_csv(OUT_DIR / "extended_safety_summary.csv", extended_summary)
    write_report(details, self_details, extended_details, routes)
    target = summarize(details)
    self_fit = summarize(self_details)
    print(f"wrote {OUT_DIR / 'REPORT.md'}")
    print(
        f"frozen target={float(target['mean_delta']):+.1f}%/{float(target['worst_delta']):+.1f}% "
        f"self={float(self_fit['mean_delta']):+.1f}%/{float(self_fit['worst_delta']):+.1f}%"
    )


if __name__ == "__main__":
    main()
