#!/usr/bin/env python3
"""Audit which parts of the step-time model are actually necessary.

This script deliberately avoids paper/slides edits.  It produces a compact
ledger of fitted parameters, schedule-only knobs, and the metric contribution
of each modeling component.
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

TMP_CACHE = Path(os.environ.get("TMPDIR", "/tmp")) / "dl-final-cache"
TMP_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(TMP_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(TMP_CACHE / "xdg"))

REPO = Path(__file__).resolve().parent
ROOT = REPO.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from reproduce_cosine_to_wsd import SCALES  # noqa: E402
from step_time_cross_family_estimator import (  # noqa: E402
    route_for_target as cross_family_route_for_target,
    run_cross_family,
)
from step_time_decomposed_estimator import summarize  # noqa: E402
from step_time_shape_routed_estimator import (  # noqa: E402
    CORE_CURVES,
    build_cache,
    fit_kappa,
    route_for_target as shape_route_for_target,
    run_shape_routed,
    schedule_stats,
    score_target,
)


OUT_DIR = ROOT / "results" / "step_time_model_complexity"


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


def eval_shape_route(
    *,
    mode: str,
    nuisance_override: str | None = None,
    self_fit: bool = False,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    cache = build_cache(CORE_CURVES)
    stats = schedule_stats(cache, CORE_CURVES)
    rows: list[dict[str, object]] = []
    routes: list[dict[str, object]] = []
    for target_curve, target_label in CORE_CURVES:
        route = shape_route_for_target(stats, target_curve)
        train_curves = (target_curve,) if self_fit else tuple(route["train_curves"])
        tau = float(route["tau"])
        nuisance = str(route["nuisance"]) if nuisance_override is None else nuisance_override
        routes.append(
            {
                "mode": mode,
                "target_curve": target_curve,
                "target_label": target_label,
                "route": route["route"],
                "train_curves": "+".join(c.replace(".csv", "") for c in train_curves) if train_curves else "none",
                "tau": tau,
                "nuisance": nuisance,
                "self_fit": int(self_fit),
            }
        )
        for scale in SCALES:
            if train_curves and tau > 0.0:
                kappa = fit_kappa(cache, scale, train_curves, tau, nuisance)
            else:
                kappa = 0.0
            scored = score_target(cache, scale, target_curve, kappa, tau)
            rows.append(
                {
                    "mode": mode,
                    "scale": scale,
                    "target_curve": target_curve,
                    "target_label": target_label,
                    "route": route["route"],
                    "train_curves": "+".join(c.replace(".csv", "") for c in train_curves) if train_curves else "none",
                    "tau": tau,
                    "nuisance": nuisance,
                    "kappa": kappa,
                    **scored,
                }
            )
    return routes, rows


def eval_cross_family_no_nuisance(
    attenuation_power: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    cache = build_cache(CORE_CURVES)
    stats = schedule_stats(cache, CORE_CURVES)
    rows: list[dict[str, object]] = []
    routes: list[dict[str, object]] = []
    mode = f"cross_family_no_nuisance_p{attenuation_power:g}"
    for target_curve, target_label in CORE_CURVES:
        route = cross_family_route_for_target(stats, target_curve, attenuation_power=attenuation_power)
        train_curves = tuple(route["train_curves"])
        tau = float(route["tau"])
        nuisance = "none"
        attenuation = float(route["attenuation"])
        routes.append(
            {
                "mode": mode,
                "target_curve": target_curve,
                "target_label": target_label,
                "route": route["route"],
                "train_curves": "+".join(c.replace(".csv", "") for c in train_curves) if train_curves else "none",
                "tau": tau,
                "nuisance": nuisance,
                "attenuation": attenuation,
            }
        )
        for scale in SCALES:
            if train_curves and tau > 0.0:
                kappa = attenuation * fit_kappa(cache, scale, train_curves, tau, nuisance)
            else:
                kappa = 0.0
            scored = score_target(cache, scale, target_curve, kappa, tau)
            rows.append(
                {
                    "mode": mode,
                    "scale": scale,
                    "target_curve": target_curve,
                    "target_label": target_label,
                    "route": route["route"],
                    "train_curves": "+".join(c.replace(".csv", "") for c in train_curves) if train_curves else "none",
                    "tau": tau,
                    "nuisance": nuisance,
                    "attenuation": attenuation,
                    "kappa": kappa,
                    **scored,
                }
            )
    return routes, rows


def variant_summary(mode: str, rows: list[dict[str, object]], role: str, reading: str) -> dict[str, object]:
    s = summarize(rows)
    return {
        "mode": mode,
        "role": role,
        "mean_delta": s["mean_delta"],
        "worst_delta": s["worst_delta"],
        "nonharm": f"{int(s['nonharm'])}/{int(s['rows'])}",
        "wins": f"{int(s['wins'])}/{int(s['rows'])}",
        "reading": reading,
    }


def route_degrees(routes: list[dict[str, object]]) -> dict[str, int]:
    nonzero_tau = {float(row["tau"]) for row in routes if float(row["tau"]) > 0.0}
    nuisances = {str(row["nuisance"]) for row in routes}
    train_sets = {str(row["train_curves"]) for row in routes}
    route_names = {str(row["route"]) for row in routes}
    return {
        "route_classes": len(route_names),
        "nonzero_tau_values": len(nonzero_tau),
        "nuisance_choices": len(nuisances),
        "source_sets": len(train_sets),
    }


def write_report(
    summaries: list[dict[str, object]],
    parameter_rows: list[dict[str, object]],
    route_rows: list[dict[str, object]],
) -> None:
    by_mode = {str(row["mode"]): row for row in summaries}
    minimal = by_mode["shape_routed_no_nuisance"]
    residualized = by_mode["shape_routed_residualized"]
    cross = by_mode["cross_family_residualized"]
    cross_min = by_mode["cross_family_no_nuisance_p3"]
    self_min = by_mode["self_fit_no_nuisance"]
    self_decomp = by_mode["decomposed_self_fit_reference"]
    lines = [
        "# Step-Time Model Complexity Audit\n\n",
        "This report answers which additions are necessary.  It treats fitted loss-dependent parameters separately from schedule-only routing choices.\n\n",
        "## Main Takeaway\n\n",
        "- The core transferable model does not need an interpretable sinusoidal component.  The one-kappa route, with nuisance projection disabled, still gives strong target-holdout generalization.\n",
        f"- Minimal shape-routed transfer (`L_MPL + kappa phi_tau`, no nuisance): mean `{fmt_pct(float(minimal['mean_delta']))}`, worst `{fmt_pct(float(minimal['worst_delta']))}`, non-harm `{minimal['nonharm']}`.\n",
        f"- Residualized shape-routed transfer: mean `{fmt_pct(float(residualized['mean_delta']))}`, worst `{fmt_pct(float(residualized['worst_delta']))}`, non-harm `{residualized['nonharm']}`.\n",
        f"- Conservative cross-family transfer: mean `{fmt_pct(float(cross['mean_delta']))}`, worst `{fmt_pct(float(cross['worst_delta']))}`, non-harm `{cross['nonharm']}`.\n",
        f"- Strict one-kappa/no-nuisance/no-same-family transfer remains useful but more conservative: mean `{fmt_pct(float(cross_min['mean_delta']))}`, worst `{fmt_pct(float(cross_min['worst_delta']))}`, non-harm `{cross_min['nonharm']}`.\n\n",
        "## Variant Table\n\n",
        "| variant | role | mean | worst | non-harm | wins | reading |\n",
        "|---|---|---:|---:|---:|---:|---|\n",
    ]
    for row in summaries:
        lines.append(
            f"| {row['mode']} | {row['role']} | {fmt_pct(float(row['mean_delta']))} | "
            f"{fmt_pct(float(row['worst_delta']))} | {row['nonharm']} | {row['wins']} | {row['reading']} |\n"
        )
    lines += [
        "\n## Parameter Ledger\n\n",
        "| component | fitted from loss? | count per calibration fit | should be headline? | explanation |\n",
        "|---|---|---:|---|---|\n",
    ]
    for row in parameter_rows:
        lines.append(
            f"| {row['component']} | {row['fitted_from_loss']} | {row['count_per_fit']} | "
            f"{row['headline']} | {row['explanation']} |\n"
        )
    lines += [
        "\n## Route Complexity\n\n",
        "| variant | route classes | nonzero tau values | nuisance choices | source sets |\n",
        "|---|---:|---:|---:|---:|\n",
    ]
    for row in route_rows:
        lines.append(
            f"| {row['mode']} | {row['route_classes']} | {row['nonzero_tau_values']} | "
            f"{row['nuisance_choices']} | {row['source_sets']} |\n"
        )
    lines += [
        "\n## Interpretation\n\n",
        f"- Self-fit improves from `{fmt_pct(float(self_min['mean_delta']))}` with one-kappa/no-nuisance to `{fmt_pct(float(self_decomp['mean_delta']))}` with the diagnostic low-frequency component.  That makes the nuisance useful for explanation, but not the main transferable mechanism.\n",
        "- For generalization, the cleanest primary claim is the minimal route: one fitted amplitude kappa, schedule-only tau/route choices, and no transferred nuisance coefficient.\n",
        "- The residualized and cross-family versions are best presented as audits: residualization checks that smooth MPL drift does not contaminate kappa; cross-family checks that the result is not just same-family calibration.\n",
        "- The schedule-only route and tau choices are still model-selection freedom, not learned parameters.  Their necessity is supported by ablations, but external frozen-rule validation is still the strongest missing evidence.\n",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_details: list[dict[str, object]] = []
    all_routes: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []

    shape_routes, shape_rows = eval_shape_route(mode="shape_routed_no_nuisance", nuisance_override="none")
    all_routes.extend(shape_routes)
    all_details.extend(shape_rows)
    summaries.append(
        variant_summary(
            "shape_routed_no_nuisance",
            shape_rows,
            "minimal target-holdout",
            "one fitted kappa; no nuisance coefficients used",
        )
    )

    _, shape_ref_rows, _ = run_shape_routed(CORE_CURVES, "shape_routed_residualized")
    all_details.extend(shape_ref_rows)
    summaries.append(
        variant_summary(
            "shape_routed_residualized",
            shape_ref_rows,
            "strong target-holdout",
            "adds projection to reduce smooth-drift contamination",
        )
    )

    self_routes, self_rows = eval_shape_route(
        mode="self_fit_no_nuisance",
        nuisance_override="none",
        self_fit=True,
    )
    all_routes.extend(self_routes)
    all_details.extend(self_rows)
    summaries.append(
        variant_summary(
            "self_fit_no_nuisance",
            self_rows,
            "minimal self-fit",
            "same target curve fits only kappa against phi_tau",
        )
    )

    decomp = ROOT / "results" / "step_time_decomposed_estimator" / "single_details.csv"
    with decomp.open(newline="", encoding="utf-8") as f:
        decomp_rows = [
            row
            for row in csv.DictReader(f)
            if row["train_curve"] == row["test_curve"]
        ]
    summaries.append(
        variant_summary(
            "decomposed_self_fit_reference",
            decomp_rows,
            "diagnostic self-fit",
            "uses fitted low-frequency nuisance for same-curve explanation",
        )
    )

    cross_routes, cross_rows, _ = run_cross_family(CORE_CURVES, "cross_family_residualized")
    all_routes.extend(cross_routes)
    all_details.extend(cross_rows)
    summaries.append(
        variant_summary(
            "cross_family_residualized",
            cross_rows,
            "no-same-family audit",
            "forbids calibration from the target schedule family",
        )
    )

    cross_min_routes, cross_min_rows = eval_cross_family_no_nuisance(attenuation_power=3.0)
    all_routes.extend(cross_min_routes)
    all_details.extend(cross_min_rows)
    summaries.append(
        variant_summary(
            "cross_family_no_nuisance_p3",
            cross_min_rows,
            "strict audit",
            "no same-family source, no nuisance projection, stronger drop attenuation",
        )
    )

    parameter_rows = [
        {
            "component": "kappa",
            "fitted_from_loss": "yes",
            "count_per_fit": 1,
            "headline": "yes",
            "explanation": "the only transferable amplitude fitted from calibration residuals",
        },
        {
            "component": "low-frequency nuisance coefficients",
            "fitted_from_loss": "yes",
            "count_per_fit": "0, 3, or 5 depending on audit",
            "headline": "no",
            "explanation": "residualization/self-fit diagnostic only; not a physical sinusoidal mechanism",
        },
        {
            "component": "tau / route / safety gate",
            "fitted_from_loss": "no",
            "count_per_fit": 0,
            "headline": "no",
            "explanation": "schedule-only model-selection rule; ablated separately because it can overfit benchmarks",
        },
        {
            "component": "drop attenuation",
            "fitted_from_loss": "no",
            "count_per_fit": 0,
            "headline": "no",
            "explanation": "schedule-only conservative shrinkage for weaker single-step targets",
        },
    ]

    route_rows = []
    for mode, routes in [
        ("shape_routed_no_nuisance", shape_routes),
        ("self_fit_no_nuisance", self_routes),
        ("cross_family_residualized", cross_routes),
        ("cross_family_no_nuisance_p3", cross_min_routes),
    ]:
        route_rows.append({"mode": mode, **route_degrees(routes)})

    write_csv(OUT_DIR / "variant_summary.csv", summaries)
    write_csv(OUT_DIR / "parameter_ledger.csv", parameter_rows)
    write_csv(OUT_DIR / "route_complexity.csv", route_rows)
    write_csv(OUT_DIR / "details.csv", all_details)
    write_report(summaries, parameter_rows, route_rows)
    print("step-time model complexity audit written")


if __name__ == "__main__":
    main()
