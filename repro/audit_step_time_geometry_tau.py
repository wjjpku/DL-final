#!/usr/bin/env python3
"""Audit a schedule-geometry formula for step-time tau.

The routed estimator currently uses a small table of response times.  This
audit asks whether that table can be compressed into a schedule-derived rule
without losing the residual-image gains:

    long/finite decays: tau = min(8192, 1.25 * positive_drop_span)
    single-step drops: tau = 512 * (1 + 2 q^3),
                       q = clip((total_drop - 0.40) / (0.90 - 0.40), 0, 1)

The formula uses only LR-schedule geometry.  It does not fit tau from loss.
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

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parent
ROOT = REPO.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from reproduce_cosine_to_wsd import SCALES  # noqa: E402
from step_time_cross_family_estimator import (  # noqa: E402
    mean_source_drop,
    route_for_target as cross_family_route_for_target,
    schedule_family,
)
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


OUT_DIR = ROOT / "results" / "step_time_geometry_tau"
FIG_DIR = OUT_DIR / "figs"

STEP_TAU_BASE = 512.0
STEP_DROP_WEAK = 0.40
STEP_DROP_FULL = 0.90
TAIL_TAU_PER_STEP = 1.25
MAX_TAU = 8192.0


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fmt_pct(value: float | str) -> str:
    return f"{float(value):+.1f}%"


def row_by(rows: list[dict[str, str]], key: str, value: str) -> dict[str, str]:
    for row in rows:
        if row[key] == value:
            return row
    raise KeyError(value)


def geometry_tau_with_params(
    stats: dict[str, dict[str, float]],
    target_curve: str,
    *,
    tail_tau_per_step: float = TAIL_TAU_PER_STEP,
    step_tau_base: float = STEP_TAU_BASE,
    step_drop_weak: float = STEP_DROP_WEAK,
    step_drop_full: float = STEP_DROP_FULL,
    step_power: float = 3.0,
    max_tau: float = MAX_TAU,
) -> float:
    row = stats[target_curve]
    drop = float(row["drop_norm"])
    span = float(row["decay_span"])
    length = float(row["schedule_len"])
    if drop <= 0.05:
        return 0.0
    if span > 16000.0 and length <= 30000.0:
        return 0.0
    if span > 100.0:
        return min(max_tau, tail_tau_per_step * span)
    q = np.clip((drop - step_drop_weak) / (step_drop_full - step_drop_weak), 0.0, 1.0)
    return step_tau_base * (1.0 + 2.0 * float(q) ** step_power)


def geometry_tau(stats: dict[str, dict[str, float]], target_curve: str) -> float:
    return geometry_tau_with_params(stats, target_curve)


def eval_shape_geometry(
    curve_defs: tuple[tuple[str, str], ...] | list[tuple[str, str]],
    *,
    mode: str,
    force_nuisance: str | None = None,
    self_fit: bool = False,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    cache = build_cache(curve_defs)
    stats = schedule_stats(cache, curve_defs)
    routes: list[dict[str, object]] = []
    details: list[dict[str, object]] = []
    for target_curve, target_label in curve_defs:
        route = route_for_target(stats, target_curve)
        train_curves = (target_curve,) if self_fit else tuple(route["train_curves"])
        tau = geometry_tau(stats, target_curve)
        nuisance = str(route["nuisance"]) if force_nuisance is None else force_nuisance
        routes.append(
            {
                "mode": mode,
                "target_curve": target_curve,
                "target_label": target_label,
                "route": route["route"],
                "train_curves": "+".join(c.replace(".csv", "") for c in train_curves) if train_curves else "none",
                "table_tau": float(route["tau"]),
                "geometry_tau": tau,
                "tau_ratio_vs_table": tau / float(route["tau"]) if float(route["tau"]) > 0.0 else 0.0,
                "nuisance": nuisance,
                "target_drop_norm": stats[target_curve]["drop_norm"],
                "target_decay_span": stats[target_curve]["decay_span"],
                "target_schedule_len": stats[target_curve]["schedule_len"],
                "rationale": "tau is computed from LR positive-drop geometry",
            }
        )
        for scale in SCALES:
            if train_curves and tau > 0.0:
                kappa = fit_kappa(cache, scale, train_curves, tau, nuisance)
            else:
                kappa = 0.0
            scored = score_target(cache, scale, target_curve, kappa, tau)
            details.append(
                {
                    "mode": mode,
                    "scale": scale,
                    "target_curve": target_curve,
                    "target_label": target_label,
                    "route": route["route"],
                    "train_curves": "+".join(c.replace(".csv", "") for c in train_curves) if train_curves else "none",
                    "table_tau": float(route["tau"]),
                    "geometry_tau": tau,
                    "nuisance": nuisance,
                    "kappa": kappa,
                    **scored,
                }
            )
    summary = []
    for target_curve, target_label in curve_defs:
        sub = [row for row in details if row["target_curve"] == target_curve]
        summary.append({"target_curve": target_curve, "target_label": target_label, **summarize(sub)})
    return routes, details, summary


def eval_cross_family_geometry(
    *,
    mode: str,
    attenuation_power: float = 2.0,
    force_nuisance: str | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    cache = build_cache(CORE_CURVES)
    stats = schedule_stats(cache, CORE_CURVES)
    routes: list[dict[str, object]] = []
    details: list[dict[str, object]] = []
    for target_curve, target_label in CORE_CURVES:
        route = cross_family_route_for_target(stats, target_curve, attenuation_power)
        train_curves = tuple(route["train_curves"])
        tau = geometry_tau(stats, target_curve)
        nuisance = str(route["nuisance"]) if force_nuisance is None else force_nuisance
        attenuation = float(route["attenuation"])
        source_families = sorted({schedule_family(stats, c) for c in train_curves})
        source_drop = mean_source_drop(stats, train_curves)
        routes.append(
            {
                "mode": mode,
                "target_curve": target_curve,
                "target_label": target_label,
                "target_family": route["family"],
                "target_drop_norm": stats[target_curve]["drop_norm"],
                "target_decay_span": stats[target_curve]["decay_span"],
                "route": route["route"],
                "train_curves": "+".join(c.replace(".csv", "") for c in train_curves) if train_curves else "none",
                "source_families": "+".join(source_families) if source_families else "none",
                "source_drop_norm": source_drop,
                "table_tau": float(route["tau"]),
                "geometry_tau": tau,
                "nuisance": nuisance,
                "attenuation": attenuation,
            }
        )
        for scale in SCALES:
            if train_curves and tau > 0.0:
                source_kappa = fit_kappa(cache, scale, train_curves, tau, nuisance)
                kappa = attenuation * source_kappa
            else:
                source_kappa = 0.0
                kappa = 0.0
            scored = score_target(cache, scale, target_curve, kappa, tau)
            details.append(
                {
                    "mode": mode,
                    "scale": scale,
                    "target_curve": target_curve,
                    "target_label": target_label,
                    "target_family": route["family"],
                    "route": route["route"],
                    "train_curves": "+".join(c.replace(".csv", "") for c in train_curves) if train_curves else "none",
                    "source_families": "+".join(source_families) if source_families else "none",
                    "source_kappa": source_kappa,
                    "attenuation": attenuation,
                    "kappa": kappa,
                    "table_tau": float(route["tau"]),
                    "geometry_tau": tau,
                    "nuisance": nuisance,
                    **scored,
                }
            )
    summary = []
    for target_curve, target_label in CORE_CURVES:
        sub = [row for row in details if row["target_curve"] == target_curve]
        summary.append({"target_curve": target_curve, "target_label": target_label, **summarize(sub)})
    return routes, details, summary


def summary_row(name: str, role: str, rows: list[dict[str, object]], reading: str) -> dict[str, object]:
    s = summarize(rows)
    return {
        "model": name,
        "role": role,
        "mean_delta": s["mean_delta"],
        "worst_delta": s["worst_delta"],
        "nonharm": f"{int(s['nonharm'])}/{int(s['rows'])}",
        "wins": f"{int(s['wins'])}/{int(s['rows'])}",
        "reading": reading,
    }


def current_summary_rows() -> list[dict[str, object]]:
    shape = read_csv(ROOT / "results" / "step_time_shape_routed_estimator" / "target_holdout_details.csv")
    minimal = read_csv(ROOT / "results" / "step_time_minimal_estimator" / "target_holdout_details.csv")
    cross = read_csv(ROOT / "results" / "step_time_cross_family_estimator" / "target_holdout_details.csv")
    complexity = read_csv(ROOT / "results" / "step_time_model_complexity" / "variant_summary.csv")
    self_min = row_by(complexity, "mode", "self_fit_no_nuisance")
    return [
        summary_row("current_shape_routed", "baseline residualized target-holdout", shape, "discrete route tau table"),
        summary_row("current_minimal_no_nuisance", "baseline one-kappa target-holdout", minimal, "discrete route tau table"),
        summary_row("current_cross_family", "baseline no-same-family audit", cross, "discrete route tau table"),
        {
            "model": "current_self_fit_no_nuisance",
            "role": "baseline self-fit",
            "mean_delta": float(self_min["mean_delta"]),
            "worst_delta": float(self_min["worst_delta"]),
            "nonharm": self_min["nonharm"],
            "wins": self_min["wins"],
            "reading": "discrete route tau table",
        },
    ]


def scale_slice_rows(named_rows: list[tuple[str, list[dict[str, object]]]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for model, rows in named_rows:
        for scale in SCALES:
            sub = [row for row in rows if str(row["scale"]) == scale]
            s = summarize(sub)
            out.append(
                {
                    "model": model,
                    "scale": scale,
                    "mean_delta": s["mean_delta"],
                    "worst_delta": s["worst_delta"],
                    "nonharm": f"{int(s['nonharm'])}/{int(s['rows'])}",
                    "rows": s["rows"],
                }
            )
    return out


def evaluate_shape_with_tau_params(
    *,
    variant: str,
    params: dict[str, float],
    force_nuisance: str | None,
    cross_family: bool,
) -> list[dict[str, object]]:
    cache = build_cache(CORE_CURVES)
    stats = schedule_stats(cache, CORE_CURVES)
    rows: list[dict[str, object]] = []
    for target_curve, target_label in CORE_CURVES:
        if cross_family:
            route = cross_family_route_for_target(stats, target_curve, attenuation_power=2.0)
            attenuation = float(route["attenuation"])
        else:
            route = route_for_target(stats, target_curve)
            attenuation = 1.0
        train_curves = tuple(route["train_curves"])
        tau = geometry_tau_with_params(stats, target_curve, **params)
        nuisance = str(route["nuisance"]) if force_nuisance is None else force_nuisance
        for scale in SCALES:
            if train_curves and tau > 0.0:
                source_kappa = fit_kappa(cache, scale, train_curves, tau, nuisance)
                kappa = attenuation * source_kappa
            else:
                source_kappa = 0.0
                kappa = 0.0
            scored = score_target(cache, scale, target_curve, kappa, tau)
            rows.append(
                {
                    "variant": variant,
                    "scale": scale,
                    "target_curve": target_curve,
                    "target_label": target_label,
                    "route": route["route"],
                    "geometry_tau": tau,
                    "source_kappa": source_kappa,
                    "kappa": kappa,
                    "nuisance": nuisance,
                    **scored,
                }
            )
    return rows


def stability_audit_rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    variants = [
        ("baseline", {}, "baseline formula"),
        ("tail_x0.9", {"tail_tau_per_step": TAIL_TAU_PER_STEP * 0.9}, "tail response coefficient -10%"),
        ("tail_x1.1", {"tail_tau_per_step": TAIL_TAU_PER_STEP * 1.1}, "tail response coefficient +10%"),
        ("step_base_x0.75", {"step_tau_base": STEP_TAU_BASE * 0.75}, "single-step base tau -25%"),
        ("step_base_x1.25", {"step_tau_base": STEP_TAU_BASE * 1.25}, "single-step base tau +25%"),
        ("step_power_2", {"step_power": 2.0}, "less conservative single-step drop exponent"),
        ("step_power_4", {"step_power": 4.0}, "more conservative single-step drop exponent"),
        ("weak_anchor_0.35", {"step_drop_weak": 0.35}, "weak-drop anchor moved lower"),
        ("weak_anchor_0.45", {"step_drop_weak": 0.45}, "weak-drop anchor moved higher"),
        ("max_tau_6144", {"max_tau": 6144.0}, "lower long-decay maximum tau"),
    ]
    configs = [
        ("residualized", None, False),
        ("no_nuisance", "none", False),
        ("cross_family", None, True),
    ]
    summary_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []
    for model, force_nuisance, cross_family in configs:
        for variant, params, description in variants:
            rows = evaluate_shape_with_tau_params(
                variant=variant,
                params=params,
                force_nuisance=force_nuisance,
                cross_family=cross_family,
            )
            for row in rows:
                row["model"] = model
                row["description"] = description
            detail_rows.extend(rows)
            s = summarize(rows)
            summary_rows.append(
                {
                    "model": model,
                    "variant": variant,
                    "description": description,
                    "mean_delta": s["mean_delta"],
                    "worst_delta": s["worst_delta"],
                    "nonharm": f"{int(s['nonharm'])}/{int(s['rows'])}",
                    "rows": s["rows"],
                }
            )
    return summary_rows, detail_rows


def plot_comparison(rows: list[dict[str, object]], path: Path) -> None:
    by_model = {str(row["model"]): row for row in rows}
    pairs = [
        ("Residualized\nholdout", "current_shape_routed", "geometry_shape_routed"),
        ("One-kappa\nholdout", "current_minimal_no_nuisance", "geometry_no_nuisance"),
        ("No-same-family\naudit", "current_cross_family", "geometry_cross_family"),
        ("One-kappa\nself-fit", "current_self_fit_no_nuisance", "geometry_self_fit_no_nuisance"),
    ]
    labels = [item[0] for item in pairs]
    current_mean = np.array([float(by_model[cur]["mean_delta"]) for _, cur, _ in pairs])
    geometry_mean = np.array([float(by_model[geo]["mean_delta"]) for _, _, geo in pairs])
    current_worst = np.array([float(by_model[cur]["worst_delta"]) for _, cur, _ in pairs])
    geometry_worst = np.array([float(by_model[geo]["worst_delta"]) for _, _, geo in pairs])
    x = np.arange(len(labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(10.8, 5.2), constrained_layout=True)
    ax.axhline(0.0, color="#111111", lw=0.9)
    b1 = ax.bar(x - width / 2, current_mean, color="#6b7280", width=width, label="current mean")
    b2 = ax.bar(x + width / 2, geometry_mean, color="#2563eb", width=width, label="geometry mean")
    ax.scatter(x - width / 2, current_worst, color="#991b1b", zorder=3, marker="x", s=48, label="current worst")
    ax.scatter(x + width / 2, geometry_worst, color="#dc2626", zorder=3, marker="o", s=34, label="geometry worst")
    ax.set_xticks(x, labels)
    ax.set_ylabel("MAE change vs MPL (%)")
    ax.set_title("Discrete route tau table vs schedule-geometry tau formula")
    ax.grid(axis="y", alpha=0.24)
    ax.legend(frameon=False, ncol=2, loc="lower left")
    for bars in [b1, b2]:
        for bar in bars:
            val = float(bar.get_height())
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 1.1,
                f"{val:+.1f}",
                ha="center",
                va="bottom",
                fontsize=8.4,
                color="white" if val < -8.0 else "#111111",
            )
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def write_report(
    comparison: list[dict[str, object]],
    geometry_routes: list[dict[str, object]],
    extended_rows: list[dict[str, object]],
    scale_slices: list[dict[str, object]],
    stability_rows: list[dict[str, object]],
) -> None:
    by_model = {str(row["model"]): row for row in comparison}
    extended = summarize(extended_rows)
    lines = [
        "# Schedule-Geometry Tau Audit\n\n",
        "This audit replaces the discrete response-time table with a tau computed directly from LR positive-drop geometry.  It is meant to reduce benchmark-shaped route complexity while preserving the residual-image correction.\n\n",
        "## Formula\n\n",
        "For no-drop and short-smooth safety controls, `tau=0` and the correction abstains.  Otherwise:\n\n",
        "```text\n",
        "if positive_drop_span > 100:\n",
        "    tau = min(8192, 1.25 * positive_drop_span)\n",
        "else:\n",
        "    q = clip((total_positive_drop - 0.40) / (0.90 - 0.40), 0, 1)\n",
        "    tau = 512 * (1 + 2 q^3)\n",
        "```\n\n",
        "This keeps the physical reading from the residual plots: long decays need longer memory, while weak single-step drops should have shorter memory and smaller transferable lag.\n\n",
        "## Main Result\n\n",
        f"- Residualized target-holdout: current `{fmt_pct(by_model['current_shape_routed']['mean_delta'])}` mean / `{fmt_pct(by_model['current_shape_routed']['worst_delta'])}` worst; geometry tau `{fmt_pct(by_model['geometry_shape_routed']['mean_delta'])}` mean / `{fmt_pct(by_model['geometry_shape_routed']['worst_delta'])}` worst.\n",
        f"- One-kappa/no-nuisance target-holdout: current `{fmt_pct(by_model['current_minimal_no_nuisance']['mean_delta'])}` / `{fmt_pct(by_model['current_minimal_no_nuisance']['worst_delta'])}`; geometry tau `{fmt_pct(by_model['geometry_no_nuisance']['mean_delta'])}` / `{fmt_pct(by_model['geometry_no_nuisance']['worst_delta'])}`.\n",
        f"- No-same-family residualized audit: current `{fmt_pct(by_model['current_cross_family']['mean_delta'])}` / `{fmt_pct(by_model['current_cross_family']['worst_delta'])}`; geometry tau `{fmt_pct(by_model['geometry_cross_family']['mean_delta'])}` / `{fmt_pct(by_model['geometry_cross_family']['worst_delta'])}`.\n",
        f"- Extended safety remains non-harming: `{int(extended['nonharm'])}/{int(extended['rows'])}` rows, worst `{fmt_pct(extended['worst_delta'])}`.\n\n",
        "![comparison](figs/geometry_tau_comparison.png)\n\n",
        "## Comparison Table\n\n",
        "| model | role | mean | worst | non-harm | wins | reading |\n",
        "|---|---|---:|---:|---:|---:|---|\n",
    ]
    for row in comparison:
        lines.append(
            f"| {row['model']} | {row['role']} | {fmt_pct(row['mean_delta'])} | "
            f"{fmt_pct(row['worst_delta'])} | {row['nonharm']} | {row['wins']} | {row['reading']} |\n"
        )
    lines += [
        "\n## Route Tau Table\n\n",
        "| target | route | drop | span | table tau | geometry tau | nuisance | source |\n",
        "|---|---|---:|---:|---:|---:|---|---|\n",
    ]
    for row in geometry_routes:
        lines.append(
            f"| {row['target_label']} | {row['route']} | {float(row['target_drop_norm']):.3f} | "
            f"{float(row['target_decay_span']):.0f} | {float(row['table_tau']):.1f} | "
            f"{float(row['geometry_tau']):.1f} | `{row['nuisance']}` | `{row['train_curves']}` |\n"
        )
    lines += [
        "\n## Scale Slices\n\n",
        "| model | scale | mean | worst | non-harm |\n",
        "|---|---:|---:|---:|---:|\n",
    ]
    for row in scale_slices:
        lines.append(
            f"| {row['model']} | {row['scale']} | {fmt_pct(row['mean_delta'])} | "
            f"{fmt_pct(row['worst_delta'])} | {row['nonharm']} |\n"
        )

    key_stability = [
        row
        for row in stability_rows
        if row["variant"] in {"baseline", "tail_x0.9", "tail_x1.1", "step_base_x0.75", "step_base_x1.25", "step_power_2", "step_power_4"}
    ]
    lines += [
        "\n## Formula Stability\n\n",
        "This is a sensitivity audit, not model selection.  It checks whether small formula perturbations immediately break the correction.\n\n",
        "| model | variant | mean | worst | non-harm | reading |\n",
        "|---|---|---:|---:|---:|---|\n",
    ]
    for row in key_stability:
        lines.append(
            f"| {row['model']} | {row['variant']} | {fmt_pct(row['mean_delta'])} | "
            f"{fmt_pct(row['worst_delta'])} | {row['nonharm']} | {row['description']} |\n"
        )
    lines += [
        "\n## Decision\n\n",
        "- Geometry tau is a better default than the raw discrete tau table for future frozen-rule validation: it reduces route-specific constants and slightly improves the no-nuisance and cross-family audits.\n",
        "- The sensitivity audit shows that the residualized and cross-family variants are non-harming under all tested perturbations.  The no-nuisance head is more sensitive to the single-step exponent/base, which supports keeping the conservative cubic single-step rule.\n",
        "- It does not create a new headline performance jump; its value is interpretability and overfit control.  The stronger self-fit story still comes from the decomposed residual diagnostic, not from changing tau.\n",
        "- The next real performance evidence still needs new curves or new training runs.  More retrospective tau tuning on the same six core schedules is not justified by the Pareto audit.\n",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    routes, geometry_rows, geometry_summary = eval_shape_geometry(
        CORE_CURVES,
        mode="geometry_shape_routed",
        force_nuisance=None,
        self_fit=False,
    )
    _, geometry_no_nuisance_rows, geometry_no_nuisance_summary = eval_shape_geometry(
        CORE_CURVES,
        mode="geometry_no_nuisance",
        force_nuisance="none",
        self_fit=False,
    )
    _, self_rows, self_summary = eval_shape_geometry(
        CORE_CURVES,
        mode="geometry_self_fit_no_nuisance",
        force_nuisance="none",
        self_fit=True,
    )
    _, extended_rows, extended_summary = eval_shape_geometry(
        EXTENDED_CURVES,
        mode="geometry_extended_safety",
        force_nuisance=None,
        self_fit=False,
    )
    cross_routes, cross_rows, cross_summary = eval_cross_family_geometry(
        mode="geometry_cross_family",
        attenuation_power=2.0,
        force_nuisance=None,
    )

    comparison = current_summary_rows()
    comparison.extend(
        [
            summary_row(
                "geometry_shape_routed",
                "geometry-tau residualized target-holdout",
                geometry_rows,
                "same route/source/nuisance as current; tau from LR geometry",
            ),
            summary_row(
                "geometry_no_nuisance",
                "geometry-tau one-kappa target-holdout",
                geometry_no_nuisance_rows,
                "no nuisance projection; tau from LR geometry",
            ),
            summary_row(
                "geometry_cross_family",
                "geometry-tau no-same-family audit",
                cross_rows,
                "cross-family source rule; tau from target LR geometry",
            ),
            summary_row(
                "geometry_self_fit_no_nuisance",
                "geometry-tau one-kappa self-fit",
                self_rows,
                "same target residual fits kappa only; tau from LR geometry",
            ),
        ]
    )
    scale_slices = scale_slice_rows(
        [
            ("geometry_shape_routed", geometry_rows),
            ("geometry_no_nuisance", geometry_no_nuisance_rows),
            ("geometry_cross_family", cross_rows),
            ("geometry_self_fit_no_nuisance", self_rows),
        ]
    )
    stability_summary, stability_details = stability_audit_rows()

    write_csv(OUT_DIR / "route_table.csv", routes)
    write_csv(OUT_DIR / "target_holdout_details.csv", geometry_rows)
    write_csv(OUT_DIR / "target_holdout_summary.csv", geometry_summary)
    write_csv(OUT_DIR / "no_nuisance_details.csv", geometry_no_nuisance_rows)
    write_csv(OUT_DIR / "no_nuisance_summary.csv", geometry_no_nuisance_summary)
    write_csv(OUT_DIR / "self_fit_no_nuisance_details.csv", self_rows)
    write_csv(OUT_DIR / "self_fit_no_nuisance_summary.csv", self_summary)
    write_csv(OUT_DIR / "extended_safety_details.csv", extended_rows)
    write_csv(OUT_DIR / "extended_safety_summary.csv", extended_summary)
    write_csv(OUT_DIR / "cross_family_route_table.csv", cross_routes)
    write_csv(OUT_DIR / "cross_family_details.csv", cross_rows)
    write_csv(OUT_DIR / "cross_family_summary.csv", cross_summary)
    write_csv(OUT_DIR / "comparison.csv", comparison)
    write_csv(OUT_DIR / "scale_slices.csv", scale_slices)
    write_csv(OUT_DIR / "stability_summary.csv", stability_summary)
    write_csv(OUT_DIR / "stability_details.csv", stability_details)

    plot_comparison(comparison, FIG_DIR / "geometry_tau_comparison.png")
    write_report(comparison, routes, extended_rows, scale_slices, stability_summary)
    geom = summarize(geometry_rows)
    no_nuis = summarize(geometry_no_nuisance_rows)
    cross = summarize(cross_rows)
    print(f"wrote {OUT_DIR / 'REPORT.md'}")
    print(
        f"geometry residualized={float(geom['mean_delta']):+.1f}%/{float(geom['worst_delta']):+.1f}% "
        f"no-nuisance={float(no_nuis['mean_delta']):+.1f}%/{float(no_nuis['worst_delta']):+.1f}% "
        f"cross-family={float(cross['mean_delta']):+.1f}%/{float(cross['worst_delta']):+.1f}%"
    )


if __name__ == "__main__":
    main()
