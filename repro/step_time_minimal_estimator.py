#!/usr/bin/env python3
"""Minimal one-kappa step-time estimator.

This is the clean candidate model after the complexity audit:

    L_hat(t) = L_MPL(t) + kappa * phi_tau(t)

No low-frequency nuisance coefficients are fitted or transferred.  The only
loss-fitted quantity is the nonnegative amplitude kappa on calibration curves.
Tau, source selection, and safety gates are schedule-only rules audited below.
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
from step_time_cross_family_estimator import route_for_target as cross_route_for_target  # noqa: E402
from step_time_decomposed_estimator import summarize  # noqa: E402
from step_time_shape_routed_estimator import (  # noqa: E402
    CORE_CURVES,
    EXTENDED_CURVES,
    build_cache,
    fit_kappa,
    naive_route_without_short_smooth_gate,
    route_for_target,
    schedule_stats,
    score_target,
)


OUT_DIR = ROOT / "results" / "step_time_minimal_estimator"
FIG_DIR = OUT_DIR / "figs"


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


def route_name(route: dict[str, object]) -> str:
    return str(route["route"])


def source_text(train_curves: tuple[str, ...]) -> str:
    return "+".join(c.replace(".csv", "") for c in train_curves) if train_curves else "none"


def score_routes(
    curve_defs: list[tuple[str, str]] | tuple[tuple[str, str], ...],
    *,
    mode: str,
    route_fn=route_for_target,
    force_tau: float | None = None,
    tau_multiplier: float = 1.0,
    attenuation_power: float | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    cache = build_cache(curve_defs)
    stats = schedule_stats(cache, curve_defs)
    routes: list[dict[str, object]] = []
    details: list[dict[str, object]] = []

    for target_curve, target_label in curve_defs:
        if attenuation_power is None:
            route = route_fn(stats, target_curve)
            attenuation = 1.0
        else:
            route = cross_route_for_target(stats, target_curve, attenuation_power=attenuation_power)
            attenuation = float(route["attenuation"])
        train_curves = tuple(route["train_curves"])
        raw_tau = float(route["tau"])
        tau = raw_tau
        if tau > 0.0:
            tau = float(force_tau) if force_tau is not None else raw_tau * tau_multiplier
        nuisance = "none"
        routes.append(
            {
                "mode": mode,
                "target_curve": target_curve,
                "target_label": target_label,
                "target_drop_norm": stats[target_curve]["drop_norm"],
                "target_decay_span": stats[target_curve]["decay_span"],
                "route": route_name(route),
                "train_curves": source_text(train_curves),
                "tau": tau,
                "base_tau": raw_tau,
                "nuisance": nuisance,
                "attenuation": attenuation,
                "rationale": route["rationale"],
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
                    "route": route_name(route),
                    "train_curves": source_text(train_curves),
                    "tau": tau,
                    "base_tau": raw_tau,
                    "nuisance": nuisance,
                    "attenuation": attenuation,
                    "source_kappa": source_kappa,
                    "kappa": kappa,
                    **scored,
                }
            )

    summary: list[dict[str, object]] = []
    for target_curve, target_label in curve_defs:
        sub = [row for row in details if row["target_curve"] == target_curve]
        summary.append({"target_curve": target_curve, "target_label": target_label, **summarize(sub)})
    return routes, details, summary


def ablation_audits() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    configs = [
        {
            "audit": "minimal_core",
            "curve_defs": CORE_CURVES,
            "route_fn": route_for_target,
            "force_tau": None,
            "tau_multiplier": 1.0,
            "attenuation_power": None,
            "description": "minimal one-kappa routed rule on core targets",
        },
        {
            "audit": "minimal_extended",
            "curve_defs": EXTENDED_CURVES,
            "route_fn": route_for_target,
            "force_tau": None,
            "tau_multiplier": 1.0,
            "attenuation_power": None,
            "description": "minimal rule plus short-cosine and constant controls",
        },
        {
            "audit": "fixed_tau_1024",
            "curve_defs": CORE_CURVES,
            "route_fn": route_for_target,
            "force_tau": 1024.0,
            "tau_multiplier": 1.0,
            "attenuation_power": None,
            "description": "one universal response time",
        },
        {
            "audit": "no_short_smooth_gate",
            "curve_defs": EXTENDED_CURVES,
            "route_fn": naive_route_without_short_smooth_gate,
            "force_tau": None,
            "tau_multiplier": 1.0,
            "attenuation_power": None,
            "description": "allows short smooth cosine to receive transient correction",
        },
        {
            "audit": "strict_cross_family_p3",
            "curve_defs": CORE_CURVES,
            "route_fn": route_for_target,
            "force_tau": None,
            "tau_multiplier": 1.0,
            "attenuation_power": 3.0,
            "description": "no same-family source, no nuisance, drop-cubed weak-step attenuation",
        },
        {
            "audit": "cross_family_no_attenuation",
            "curve_defs": CORE_CURVES,
            "route_fn": route_for_target,
            "force_tau": None,
            "tau_multiplier": 1.0,
            "attenuation_power": 0.0,
            "description": "no same-family source and no weak-step attenuation",
        },
    ]
    for mult in [0.5, 0.75, 1.25, 1.5, 2.0]:
        configs.append(
            {
                "audit": f"tau_x{mult:g}",
                "curve_defs": CORE_CURVES,
                "route_fn": route_for_target,
                "force_tau": None,
                "tau_multiplier": mult,
                "attenuation_power": None,
                "description": f"minimal route with all nonzero taus multiplied by {mult:g}",
            }
        )

    all_details: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for config in configs:
        _, details, _ = score_routes(
            config["curve_defs"],
            mode=str(config["audit"]),
            route_fn=config["route_fn"],
            force_tau=config["force_tau"],
            tau_multiplier=float(config["tau_multiplier"]),
            attenuation_power=config["attenuation_power"],
        )
        for row in details:
            row["audit"] = config["audit"]
            row["description"] = config["description"]
        all_details.extend(details)
        summary_rows.append(
            {
                "audit": config["audit"],
                "description": config["description"],
                **summarize(details),
            }
        )
    return summary_rows, all_details


def plot_summary(path: Path, summary: list[dict[str, object]]) -> None:
    labels = [
        str(row["target_label"])
        .replace("WSD sharp", "WSD\nsharp")
        .replace("WSD linear", "WSD\nlinear")
        .replace("WSD-con ", "con\n")
        for row in summary
    ]
    means = np.array([float(row["mean_delta"]) for row in summary])
    worst = np.array([float(row["worst_delta"]) for row in summary])
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10.8, 5.2))
    ax.axhline(0.0, color="#111111", lw=0.9)
    ax.bar(x, means, color="#2563eb", width=0.66, label="mean over scales")
    ax.scatter(x, worst, color="#dc2626", zorder=3, label="worst scale")
    ax.set_xticks(x, labels)
    ax.set_ylabel("MAE change vs MPL (%)")
    ax.set_title("Minimal one-kappa step-time target-holdout")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.18)
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def write_report(
    routes: list[dict[str, object]],
    details: list[dict[str, object]],
    summary: list[dict[str, object]],
    extended_details: list[dict[str, object]],
    ablation_summary: list[dict[str, object]],
) -> None:
    main = summarize(details)
    extended = summarize(extended_details)
    strict = next(row for row in ablation_summary if row["audit"] == "strict_cross_family_p3")
    fixed_tau = next(row for row in ablation_summary if row["audit"] == "fixed_tau_1024")
    no_gate = next(row for row in ablation_summary if row["audit"] == "no_short_smooth_gate")
    no_atten = next(row for row in ablation_summary if row["audit"] == "cross_family_no_attenuation")
    lines = [
        "# Minimal One-Kappa Step-Time Estimator\n\n",
        "This is the current recommended clean model candidate.  It keeps only the transferable transient response and removes all fitted low-frequency nuisance coefficients from the prediction rule.\n\n",
        "## Formula\n\n",
        "```text\n",
        "r(t) = L_true(t) - L_MPL(t)\n",
        "phi_tau(t) = sum_{u<=t} exp(-(t-u)/tau) * relu(eta_{u-1}-eta_u) / eta_peak\n",
        "kappa_hat = max(0, <phi_tau, r_source> / ||phi_tau||^2)\n",
        "L_hat_target(t) = L_MPL,target(t) + kappa_hat * phi_tau,target(t)\n",
        "```\n\n",
        "Only `kappa` is fitted from calibration loss residuals.  The route, tau, source set, and safety gates are schedule-only choices and are audited as model-selection freedom.\n\n",
        "## Main Result\n\n",
        f"- Minimal target-holdout: mean `{fmt_pct(float(main['mean_delta']))}`, worst `{fmt_pct(float(main['worst_delta']))}`, non-harm `{int(main['nonharm'])}/{int(main['rows'])}`.\n",
        f"- Extended controls: mean `{fmt_pct(float(extended['mean_delta']))}`, worst `{fmt_pct(float(extended['worst_delta']))}`, non-harm `{int(extended['nonharm'])}/{int(extended['rows'])}`.\n",
        f"- Strict no-same-family/no-nuisance audit: mean `{fmt_pct(float(strict['mean_delta']))}`, worst `{fmt_pct(float(strict['worst_delta']))}`, non-harm `{int(strict['nonharm'])}/{int(strict['rows'])}`.\n\n",
        "![minimal holdout](figs/minimal_target_holdout.png)\n\n",
        "## Route Table\n\n",
        "| target | drop | span | route | source | tau |\n",
        "|---|---:|---:|---|---|---:|\n",
    ]
    for row in routes:
        lines.append(
            f"| {row['target_label']} | {float(row['target_drop_norm']):.3f} | "
            f"{float(row['target_decay_span']):.0f} | {row['route']} | "
            f"`{row['train_curves']}` | {float(row['tau']):.0f} |\n"
        )
    lines += [
        "\n## Per-Target Summary\n\n",
        "| target | mean | worst | non-harm |\n",
        "|---|---:|---:|---:|\n",
    ]
    for row in summary:
        lines.append(
            f"| {row['target_label']} | {fmt_pct(float(row['mean_delta']))} | "
            f"{fmt_pct(float(row['worst_delta']))} | {int(row['nonharm'])}/{int(row['rows'])} |\n"
        )
    lines += [
        "\n## Necessity Tests\n\n",
        "| audit | mean | worst | non-harm | reading |\n",
        "|---|---:|---:|---:|---|\n",
    ]
    for row in ablation_summary:
        lines.append(
            f"| {row['audit']} | {fmt_pct(float(row['mean_delta']))} | "
            f"{fmt_pct(float(row['worst_delta']))} | {int(row['nonharm'])}/{int(row['rows'])} | "
            f"{row['description']} |\n"
        )
    lines += [
        "\n## Reading\n\n",
        "- The core improvement does not require a fitted sinusoidal or DCT nuisance component.\n",
        f"- A universal tau is insufficient: `fixed_tau_1024` reaches worst `{fmt_pct(float(fixed_tau['worst_delta']))}`.\n",
        f"- The short-smooth safety gate is necessary: removing it reaches worst `{fmt_pct(float(no_gate['worst_delta']))}`.\n",
        f"- Cross-family weak-step attenuation is necessary: removing it reaches worst `{fmt_pct(float(no_atten['worst_delta']))}`.\n",
        "- This is the preferred headline model if interpretability and overfit control are prioritized over the largest same-curve self-fit number.\n",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    routes, details, summary = score_routes(CORE_CURVES, mode="minimal_target_holdout")
    extended_routes, extended_details, extended_summary = score_routes(
        EXTENDED_CURVES,
        mode="minimal_extended_safety",
    )
    ablation_summary, ablation_details = ablation_audits()
    write_csv(OUT_DIR / "route_table.csv", routes)
    write_csv(OUT_DIR / "target_holdout_details.csv", details)
    write_csv(OUT_DIR / "target_holdout_summary.csv", summary)
    write_csv(OUT_DIR / "extended_route_table.csv", extended_routes)
    write_csv(OUT_DIR / "extended_safety_details.csv", extended_details)
    write_csv(OUT_DIR / "extended_safety_summary.csv", extended_summary)
    write_csv(OUT_DIR / "ablation_summary.csv", ablation_summary)
    write_csv(OUT_DIR / "ablation_details.csv", ablation_details)
    plot_summary(FIG_DIR / "minimal_target_holdout.png", summary)
    write_report(routes, details, summary, extended_details, ablation_summary)

    main_summary = summarize(details)
    extended_summary_stats = summarize(extended_details)
    print(f"wrote {OUT_DIR / 'REPORT.md'}")
    print(
        f"minimal={fmt_pct(float(main_summary['mean_delta']))}/"
        f"{fmt_pct(float(main_summary['worst_delta']))} "
        f"nonharm={int(main_summary['nonharm'])}/{int(main_summary['rows'])}"
    )
    print(
        f"extended={fmt_pct(float(extended_summary_stats['mean_delta']))}/"
        f"{fmt_pct(float(extended_summary_stats['worst_delta']))} "
        f"nonharm={int(extended_summary_stats['nonharm'])}/{int(extended_summary_stats['rows'])}"
    )


if __name__ == "__main__":
    main()
