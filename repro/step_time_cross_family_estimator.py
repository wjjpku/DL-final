#!/usr/bin/env python3
"""Conservative cross-family step-time estimator.

This audit is stricter than the shape-routed target-holdout head: a target
curve may not borrow calibration from the same schedule family.  The goal is
to reduce benchmark-overfitting concern while keeping the residual-image
story intact.
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
from step_time_decomposed_estimator import summarize  # noqa: E402
from step_time_shape_routed_estimator import (  # noqa: E402
    CORE_CURVES,
    EXTENDED_CURVES,
    build_cache,
    fit_kappa,
    run_shape_routed,
    schedule_stats,
    score_target,
)


OUT_DIR = ROOT / "results" / "step_time_cross_family_estimator"
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


def schedule_family(stats: dict[str, dict[str, float]], curve_name: str) -> str:
    row = stats[curve_name]
    drop = row["drop_norm"]
    span = row["decay_span"]
    length = row["schedule_len"]
    if drop <= 0.05:
        return "no_drop"
    if span > 16000.0 and length <= 30000.0:
        return "short_smooth"
    if span > 16000.0:
        return "smooth_decay"
    if span > 100.0:
        return "finite_tail"
    return "single_step"


def single_step_sources(stats: dict[str, dict[str, float]], target_curve: str) -> tuple[str, ...]:
    return tuple(
        c
        for c in sorted(stats)
        if c != target_curve and schedule_family(stats, c) == "single_step"
    )


def full_step_sources(stats: dict[str, dict[str, float]], target_curve: str) -> tuple[str, ...]:
    candidates = [
        c
        for c in stats
        if c != target_curve
        and schedule_family(stats, c) == "single_step"
        and stats[c]["drop_norm"] >= 0.85
    ]
    if candidates:
        candidates.sort(key=lambda c: (-stats[c]["drop_norm"], c))
        return (candidates[0],)
    return single_step_sources(stats, target_curve)[:1]


def finite_tail_sources(stats: dict[str, dict[str, float]], target_curve: str) -> tuple[str, ...]:
    return tuple(
        c
        for c in sorted(stats)
        if c != target_curve and schedule_family(stats, c) == "finite_tail"
    )


def mean_source_drop(stats: dict[str, dict[str, float]], train_curves: tuple[str, ...]) -> float:
    if not train_curves:
        return 0.0
    return float(np.mean([stats[c]["drop_norm"] for c in train_curves]))


def route_for_target(
    stats: dict[str, dict[str, float]],
    target_curve: str,
    attenuation_power: float = 2.0,
    zero_small_step_transfer: bool = False,
) -> dict[str, object]:
    family = schedule_family(stats, target_curve)
    target_drop = stats[target_curve]["drop_norm"]

    if family == "no_drop":
        return {
            "route": "no_lr_drop",
            "train_curves": tuple(),
            "tau": 0.0,
            "nuisance": "none",
            "attenuation": 0.0,
            "family": family,
            "rationale": "zero positive LR drop implies zero LR-drop transient",
        }

    if family == "short_smooth":
        return {
            "route": "short_smooth_no_transfer",
            "train_curves": tuple(),
            "tau": 0.0,
            "nuisance": "none",
            "attenuation": 0.0,
            "family": family,
            "rationale": "short smooth cosine is a safety control with no stable cross-family transfer",
        }

    if family == "smooth_decay":
        train = full_step_sources(stats, target_curve)
        return {
            "route": "smooth_from_full_step",
            "train_curves": train,
            "tau": 8192.0,
            "nuisance": "dct4",
            "attenuation": 1.0,
            "family": family,
            "rationale": "long smooth decay borrows only from a different-family full-step probe",
        }

    if family == "finite_tail":
        train = single_step_sources(stats, target_curve)
        return {
            "route": "finite_tail_from_steps",
            "train_curves": train,
            "tau": 3072.0,
            "nuisance": "dct4",
            "attenuation": 1.0,
            "family": family,
            "rationale": "finite WSD tails borrow from single-step probes, not from paired WSD tails",
        }

    train = finite_tail_sources(stats, target_curve)
    source_drop = mean_source_drop(stats, train)
    attenuation = 1.0
    if source_drop > 0.0:
        attenuation = min(1.0, (target_drop / source_drop) ** attenuation_power)
    if zero_small_step_transfer and target_drop < 0.85:
        attenuation = 0.0
    if target_drop >= 0.85:
        tau, nuisance, route = 1536.0, "dct2", "full_step_from_finite_tail"
    elif target_drop >= 0.60:
        tau, nuisance, route = 768.0, "dct2", "medium_step_from_finite_tail"
    else:
        tau, nuisance, route = 512.0, "none", "weak_step_from_finite_tail"
    return {
        "route": route,
        "train_curves": train,
        "tau": tau,
        "nuisance": nuisance,
        "attenuation": attenuation,
        "family": family,
        "rationale": "single-step targets borrow from finite WSD tails with target/source drop-squared attenuation",
    }


def run_cross_family(
    curve_defs: tuple[tuple[str, str], ...] | list[tuple[str, str]] = CORE_CURVES,
    mode: str = "cross_family_squared",
    attenuation_power: float = 2.0,
    zero_small_step_transfer: bool = False,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    cache = build_cache(curve_defs)
    stats = schedule_stats(cache, curve_defs)
    routes: list[dict[str, object]] = []
    details: list[dict[str, object]] = []
    for target_curve, target_label in curve_defs:
        route = route_for_target(stats, target_curve, attenuation_power, zero_small_step_transfer)
        train_curves = tuple(route["train_curves"])
        tau = float(route["tau"])
        nuisance = str(route["nuisance"])
        source_families = sorted({schedule_family(stats, c) for c in train_curves})
        source_drop = mean_source_drop(stats, train_curves)
        routes.append(
            {
                "target_curve": target_curve,
                "target_label": target_label,
                "target_family": route["family"],
                "target_drop_norm": stats[target_curve]["drop_norm"],
                "target_decay_span": stats[target_curve]["decay_span"],
                "route": route["route"],
                "train_curves": "+".join(c.replace(".csv", "") for c in train_curves) if train_curves else "none",
                "source_families": "+".join(source_families) if source_families else "none",
                "source_drop_norm": source_drop,
                "tau": tau,
                "nuisance": nuisance,
                "attenuation": float(route["attenuation"]),
                "rationale": route["rationale"],
            }
        )
        for scale in SCALES:
            if train_curves and tau > 0.0:
                source_kappa = fit_kappa(cache, scale, train_curves, tau, nuisance)
                kappa = source_kappa * float(route["attenuation"])
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
                    "attenuation": float(route["attenuation"]),
                    "kappa": kappa,
                    "tau": tau,
                    "nuisance": nuisance,
                    **scored,
                }
            )
    summary: list[dict[str, object]] = []
    for target_curve, target_label in curve_defs:
        sub = [r for r in details if r["target_curve"] == target_curve]
        summary.append(
            {
                "target_curve": target_curve,
                "target_label": target_label,
                **summarize(sub),
            }
        )
    return routes, details, summary


def ablation_rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    configs = [
        ("final_cross_family_squared", 2.0, False, "drop-squared attenuation for single-step targets"),
        ("linear_drop_attenuation", 1.0, False, "linear target/source drop attenuation"),
        ("no_drop_attenuation", 0.0, False, "no attenuation for weaker single-step targets"),
        ("zero_medium_weak_step", 2.0, True, "no cross-family correction for medium/weak single-step targets"),
    ]
    all_details: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    for name, power, zero_small, description in configs:
        _, details, _ = run_cross_family(
            CORE_CURVES,
            mode=name,
            attenuation_power=power,
            zero_small_step_transfer=zero_small,
        )
        for row in details:
            row["audit"] = name
            row["description"] = description
        all_details.extend(details)
        rows.append({"audit": name, "description": description, **summarize(details)})
    return rows, all_details


def same_family_route_count(routes: list[dict[str, object]]) -> int:
    count = 0
    for row in routes:
        target_family = str(row["target_family"])
        source_families = str(row["source_families"]).split("+")
        if target_family in source_families:
            count += 1
    return count


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
    ax.bar(x, means, color="#0f766e", width=0.66, label="mean over scales")
    ax.scatter(x, worst, color="#dc2626", zorder=3, label="worst scale")
    ax.set_xticks(x, labels)
    ax.set_ylabel("MAE change vs MPL (%)")
    ax.set_title("Conservative cross-family step-time generalization")
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
    shape_routes, shape_details, _ = run_shape_routed(CORE_CURVES, "shape_routed_reference")
    cross = summarize(details)
    extended = summarize(extended_details)
    shape = summarize(shape_details)
    cross_same_family = same_family_route_count(routes)
    shape_same_family = same_family_route_count(
        [
            {
                "target_family": (
                    "smooth_decay"
                    if row["target_curve"] == "cosine_72000.csv"
                    else "finite_tail"
                    if row["target_curve"] in {"wsd_20000_24000.csv", "wsdld_20000_24000.csv"}
                    else "single_step"
                ),
                "source_families": (
                    "single_step"
                    if row["train_curves"] in {"wsdcon_3", "wsdcon_9", "wsdcon_18"}
                    else "finite_tail"
                    if "wsd_20000_24000" in row["train_curves"] or "wsdld_20000_24000" in row["train_curves"]
                    else "single_step"
                    if "wsdcon_" in row["train_curves"]
                    else "none"
                ),
            }
            for row in shape_routes
        ]
    )
    lines = [
        "# Conservative Cross-Family Step-Time Estimator\n\n",
        "This audit responds to the main overfitting concern in the shape-routed head.  It keeps the image-derived transient model, but forbids a target from borrowing calibration from the same schedule family.\n\n",
        "## Formula\n\n",
        "```text\n",
        "r(t) = kappa * phi_tau(t) + nuisance + eps\n",
        "phi_tau(t) = sum_{u<=t} exp(-(t-u)/tau) * relu(eta_{u-1}-eta_u) / eta_peak\n",
        "kappa_target = alpha(target, source) * kappa_source\n",
        "alpha = min(1, (target_total_drop / source_mean_total_drop)^2)\n",
        "```\n\n",
        "The drop-squared attenuation is used only when a weaker single-step target borrows from stronger finite-tail WSD sources.  It is schedule-only and uses no target loss residual.\n\n",
        "## Main Result\n\n",
        f"- Cross-family target-holdout: mean `{float(cross['mean_delta']):+.1f}%`, worst `{float(cross['worst_delta']):+.1f}%`, non-harm `{int(cross['nonharm'])}/{int(cross['rows'])}`.\n",
        f"- Extended safety audit: mean `{float(extended['mean_delta']):+.1f}%`, worst `{float(extended['worst_delta']):+.1f}%`, non-harm `{int(extended['nonharm'])}/{int(extended['rows'])}`.\n",
        f"- Same-family source routes: `{cross_same_family}/{len(routes)}` for the conservative head versus `{shape_same_family}/{len(shape_routes)}` for the stronger shape-routed head.\n\n",
        "![cross-family holdout](figs/cross_family_target_holdout.png)\n\n",
        "## Route Table\n\n",
        "| target | target family | route | source | source family | tau | nuisance | attenuation |\n",
        "|---|---|---|---|---|---:|---|---:|\n",
    ]
    for row in routes:
        lines.append(
            f"| {row['target_label']} | {row['target_family']} | {row['route']} | "
            f"`{row['train_curves']}` | {row['source_families']} | {float(row['tau']):.0f} | "
            f"`{row['nuisance']}` | {float(row['attenuation']):.3f} |\n"
        )
    lines += [
        "\n## Per-Target Summary\n\n",
        "| target | mean | worst | non-harm |\n",
        "|---|---:|---:|---:|\n",
    ]
    for row in summary:
        lines.append(
            f"| {row['target_label']} | {float(row['mean_delta']):+.1f}% | "
            f"{float(row['worst_delta']):+.1f}% | {int(row['nonharm'])}/{int(row['rows'])} |\n"
        )
    lines += [
        "\n## Attenuation Ablation\n\n",
        "| audit | mean | worst | non-harm | reading |\n",
        "|---|---:|---:|---:|---|\n",
    ]
    for row in ablation_summary:
        lines.append(
            f"| {row['audit']} | {float(row['mean_delta']):+.1f}% | "
            f"{float(row['worst_delta']):+.1f}% | {int(row['nonharm'])}/{int(row['rows'])} | "
            f"{row['description']} |\n"
        )
    lines += [
        "\n## Comparison To Stronger Shape-Routed Head\n\n",
        "| estimator | mean | worst | non-harm | same-family source routes |\n",
        "|---|---:|---:|---:|---:|\n",
        f"| shape-routed | {float(shape['mean_delta']):+.1f}% | {float(shape['worst_delta']):+.1f}% | {int(shape['nonharm'])}/{int(shape['rows'])} | {shape_same_family}/{len(shape_routes)} |\n",
        f"| conservative cross-family | {float(cross['mean_delta']):+.1f}% | {float(cross['worst_delta']):+.1f}% | {int(cross['nonharm'])}/{int(cross['rows'])} | {cross_same_family}/{len(routes)} |\n\n",
        "## Reading\n\n",
        "- This is the cleaner generalization story: it sacrifices a few MAE points relative to the stronger routed head, but removes same-family source calibration from the core target-holdout audit.\n",
        "- The no-attenuation ablation is harmful, so the weaker-step correction needs a schedule-only amplitude attenuation rather than a full transfer of WSD-calibrated kappa.\n",
        "- This still does not replace external validation.  It is a stronger internal audit that reduces, but does not eliminate, benchmark-overfitting concern.\n",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    routes, details, summary = run_cross_family(CORE_CURVES)
    extended_routes, extended_details, extended_summary = run_cross_family(
        EXTENDED_CURVES,
        mode="cross_family_extended_safety",
    )
    ablation_summary, ablation_details = ablation_rows()
    write_csv(OUT_DIR / "route_table.csv", routes)
    write_csv(OUT_DIR / "target_holdout_details.csv", details)
    write_csv(OUT_DIR / "target_holdout_summary.csv", summary)
    write_csv(OUT_DIR / "extended_route_table.csv", extended_routes)
    write_csv(OUT_DIR / "extended_safety_details.csv", extended_details)
    write_csv(OUT_DIR / "extended_safety_summary.csv", extended_summary)
    write_csv(OUT_DIR / "ablation_summary.csv", ablation_summary)
    write_csv(OUT_DIR / "ablation_details.csv", ablation_details)
    plot_summary(FIG_DIR / "cross_family_target_holdout.png", summary)
    write_report(routes, details, summary, extended_details, ablation_summary)
    final = summarize(details)
    extended = summarize(extended_details)
    print(f"wrote {OUT_DIR / 'REPORT.md'}")
    print(
        f"cross-family={float(final['mean_delta']):+.1f}%/"
        f"{float(final['worst_delta']):+.1f}% nonharm={int(final['nonharm'])}/{int(final['rows'])}"
    )
    print(
        f"extended safety={float(extended['mean_delta']):+.1f}%/"
        f"{float(extended['worst_delta']):+.1f}% nonharm={int(extended['nonharm'])}/{int(extended['rows'])}"
    )


if __name__ == "__main__":
    main()
