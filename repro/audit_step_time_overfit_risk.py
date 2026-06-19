#!/usr/bin/env python3
"""Summarize overfitting risk for the step-time error model.

This is not a proof of external generalization.  It records the evidence that
reduces overfitting concern and the remaining reasons to keep the claim scoped.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results" / "step_time_shape_routed_estimator"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise AssertionError(f"missing CSV: {path}")
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, str]]) -> dict[str, object]:
    deltas = [float(row["delta_pct"]) for row in rows]
    return {
        "rows": len(rows),
        "mean_delta": sum(deltas) / len(deltas),
        "worst_delta": max(deltas),
        "nonharm": sum(delta <= 1e-10 for delta in deltas),
        "wins": sum(delta < 0.0 for delta in deltas),
    }


def curve_family(curve: str) -> str:
    if curve.startswith("cosine"):
        return "cosine"
    if curve.startswith("constant"):
        return "constant"
    if curve.startswith("wsd_") or curve.startswith("wsdld_"):
        return "finite_tail_wsd"
    if curve.startswith("wsdcon_"):
        return "single_step_probe"
    return "other"


def parse_sources(text: str) -> list[str]:
    if text == "none" or not text:
        return []
    return [part + ".csv" for part in text.split("+")]


def source_overlap_rows(routes: list[dict[str, str]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in routes:
        sources = parse_sources(row["train_curves"])
        target_family = curve_family(row["target_curve"])
        source_families = sorted({curve_family(source) for source in sources})
        rows.append(
            {
                "target_curve": row["target_curve"],
                "target_label": row["target_label"],
                "route": row["route"],
                "target_family": target_family,
                "source_families": "+".join(source_families) if source_families else "none",
                "uses_same_family_source": int(target_family in source_families),
                "source_count": len(sources),
            }
        )
    return rows


def row_by(rows: list[dict[str, str]], key: str, value: str) -> dict[str, str]:
    matches = [row for row in rows if row[key] == value]
    if len(matches) != 1:
        raise AssertionError(f"expected one {key}={value}, got {len(matches)}")
    return matches[0]


def fmt_pct(value: float) -> str:
    return f"{value:+.1f}%"


def main() -> None:
    details = read_csv(OUT_DIR / "target_holdout_details.csv")
    ablation = read_csv(OUT_DIR / "ablation_summary.csv")
    protocol = read_csv(OUT_DIR / "protocol_audit.csv")
    routes = read_csv(OUT_DIR / "route_table.csv")
    overlap = source_overlap_rows(routes)

    by_scale: list[dict[str, object]] = []
    for scale in sorted({row["scale"] for row in details}, key=lambda x: int(x)):
        sub = [row for row in details if row["scale"] == scale]
        by_scale.append({"slice": f"scale_{scale}", **summarize(sub)})

    by_route: list[dict[str, object]] = []
    for route in sorted({row["route"] for row in details}):
        sub = [row for row in details if row["route"] == route]
        by_route.append({"slice": f"route_{route}", **summarize(sub)})

    final_core = row_by(ablation, "audit", "final_core")
    no_nuisance = row_by(ablation, "audit", "no_nuisance_projection")
    fixed_tau = row_by(ablation, "audit", "fixed_tau_1024")
    no_gate = row_by(ablation, "audit", "no_short_smooth_gate")
    tau_075 = row_by(ablation, "audit", "tau_x0.75")
    tau_125 = row_by(ablation, "audit", "tau_x1.25")
    tau_15 = row_by(ablation, "audit", "tau_x1.5")
    tau_2 = row_by(ablation, "audit", "tau_x2")

    protocol_pass = sum(row["status"] == "pass" for row in protocol)
    target_excluded = sum(row["target_not_in_train"] == "1" for row in protocol)
    max_kappa_diff = max(float(row["kappa_abs_diff"]) for row in protocol)
    max_correction_diff = max(float(row["max_correction_abs_diff"]) for row in protocol)

    unique_routes = sorted({row["route"] for row in routes})
    unique_nonzero_taus = sorted({float(row["tau"]) for row in routes if float(row["tau"]) > 0.0})
    unique_nuisances = sorted({row["nuisance"] for row in routes})
    same_family_routes = sum(int(row["uses_same_family_source"]) for row in overlap)

    summary_rows: list[dict[str, object]] = [
        {
            "category": "guardrail",
            "metric": "target_loss_blindness",
            "value": f"{protocol_pass}/{len(protocol)} pass; max_delta_kappa={max_kappa_diff:.3e}; max_delta_correction={max_correction_diff:.3e}",
            "reading": "rules do not use target residual after being fixed",
        },
        {
            "category": "guardrail",
            "metric": "target_exclusion",
            "value": f"{target_excluded}/{len(protocol)}",
            "reading": "target curve is excluded from source calibration",
        },
        {
            "category": "robustness",
            "metric": "scale_slices",
            "value": "; ".join(
                f"{row['slice']} mean={fmt_pct(float(row['mean_delta']))} worst={fmt_pct(float(row['worst_delta']))} nonharm={row['nonharm']}/{row['rows']}"
                for row in by_scale
            ),
            "reading": "improvement is not carried by a single scale",
        },
        {
            "category": "robustness",
            "metric": "local_tau_window",
            "value": f"0.75x worst={fmt_pct(float(tau_075['worst_delta']))}, 1.25x worst={fmt_pct(float(tau_125['worst_delta']))}",
            "reading": "the selected tau values are locally stable",
        },
        {
            "category": "fragility",
            "metric": "wide_tau_window",
            "value": f"1.5x nonharm={tau_15['nonharm']}/{tau_15['rows']}, 2x nonharm={tau_2['nonharm']}/{tau_2['rows']}",
            "reading": "large tau changes break some cells, so tau is not arbitrary",
        },
        {
            "category": "fragility",
            "metric": "short_smooth_gate",
            "value": f"without gate worst={fmt_pct(float(no_gate['worst_delta']))}",
            "reading": "safety gate is necessary and must be presented as a limitation",
        },
        {
            "category": "fragility",
            "metric": "fixed_tau",
            "value": f"fixed_tau_1024 worst={fmt_pct(float(fixed_tau['worst_delta']))}",
            "reading": "one universal tau is insufficient on the current curves",
        },
        {
            "category": "complexity",
            "metric": "route_degrees_of_freedom",
            "value": f"{len(unique_routes)} route classes, {len(unique_nonzero_taus)} nonzero tau values, {len(unique_nuisances)} nuisance choices for 18 core target-scale cells",
            "reading": "too much to claim prospectively tuned external generalization",
        },
        {
            "category": "scope",
            "metric": "source_family_overlap",
            "value": f"{same_family_routes}/{len(overlap)} target routes use a same-family source",
            "reading": "target-holdout is stronger than self-fit but weaker than leave-family validation",
        },
        {
            "category": "scope",
            "metric": "claim_strength",
            "value": "internal public-curve target-holdout, not external prospective validation",
            "reading": "do not present the shape-routed head as fully proven on unseen training regimes",
        },
    ]

    write_csv(OUT_DIR / "overfit_risk_summary.csv", summary_rows)
    write_csv(OUT_DIR / "overfit_source_overlap.csv", overlap)

    lines = [
        "# Step-Time Overfit-Risk Audit\n\n",
        "Yes, the current estimator can still overfit the public-curve benchmark.  The protocol audit rules out target-residual leakage, but it does not prove that the route thresholds, tau values, and safety gates were selected prospectively.  This file separates what the evidence supports from what remains a risk.\n\n",
        "## What Reduces The Concern\n\n",
        f"- Target-loss blindness: `{protocol_pass}/{len(protocol)}` protocol rows pass; scrambling a target residual changes max `kappa` by `{max_kappa_diff:.3e}` and max correction by `{max_correction_diff:.3e}`.\n",
        f"- Target exclusion: `{target_excluded}/{len(protocol)}` predictions exclude the target curve from calibration.\n",
        f"- Final core target-holdout: mean `{fmt_pct(float(final_core['mean_delta']))}`, worst `{fmt_pct(float(final_core['worst_delta']))}`, non-harm `{final_core['nonharm']}/{final_core['rows']}`.\n",
        f"- Removing nuisance projection still stays non-harming: mean `{fmt_pct(float(no_nuisance['mean_delta']))}`, worst `{fmt_pct(float(no_nuisance['worst_delta']))}`.\n",
        f"- Local tau perturbations stay non-harming: `0.75x` worst `{fmt_pct(float(tau_075['worst_delta']))}`, `1.25x` worst `{fmt_pct(float(tau_125['worst_delta']))}`.\n\n",
        "## Scale Slices\n\n",
        "| slice | mean | worst | non-harm |\n",
        "|---|---:|---:|---:|\n",
    ]
    for row in by_scale:
        lines.append(
            f"| {row['slice']} | {fmt_pct(float(row['mean_delta']))} | "
            f"{fmt_pct(float(row['worst_delta']))} | {row['nonharm']}/{row['rows']} |\n"
        )
    lines += [
        "\n## Route Slices\n\n",
        "| slice | mean | worst | non-harm |\n",
        "|---|---:|---:|---:|\n",
    ]
    for row in by_route:
        lines.append(
            f"| {row['slice']} | {fmt_pct(float(row['mean_delta']))} | "
            f"{fmt_pct(float(row['worst_delta']))} | {row['nonharm']}/{row['rows']} |\n"
        )
    lines += [
        "\n## What Still Looks Like Overfitting Risk\n\n",
        f"- Model-selection complexity is high for the available benchmark: `{len(unique_routes)}` route classes, `{len(unique_nonzero_taus)}` nonzero tau values, and `{len(unique_nuisances)}` nuisance choices for only `18` core target-scale cells.\n",
        f"- Wide tau perturbations are not safe: `1.5x` gives `{tau_15['nonharm']}/{tau_15['rows']}` non-harming cells and `2x` gives `{tau_2['nonharm']}/{tau_2['rows']}`.\n",
        f"- The short-smooth safety gate is necessary: removing it gives worst `{fmt_pct(float(no_gate['worst_delta']))}`.\n",
        f"- `{same_family_routes}/{len(overlap)}` target routes use a same-family source.  This is target-holdout, not leave-family validation.\n",
        "- There is no new external schedule family in the current repository that was untouched during model design.\n\n",
        "## Recommended Claim\n\n",
        "Use the shape-routed estimator as the strongest current internal evidence and as a candidate deployment rule.  Do not claim that it is fully validated on unseen regimes.  The next decisive experiment is to freeze the route rule and evaluate it on a new schedule family or a new training run that was not used while designing the residual model.\n",
    ]
    (OUT_DIR / "OVERFIT_RISK_AUDIT.md").write_text("".join(lines), encoding="utf-8")
    print("step-time overfit risk audit written")


if __name__ == "__main__":
    main()
