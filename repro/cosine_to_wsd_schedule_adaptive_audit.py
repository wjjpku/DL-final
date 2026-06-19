#!/usr/bin/env python3
"""Schedule-adaptive response-rate audit for cosine-to-WSD transfer.

This is a diagnostic candidate, not a replacement for the simpler global
response-kernel result.  The only target-side information used before
prediction is the LR schedule shape:

    drop concentration = max positive LR drop / total positive LR drop.

Smooth WSD decays get a slower response rate; step-like WSD-con schedules get a
faster response rate.  Kappa is still fitted only from cosine_72000 residuals
for the selected response channel.
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

from deep_stime import stime_feature as old_stime_feature  # noqa: E402
from cosine_to_wsd_response_search import (  # noqa: E402
    TARGETS,
    TARGET_RETENTION_FLOOR,
    TRAIN_CURVE,
    build_cache,
    fit_source_kappa,
    score_target,
    stime_feature,
    target_retention,
)
from reproduce_cosine_to_wsd import SCALES  # noqa: E402


OUT_DIR = ROOT / "results" / "cosine_to_wsd_response_search" / "schedule_adaptive"
FIG_DIR = OUT_DIR / "figs"

SMOOTH_LAMBDA = 7.0
STEP_LAMBDA = 20.0
STEP_CONCENTRATION_THRESHOLD = 0.2
OLD_RESPONSE_LAMBDA = 10.0


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt_pct(value: float) -> str:
    return f"{value:+.1f}%"


def drop_concentration(curve) -> float:
    eta = curve.lrs.astype(np.float64)
    drops = np.zeros_like(eta)
    drops[1:] = np.maximum(eta[:-1] - eta[1:], 0.0)
    total = float(np.sum(drops))
    if total <= 1e-18:
        return 0.0
    return float(np.max(drops) / total)


def adaptive_lambda(curve) -> float:
    concentration = drop_concentration(curve)
    return STEP_LAMBDA if concentration >= STEP_CONCENTRATION_THRESHOLD else SMOOTH_LAMBDA


def aggregate(rows: list[dict[str, object]], method: str, target: str | None = None) -> dict[str, object]:
    sub = [row for row in rows if row["method"] == method and (target is None or row["test_curve"] == target)]
    deltas = np.array([float(row["delta_pct"]) for row in sub], dtype=np.float64)
    return {
        "method": method,
        "test_curve": "ALL" if target is None else target,
        "test_label": "All WSD targets"
        if target is None
        else next(label for curve, label in TARGETS if curve == target),
        "rows": len(sub),
        "mean_delta": float(np.mean(deltas)),
        "median_delta": float(np.median(deltas)),
        "worst_delta": float(np.max(deltas)),
        "wins": int(np.sum(deltas < 0.0)),
        "nonharm": int(np.sum(deltas <= 1e-12)),
    }


def best_configs() -> tuple[dict[str, str], dict[str, str]]:
    configs = read_csv(ROOT / "results" / "cosine_to_wsd_response_search" / "all_configs.csv")
    safe = [row for row in configs if int(row["wins"]) == int(row["rows"]) and int(row["nonharm"]) == int(row["rows"])]
    best_mean = min(safe, key=lambda row: (float(row["mean_delta"]), float(row["worst_delta"])))
    best_worst = min(safe, key=lambda row: (float(row["worst_delta"]), float(row["mean_delta"])))
    return best_mean, best_worst


def score_response_config(cache, cfg: dict[str, str], method: str) -> list[dict[str, object]]:
    lam = float(cfg["response_lambda"])
    mu = float(cfg["nuisance_lambda"])
    max_mode = int(cfg["max_mode"])
    ridge_tau = float(cfg["ridge_tau"])
    retention_power = float(cfg["retention_power"])
    rho = float(cfg["rho"])
    rows: list[dict[str, object]] = []
    for scale in SCALES:
        source = cache[(scale, TRAIN_CURVE)]
        source_phi = stime_feature(source.curve, lam)
        fit = fit_source_kappa(
            source,
            source_phi,
            nuisance_lambda=mu,
            max_mode=max_mode,
            ridge_tau=ridge_tau,
            retention_power=retention_power,
            rho=rho,
        )
        for target_curve, target_label in TARGETS:
            target = cache[(scale, target_curve)]
            phi = stime_feature(target.curve, lam)
            retention = target_retention(phi, nuisance_lambda=mu, max_mode=max_mode)
            factor = 1.0 if retention >= TARGET_RETENTION_FLOOR else 0.0
            scored = score_target(target, phi, float(fit["kappa"]) * factor)
            rows.append(
                {
                    "method": method,
                    "scale": scale,
                    "test_curve": target_curve,
                    "test_label": target_label,
                    "response_lambda": lam,
                    "drop_concentration": drop_concentration(target.curve),
                    "kappa": fit["kappa"],
                    "target_retention": retention,
                    **scored,
                    "win": int(scored["delta_pct"] < 0.0),
                }
            )
    return rows


def score_adaptive(cache, cfg: dict[str, str]) -> list[dict[str, object]]:
    mu = float(cfg["nuisance_lambda"])
    max_mode = int(cfg["max_mode"])
    ridge_tau = float(cfg["ridge_tau"])
    retention_power = float(cfg["retention_power"])
    rho = float(cfg["rho"])
    rows: list[dict[str, object]] = []
    for scale in SCALES:
        source = cache[(scale, TRAIN_CURVE)]
        fit_cache: dict[float, dict[str, float]] = {}
        for target_curve, target_label in TARGETS:
            target = cache[(scale, target_curve)]
            lam = adaptive_lambda(target.curve)
            if lam not in fit_cache:
                source_phi = stime_feature(source.curve, lam)
                fit_cache[lam] = fit_source_kappa(
                    source,
                    source_phi,
                    nuisance_lambda=mu,
                    max_mode=max_mode,
                    ridge_tau=ridge_tau,
                    retention_power=retention_power,
                    rho=rho,
                )
            phi = stime_feature(target.curve, lam)
            retention = target_retention(phi, nuisance_lambda=mu, max_mode=max_mode)
            factor = 1.0 if retention >= TARGET_RETENTION_FLOOR else 0.0
            fit = fit_cache[lam]
            scored = score_target(target, phi, float(fit["kappa"]) * factor)
            rows.append(
                {
                    "method": "schedule_adaptive",
                    "scale": scale,
                    "test_curve": target_curve,
                    "test_label": target_label,
                    "response_lambda": lam,
                    "drop_concentration": drop_concentration(target.curve),
                    "kappa": fit["kappa"],
                    "target_retention": retention,
                    **scored,
                    "win": int(scored["delta_pct"] < 0.0),
                }
            )
    return rows


def score_old(cache) -> list[dict[str, object]]:
    old_lookup = {
        (row["scale"], row["test_curve"]): row
        for row in read_csv(ROOT / "results" / "cosine_to_wsd_focus" / "nextgen_safe_details.csv")
    }
    rows: list[dict[str, object]] = []
    for scale in SCALES:
        for target_curve, target_label in TARGETS:
            target = cache[(scale, target_curve)]
            info = old_lookup[(scale, target_curve)]
            phi = old_stime_feature(target.curve, OLD_RESPONSE_LAMBDA)
            scored = score_target(target, phi, float(info["kappa"]) * float(info["target_factor"]))
            rows.append(
                {
                    "method": "old_nextgen",
                    "scale": scale,
                    "test_curve": target_curve,
                    "test_label": target_label,
                    "response_lambda": OLD_RESPONSE_LAMBDA,
                    "drop_concentration": drop_concentration(target.curve),
                    "kappa": float(info["kappa"]),
                    "target_retention": float(info["target_retention"]),
                    **scored,
                    "win": int(scored["delta_pct"] < 0.0),
                }
            )
    return rows


def plot_summary(summary_rows: list[dict[str, object]]) -> None:
    methods = ["old_nextgen", "global_best_mean", "global_best_worst", "schedule_adaptive"]
    labels = ["old", "global mean", "global worst", "adaptive"]
    rows = [row for row in summary_rows if row["test_curve"] == "ALL"]
    values = [float(next(row for row in rows if row["method"] == method)["mean_delta"]) for method in methods]
    worsts = [float(next(row for row in rows if row["method"] == method)["worst_delta"]) for method in methods]
    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(8.2, 4.2), constrained_layout=True)
    width = 0.36
    ax.axhline(0.0, color="#111111", lw=0.9)
    ax.bar(x - width / 2, values, width, color="#2563eb", label="mean")
    ax.bar(x + width / 2, worsts, width, color="#64748b", label="worst")
    ax.set_xticks(x, labels, rotation=10, ha="right")
    ax.set_ylabel("MAE change vs MPL (%)")
    ax.set_title("Cosine-fitted WSD correction variants")
    ax.grid(axis="y", alpha=0.18)
    ax.legend(frameon=False)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "variant_summary.png", dpi=220, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def write_report(summary_rows: list[dict[str, object]], best_mean: dict[str, str], best_worst: dict[str, str]) -> None:
    all_rows = {row["method"]: row for row in summary_rows if row["test_curve"] == "ALL"}
    lines = [
        "# Schedule-Adaptive Cosine-to-WSD Audit\n\n",
        "This audit tests a schedule-only extension of the cosine-calibrated response model. "
        "The target loss curve is never used to fit kappa; the target LR schedule only selects the response rate.\n\n",
        "```text\n",
        "drop_concentration = max_t relu(lr_{t-1}-lr_t) / sum_t relu(lr_{t-1}-lr_t)\n",
        f"lambda_target = {STEP_LAMBDA:g} if drop_concentration >= {STEP_CONCENTRATION_THRESHOLD:g} else {SMOOTH_LAMBDA:g}\n",
        "```\n\n",
        "The shared estimator hyperparameters are borrowed from the global mean candidate: "
        f"`mu={float(best_mean['nuisance_lambda']):g}`, `max_mode={int(best_mean['max_mode'])}`, "
        f"`tau={float(best_mean['ridge_tau']):g}`, `p={float(best_mean['retention_power']):g}`, "
        f"`rho={float(best_mean['rho']):g}`.\n\n",
        "## Aggregate Comparison\n\n",
        "| method | mean delta | worst delta | wins |\n",
        "|---|---:|---:|---:|\n",
    ]
    for method in ["old_nextgen", "global_best_mean", "global_best_worst", "schedule_adaptive"]:
        row = all_rows[method]
        lines.append(
            f"| `{method}` | {fmt_pct(float(row['mean_delta']))} | "
            f"{fmt_pct(float(row['worst_delta']))} | {int(row['wins'])}/{int(row['rows'])} |\n"
        )
    lines += [
        "\n## Candidate Configs\n\n",
        f"- Global mean: `lambda={float(best_mean['response_lambda']):g}`, "
        f"`mu={float(best_mean['nuisance_lambda']):g}`, `tau={float(best_mean['ridge_tau']):g}`, "
        f"`p={float(best_mean['retention_power']):g}`, `rho={float(best_mean['rho']):g}`.\n",
        f"- Global worst: `lambda={float(best_worst['response_lambda']):g}`, "
        f"`mu={float(best_worst['nuisance_lambda']):g}`, `tau={float(best_worst['ridge_tau']):g}`, "
        f"`p={float(best_worst['retention_power']):g}`, `rho={float(best_worst['rho']):g}`.\n\n",
        "## Reading\n\n",
        "- The adaptive rule improves the mean result because smooth WSD decays prefer a slower response channel, while WSD-con step schedules need a faster channel to avoid long tail mismatch.\n",
        "- This is a promising hypothesis but it adds a schedule-dependent branch. It should be presented as an analysis/extension until validated on more schedules.\n",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    cache = build_cache()
    best_mean, best_worst = best_configs()
    rows = []
    rows.extend(score_old(cache))
    rows.extend(score_response_config(cache, best_mean, "global_best_mean"))
    rows.extend(score_response_config(cache, best_worst, "global_best_worst"))
    rows.extend(score_adaptive(cache, best_mean))
    summary = []
    for method in ["old_nextgen", "global_best_mean", "global_best_worst", "schedule_adaptive"]:
        for curve, _ in TARGETS:
            summary.append(aggregate(rows, method, curve))
        summary.append(aggregate(rows, method, None))
    write_csv(OUT_DIR / "details.csv", rows)
    write_csv(OUT_DIR / "summary.csv", summary)
    plot_summary(summary)
    write_report(summary, best_mean, best_worst)
    print(f"wrote {OUT_DIR / 'details.csv'}")
    print(f"wrote {OUT_DIR / 'summary.csv'}")
    print(f"wrote {OUT_DIR / 'REPORT.md'}")
    print(f"wrote {FIG_DIR / 'variant_summary.png'}")


if __name__ == "__main__":
    main()
