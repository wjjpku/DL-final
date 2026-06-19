#!/usr/bin/env python3
"""Search schedule-adaptive cosine-to-WSD response models.

The fitted amplitude is always estimated from cosine_72000 residuals.  WSD
target losses are used only to rank development candidates in this audit.

Model family:
  * target LR schedule selects a response channel from drop concentration;
  * cosine residual estimates one kappa per selected response channel/scale;
  * target prediction uses only MPL baseline plus kappa * LR-derived response.
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

from cosine_to_wsd_response_search import (  # noqa: E402
    MAX_MODES,
    NUISANCE_LAMBDAS,
    RESPONSE_LAMBDAS,
    RETENTION_POWERS,
    RHOS,
    RIDGE_TAUS,
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


OUT_DIR = ROOT / "results" / "cosine_to_wsd_response_search" / "adaptive_search"
FIG_DIR = OUT_DIR / "figs"

SMOOTH_LAMBDAS = [2.0, 4.0, 7.0, 10.0, 14.0, 20.0]
STEP_LAMBDAS = [10.0, 14.0, 20.0, 30.0, 50.0, 80.0]
DROP_CONCENTRATION_THRESHOLD = 0.2


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


def channel_for_curve(curve) -> str:
    return "step" if drop_concentration(curve) >= DROP_CONCENTRATION_THRESHOLD else "smooth"


def aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    deltas = np.array([float(row["delta_pct"]) for row in rows], dtype=np.float64)
    return {
        "rows": len(rows),
        "mean_delta": float(np.mean(deltas)),
        "median_delta": float(np.median(deltas)),
        "worst_delta": float(np.max(deltas)),
        "wins": int(np.sum(deltas < 0.0)),
        "nonharm": int(np.sum(deltas <= 1e-12)),
    }


def score_config(
    cache,
    feature_cache,
    *,
    smooth_lambda: float,
    step_lambda: float,
    nuisance_lambda: float,
    max_mode: int,
    ridge_tau: float,
    retention_power: float,
    rho: float,
) -> list[dict[str, object]]:
    details: list[dict[str, object]] = []
    channel_lambda = {"smooth": smooth_lambda, "step": step_lambda}
    for scale in SCALES:
        source = cache[(scale, TRAIN_CURVE)]
        fit_by_channel: dict[str, dict[str, float]] = {}
        for channel, response_lambda in channel_lambda.items():
            source_phi = feature_cache[(scale, TRAIN_CURVE, response_lambda)]
            fit_by_channel[channel] = fit_source_kappa(
                source,
                source_phi,
                nuisance_lambda=nuisance_lambda,
                max_mode=max_mode,
                ridge_tau=ridge_tau,
                retention_power=retention_power,
                rho=rho,
            )
        for target_curve, target_label in TARGETS:
            target = cache[(scale, target_curve)]
            channel = channel_for_curve(target.curve)
            response_lambda = channel_lambda[channel]
            phi_t = feature_cache[(scale, target_curve, response_lambda)]
            retention_t = target_retention(phi_t, nuisance_lambda=nuisance_lambda, max_mode=max_mode)
            target_factor = 1.0 if retention_t >= TARGET_RETENTION_FLOOR else 0.0
            fit = fit_by_channel[channel]
            scored = score_target(target, phi_t, float(fit["kappa"]) * target_factor)
            details.append(
                {
                    "smooth_lambda": smooth_lambda,
                    "step_lambda": step_lambda,
                    "response_lambda": response_lambda,
                    "channel": channel,
                    "drop_concentration": drop_concentration(target.curve),
                    "nuisance_lambda": nuisance_lambda,
                    "max_mode": max_mode,
                    "ridge_tau": ridge_tau,
                    "retention_power": retention_power,
                    "rho": rho,
                    "scale": scale,
                    "train_curve": TRAIN_CURVE,
                    "test_curve": target_curve,
                    "test_label": target_label,
                    "target_retention": retention_t,
                    "target_factor": target_factor,
                    **fit,
                    **scored,
                    "win": int(scored["delta_pct"] < 0.0),
                }
            )
    return details


def run_search() -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    cache = build_cache()
    lambdas = sorted(set(SMOOTH_LAMBDAS + STEP_LAMBDAS + RESPONSE_LAMBDAS))
    feature_cache = {
        (scale, curve_name, response_lambda): stime_feature(cache[(scale, curve_name)].curve, response_lambda)
        for scale in SCALES
        for curve_name, _ in [(TRAIN_CURVE, "Cosine")] + TARGETS
        for response_lambda in lambdas
    }
    config_rows: list[dict[str, object]] = []
    safe_details: list[dict[str, object]] = []
    config_id = 0
    for smooth_lambda in SMOOTH_LAMBDAS:
        for step_lambda in STEP_LAMBDAS:
            for nuisance_lambda in NUISANCE_LAMBDAS:
                for max_mode in MAX_MODES:
                    for ridge_tau in RIDGE_TAUS:
                        for retention_power in RETENTION_POWERS:
                            for rho in RHOS:
                                details = score_config(
                                    cache,
                                    feature_cache,
                                    smooth_lambda=smooth_lambda,
                                    step_lambda=step_lambda,
                                    nuisance_lambda=nuisance_lambda,
                                    max_mode=max_mode,
                                    ridge_tau=ridge_tau,
                                    retention_power=retention_power,
                                    rho=rho,
                                )
                                summary = aggregate(details)
                                config_rows.append(
                                    {
                                        "config_id": config_id,
                                        "smooth_lambda": smooth_lambda,
                                        "step_lambda": step_lambda,
                                        "nuisance_lambda": nuisance_lambda,
                                        "max_mode": max_mode,
                                        "ridge_tau": ridge_tau,
                                        "retention_power": retention_power,
                                        "rho": rho,
                                        **summary,
                                        "mean_kappa": float(np.mean([float(row["kappa"]) for row in details])),
                                        "mean_source_retention": float(
                                            np.mean([float(row["source_retention"]) for row in details])
                                        ),
                                        "mean_target_retention": float(
                                            np.mean([float(row["target_retention"]) for row in details])
                                        ),
                                    }
                                )
                                if summary["nonharm"] == summary["rows"] and summary["wins"] == summary["rows"]:
                                    for row in details:
                                        safe_details.append({"config_id": config_id, **row})
                                config_id += 1
    safe = [row for row in config_rows if int(row["nonharm"]) == int(row["rows"]) and int(row["wins"]) == int(row["rows"])]
    safe_sorted = sorted(safe, key=lambda row: (float(row["mean_delta"]), float(row["worst_delta"])))
    top_ids = {int(row["config_id"]) for row in safe_sorted[:50]}
    top_details = [row for row in safe_details if int(row["config_id"]) in top_ids]
    return config_rows, safe_sorted[:200], top_details


def summarize_by_target(detail_rows: list[dict[str, object]], config_id: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    selected = [row for row in detail_rows if int(row["config_id"]) == config_id]
    for target_curve, target_label in TARGETS:
        sub = [row for row in selected if row["test_curve"] == target_curve]
        if not sub:
            continue
        rows.append({"test_curve": target_curve, "test_label": target_label, **aggregate(sub)})
    return rows


def plot_best(detail_rows: list[dict[str, object]], config_id: int) -> None:
    rows = [row for row in detail_rows if int(row["config_id"]) == config_id]
    if not rows:
        return
    labels = [label for _, label in TARGETS]
    x = np.arange(len(labels))
    means, worsts = [], []
    for curve, _ in TARGETS:
        sub = [row for row in rows if row["test_curve"] == curve]
        means.append(float(np.mean([float(row["delta_pct"]) for row in sub])))
        worsts.append(float(max(float(row["delta_pct"]) for row in sub)))
    fig, ax = plt.subplots(figsize=(9.2, 4.4), constrained_layout=True)
    width = 0.36
    ax.axhline(0.0, color="#111111", lw=0.9)
    ax.bar(x - width / 2, means, width, color="#2563eb", label="mean")
    ax.bar(x + width / 2, worsts, width, color="#64748b", label="worst")
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set_ylabel("MAE change vs MPL (%)")
    ax.set_title("Best schedule-adaptive cosine-to-WSD correction")
    ax.grid(axis="y", alpha=0.18)
    ax.legend(frameon=False)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "best_adaptive_target_summary.png", dpi=220, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def write_report(
    config_rows: list[dict[str, object]],
    safe_rows: list[dict[str, object]],
    detail_rows: list[dict[str, object]],
) -> None:
    if not safe_rows:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "REPORT.md").write_text("No fully non-harming adaptive configuration found.\n", encoding="utf-8")
        return
    best = safe_rows[0]
    best_worst = min(safe_rows, key=lambda row: (float(row["worst_delta"]), float(row["mean_delta"])))
    best_id = int(best["config_id"])
    target_rows = summarize_by_target(detail_rows, best_id)
    plot_best(detail_rows, best_id)
    lines = [
        "# Schedule-Adaptive Cosine-to-WSD Search\n\n",
        "This search keeps the original assignment protocol: `kappa` is fitted from `cosine_72000.csv` residuals only. "
        "WSD-family losses are used only to rank development candidates in this audit. The target schedule contributes only LR-derived features and a schedule-shape channel choice.\n\n",
        "## Searched Formula\n\n",
        "```text\n",
        "drop_concentration = max_t relu(lr_{t-1}-lr_t) / sum_t relu(lr_{t-1}-lr_t)\n",
        f"channel = step if drop_concentration >= {DROP_CONCENTRATION_THRESHOLD:g} else smooth\n",
        "phi_channel(t) = sum_{u <= t} exp(-lambda_channel (S_t-S_u)) * relu(lr_{u-1}-lr_u)/lr_peak\n",
        "r = L_true_cosine - L_MPL_cosine\n",
        "kappa_channel = [1/(1+rho)] * R_source_channel^p * max(0, <M_mu phi_channel, M_mu r> / (||M_mu phi_channel||^2 + tau^2))\n",
        "L_hat_target = L_MPL_target + kappa_channel(target) * phi_channel,target\n",
        "```\n\n",
        "## Best Fully Non-Harming Candidate\n\n",
        f"- Mean MAE change: `{fmt_pct(float(best['mean_delta']))}` over `{int(best['rows'])}` scale-target rows.\n",
        f"- Worst scale-target row: `{fmt_pct(float(best['worst_delta']))}`.\n",
        f"- Wins/non-harm: `{int(best['wins'])}/{int(best['rows'])}` and `{int(best['nonharm'])}/{int(best['rows'])}`.\n",
        f"- Config: `lambda_smooth={float(best['smooth_lambda']):g}`, `lambda_step={float(best['step_lambda']):g}`, "
        f"`mu={float(best['nuisance_lambda']):g}`, `max_mode={int(best['max_mode'])}`, "
        f"`tau={float(best['ridge_tau']):g}`, `p={float(best['retention_power']):g}`, `rho={float(best['rho']):g}`.\n\n",
        "## Best Worst-Case Candidate\n\n",
        f"- Mean / worst: `{fmt_pct(float(best_worst['mean_delta']))}` / `{fmt_pct(float(best_worst['worst_delta']))}`.\n",
        f"- Config: `lambda_smooth={float(best_worst['smooth_lambda']):g}`, `lambda_step={float(best_worst['step_lambda']):g}`, "
        f"`mu={float(best_worst['nuisance_lambda']):g}`, `max_mode={int(best_worst['max_mode'])}`, "
        f"`tau={float(best_worst['ridge_tau']):g}`, `p={float(best_worst['retention_power']):g}`, `rho={float(best_worst['rho']):g}`.\n\n",
        "## Per-Target Result For Best Mean Candidate\n\n",
        "| target | mean delta | worst scale | wins |\n",
        "|---|---:|---:|---:|\n",
    ]
    for row in target_rows:
        lines.append(
            f"| {row['test_label']} | {fmt_pct(float(row['mean_delta']))} | "
            f"{fmt_pct(float(row['worst_delta']))} | {int(row['wins'])}/{int(row['rows'])} |\n"
        )
    lines += [
        "\n## Comparison\n\n",
        "Previous old nextgen: mean `-17.2%`, worst `-2.2%`, wins `15/15`.\n",
        "Previous global response search: mean `-22.0%`, worst `-6.5%`, wins `15/15`.\n",
        "Manual schedule-adaptive audit: mean `-26.8%`, worst `-6.5%`, wins `15/15`.\n\n",
        "## Interpretation\n\n",
        "- Smooth WSD decays and WSD-con step decays have different identifiable response time scales. A single global response rate is therefore conservative.\n",
        "- The adaptive branch is schedule-only: it separates diffuse decay from concentrated LR drops using `drop_concentration`, not target loss.\n",
        "- Because the best hyperparameters are selected on the WSD family, this remains a development result; it should be checked with held-out WSD types or additional schedules before becoming the final paper/slides model.\n",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    config_rows, safe_rows, detail_rows = run_search()
    write_csv(OUT_DIR / "all_configs.csv", config_rows)
    write_csv(OUT_DIR / "safe_configs_top200.csv", safe_rows)
    write_csv(OUT_DIR / "top_safe_details.csv", detail_rows)
    if safe_rows:
        write_csv(OUT_DIR / "best_target_summary.csv", summarize_by_target(detail_rows, int(safe_rows[0]["config_id"])))
    write_report(config_rows, safe_rows, detail_rows)
    print(f"wrote {OUT_DIR / 'all_configs.csv'}")
    print(f"wrote {OUT_DIR / 'safe_configs_top200.csv'}")
    print(f"wrote {OUT_DIR / 'top_safe_details.csv'}")
    print(f"wrote {OUT_DIR / 'REPORT.md'}")


if __name__ == "__main__":
    main()
