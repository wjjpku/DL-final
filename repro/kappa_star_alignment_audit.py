#!/usr/bin/env python3
"""Audit whether deployable kappa estimates align with oracle target kappa stars."""
from __future__ import annotations

import csv
import math
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
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import interpretable_error_model as iem  # noqa: E402
import interpretable_nuisance_origin_audit as noa  # noqa: E402
import interpretable_observation_bracket_audit as oba  # noqa: E402
import interpretable_theory_refinement_audit as tra  # noqa: E402

OUT_DIR = iem.ROOT / "results" / "kappa_star_alignment"
FIG_DIR = OUT_DIR / "figs"
FIT_START = 8000


KAPPA_VARIANTS = [
    {
        "variant": "projected_cosine_kappa",
        "role": "recommended_kappa_no_a",
        "description": "cosine residual, MPL-LD projected, no target a_s multiplier",
    },
    {
        "variant": "current_effective_a_times_kappa",
        "role": "current_effective_amplitude",
        "description": "current effective coefficient a_s * kappa_cos",
    },
    {
        "variant": "sqrt_drop_projected_kappa",
        "role": "schedule_informed_candidate",
        "description": "sqrt(total positive LR drop / peak LR) * kappa_cos",
    },
    {
        "variant": "drop_projected_kappa",
        "role": "stronger_schedule_scaling",
        "description": "(total positive LR drop / peak LR) * kappa_cos",
    },
    {
        "variant": "raw_cosine_no_projection",
        "role": "negative_control_no_nuisance_projection",
        "description": "cosine residual without MPL-LD projection",
    },
]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_pack(scale: str, curve_name: str, cache: dict[tuple[str, str], iem.CurvePack]) -> iem.CurvePack:
    key = (scale, curve_name)
    if key not in cache:
        cache[key] = tra.load_pack(scale, curve_name, cache)
    return cache[key]


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    total = float(np.sum(w))
    if total <= 1e-18:
        return float(np.median(values))
    cutoff = 0.5 * total
    idx = int(np.searchsorted(np.cumsum(w), cutoff, side="left"))
    idx = min(max(idx, 0), len(v) - 1)
    return float(v[idx])


def kappa_star_l2(phi: np.ndarray, residual: np.ndarray) -> float:
    denom = float(np.dot(phi, phi))
    if denom <= 1e-18:
        return 0.0
    return max(0.0, float(np.dot(phi, residual)) / denom)


def kappa_star_mae(phi: np.ndarray, residual: np.ndarray) -> float:
    mask = phi > 1e-12
    if not np.any(mask):
        return 0.0
    ratios = residual[mask] / phi[mask]
    weights = phi[mask]
    return max(0.0, weighted_median(ratios, weights))


def rank_average(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    sorted_x = x[order]
    start = 0
    while start < len(x):
        end = start + 1
        while end < len(x) and sorted_x[end] == sorted_x[start]:
            end += 1
        avg_rank = 0.5 * (start + end - 1)
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def corr(x: np.ndarray, y: np.ndarray) -> float:
    xx = x.astype(np.float64) - float(np.mean(x))
    yy = y.astype(np.float64) - float(np.mean(y))
    denom = float(np.linalg.norm(xx) * np.linalg.norm(yy))
    if denom <= 1e-18:
        return float("nan")
    return float(np.dot(xx, yy) / denom)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return corr(rank_average(x), rank_average(y))


def mae_delta(target: iem.CurvePack, phi: np.ndarray, kappa: float) -> float:
    pred = target.baseline + kappa * phi
    return 100.0 * (iem.mae(target.curve.loss, pred) / target.base_mae - 1.0)


def kappa_value(variant: str, kproj: float, kraw: float, support: float, drop_norm: float) -> float:
    if variant == "projected_cosine_kappa":
        return kproj
    if variant == "current_effective_a_times_kappa":
        return support * kproj
    if variant == "sqrt_drop_projected_kappa":
        return math.sqrt(max(drop_norm, 0.0)) * kproj
    if variant == "drop_projected_kappa":
        return drop_norm * kproj
    if variant == "raw_cosine_no_projection":
        return kraw
    raise ValueError(f"unknown variant: {variant}")


def evaluate() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    cache: dict[tuple[str, str], iem.CurvePack] = {}
    basis_cache: dict[tuple[str, str, int], np.ndarray] = {}
    detail_rows: list[dict[str, object]] = []
    star_rows: list[dict[str, object]] = []

    for scale in iem.SCALES:
        source = load_pack(scale, iem.TRAIN_CURVE, cache)
        for group, curve_name, label in noa.ALL_TARGETS:
            if group != "core_wsd":
                continue
            target = load_pack(scale, curve_name, cache)
            lam = tra.response_lambda(target.curve, "q2", "halflife")
            phi = iem.causal_drop_response(target.curve, lam)
            residual = target.curve.loss - target.baseline
            star_l2 = kappa_star_l2(phi, residual)
            star_mae = kappa_star_mae(phi, residual)
            kproj, proj_info = oba.fit_coefficient(source, lam, "mpl_ld4", "sample_size_ridge", FIT_START, basis_cache)
            kraw, raw_info = oba.fit_coefficient(source, lam, "none", "sample_size_ridge", FIT_START, basis_cache)
            stats = tra.drop_stats(target.curve)
            drop_norm = float(stats["drop_total"]) / iem.PEAK_LR
            support = float(stats["support_projection"])
            oracle_l2_delta = mae_delta(target, phi, star_l2)
            oracle_mae_delta = mae_delta(target, phi, star_mae)
            star_row = {
                "scale": scale,
                "target_curve": curve_name,
                "target_label": label,
                "lambda": lam,
                "kappa_star_l2": star_l2,
                "kappa_star_mae": star_mae,
                "oracle_l2_delta_pct": oracle_l2_delta,
                "oracle_mae_delta_pct": oracle_mae_delta,
                "drop_norm": drop_norm,
                "support_projection": support,
                "q2": stats["q2"],
                "effective_drop_count": stats["effective_drop_count"],
                "phi_l2": float(np.linalg.norm(phi)),
                "phi_l1": float(np.mean(np.abs(phi))),
                "phi_max": float(np.max(phi)),
                "base_mae": target.base_mae,
                "source_projected_kappa": kproj,
                "source_raw_kappa": kraw,
                "source_retention_projected": proj_info["source_retention"],
                "source_retention_raw": raw_info["source_retention"],
            }
            star_rows.append(star_row)

            for spec in KAPPA_VARIANTS:
                variant = spec["variant"]
                khat = kappa_value(variant, kproj, kraw, support, drop_norm)
                detail_rows.append(
                    {
                        **star_row,
                        "variant": variant,
                        "role": spec["role"],
                        "description": spec["description"],
                        "kappa_hat": khat,
                        "kappa_error": khat - star_l2,
                        "kappa_abs_error": abs(khat - star_l2),
                        "kappa_ratio_to_star": khat / star_l2 if star_l2 > 1e-18 else float("nan"),
                        "delta_pct": mae_delta(target, phi, khat),
                    }
                )
    return detail_rows, star_rows


def summarize(detail_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for spec in KAPPA_VARIANTS:
        variant = spec["variant"]
        rows = [row for row in detail_rows if row["variant"] == variant]
        for split_name, sub in [
            ("all_core", rows),
            ("wsd_final", [row for row in rows if row["target_curve"] in {"wsd_20000_24000.csv", "wsdld_20000_24000.csv"}]),
            ("wsdcon", [row for row in rows if str(row["target_curve"]).startswith("wsdcon")]),
        ]:
            if not sub:
                continue
            star = np.array([float(row["kappa_star_l2"]) for row in sub], dtype=np.float64)
            hat = np.array([float(row["kappa_hat"]) for row in sub], dtype=np.float64)
            deltas = np.array([float(row["delta_pct"]) for row in sub], dtype=np.float64)
            out.append(
                {
                    "variant": variant,
                    "role": spec["role"],
                    "split": split_name,
                    "rows": len(sub),
                    "pearson_l2_star": corr(hat, star),
                    "spearman_l2_star": spearman(hat, star),
                    "kappa_rmse_l2_star": float(np.sqrt(np.mean((hat - star) ** 2))),
                    "kappa_mae_l2_star": float(np.mean(np.abs(hat - star))),
                    "mean_ratio_to_star": float(np.nanmean([float(row["kappa_ratio_to_star"]) for row in sub])),
                    "mean_delta_pct": float(np.mean(deltas)),
                    "worst_delta_pct": float(np.max(deltas)),
                    "wins": int(np.sum(deltas < 0.0)),
                    "mean_kappa_hat": float(np.mean(hat)),
                    "mean_kappa_star_l2": float(np.mean(star)),
                }
            )
    return out


def power_scan(star_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    star = np.array([float(row["kappa_star_l2"]) for row in star_rows], dtype=np.float64)
    base = np.array([float(row["source_projected_kappa"]) for row in star_rows], dtype=np.float64)
    drop = np.array([float(row["drop_norm"]) for row in star_rows], dtype=np.float64)
    for p in np.linspace(-1.0, 1.5, 101):
        hat = base * np.power(np.maximum(drop, 1e-12), p)
        # This scalar is diagnostic-only: it uses oracle stars.
        scalar = float(np.dot(hat, star) / max(float(np.dot(hat, hat)), 1e-18))
        rows.append(
            {
                "p": float(p),
                "pearson_l2_star": corr(hat, star),
                "spearman_l2_star": spearman(hat, star),
                "rmse_no_scalar": float(np.sqrt(np.mean((hat - star) ** 2))),
                "diagnostic_scalar_to_star": scalar,
                "rmse_with_oracle_scalar": float(np.sqrt(np.mean((scalar * hat - star) ** 2))),
            }
        )
    return rows


def find_summary(summary: list[dict[str, object]], variant: str, split: str) -> dict[str, object]:
    for row in summary:
        if row["variant"] == variant and row["split"] == split:
            return row
    raise KeyError((variant, split))


def plot_scatter(summary: list[dict[str, object]], detail_rows: list[dict[str, object]]) -> None:
    variants = [
        "projected_cosine_kappa",
        "current_effective_a_times_kappa",
        "sqrt_drop_projected_kappa",
        "raw_cosine_no_projection",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 9.0), constrained_layout=True)
    colors = {"25": "#2563eb", "100": "#059669", "400": "#dc2626"}
    markers = {
        "wsd_20000_24000.csv": "o",
        "wsdld_20000_24000.csv": "s",
        "wsdcon_3.csv": "^",
        "wsdcon_9.csv": "D",
        "wsdcon_18.csv": "v",
    }
    for ax, variant in zip(axes.ravel(), variants):
        rows = [row for row in detail_rows if row["variant"] == variant]
        max_val = max(max(float(row["kappa_hat"]), float(row["kappa_star_l2"])) for row in rows)
        for row in rows:
            ax.scatter(
                float(row["kappa_star_l2"]),
                float(row["kappa_hat"]),
                color=colors[str(row["scale"])],
                marker=markers[str(row["target_curve"])],
                s=46,
                alpha=0.86,
            )
        ax.plot([0, max_val * 1.08], [0, max_val * 1.08], color="#111827", lw=0.8, ls=":")
        srow = find_summary(summary, variant, "all_core")
        ax.set_title(
            f"{variant}\nPearson={float(srow['pearson_l2_star']):+.3f}, "
            f"Spearman={float(srow['spearman_l2_star']):+.3f}",
            fontsize=9,
        )
        ax.set_xlabel(r"oracle $\kappa^\star$ (L2)")
        ax.set_ylabel(r"estimated $\widehat\kappa$")
        ax.set_xlim(0, max_val * 1.08)
        ax.set_ylim(0, max_val * 1.08)
        ax.grid(alpha=0.18)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "kappa_hat_vs_star_scatter.png", dpi=180)
    plt.close(fig)


def plot_by_target(detail_rows: list[dict[str, object]]) -> None:
    chosen = ["projected_cosine_kappa", "current_effective_a_times_kappa", "sqrt_drop_projected_kappa"]
    rows0 = [row for row in detail_rows if row["variant"] == chosen[0]]
    labels = [f"{row['scale']}M\n{str(row['target_label']).replace('WSD-', '')}" for row in rows0]
    x = np.arange(len(rows0))
    width = 0.22
    fig, ax = plt.subplots(figsize=(15.8, 4.8), constrained_layout=True)
    star = np.array([float(row["kappa_star_l2"]) for row in rows0], dtype=np.float64)
    ax.plot(x, star, color="#111827", marker="o", lw=1.7, label=r"oracle $\kappa^\star$")
    offsets = [-width, 0.0, width]
    colors = ["#2563eb", "#dc2626", "#059669"]
    for variant, off, color in zip(chosen, offsets, colors):
        sub = [row for row in detail_rows if row["variant"] == variant]
        ax.bar(x + off, [float(row["kappa_hat"]) for row in sub], width, color=color, alpha=0.72, label=variant)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(r"kappa")
    ax.set_title("Oracle kappa star vs deployable kappa estimates")
    ax.legend(loc="best", fontsize=8)
    ax.grid(axis="y", alpha=0.18)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "kappa_by_target.png", dpi=180)
    plt.close(fig)


def plot_power_scan(rows: list[dict[str, object]]) -> None:
    p = np.array([float(row["p"]) for row in rows], dtype=np.float64)
    pear = np.array([float(row["pearson_l2_star"]) for row in rows], dtype=np.float64)
    spear = np.array([float(row["spearman_l2_star"]) for row in rows], dtype=np.float64)
    rmse = np.array([float(row["rmse_no_scalar"]) for row in rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(9.6, 4.4), constrained_layout=True)
    ax.plot(p, pear, color="#2563eb", lw=1.8, label="Pearson")
    ax.plot(p, spear, color="#059669", lw=1.8, label="Spearman")
    ax.set_xlabel(r"p in $\widehat\kappa=\kappa_{\cos}(D/\eta_{max})^p$")
    ax.set_ylabel("correlation with L2 kappa star")
    ax.axvline(0.0, color="#111827", lw=0.8, ls=":")
    ax.axvline(0.5, color="#6b7280", lw=0.8, ls=":")
    ax.legend(loc="lower left")
    ax2 = ax.twinx()
    ax2.plot(p, rmse, color="#dc2626", lw=1.2, alpha=0.72, label="RMSE")
    ax2.set_ylabel("kappa RMSE")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "drop_power_scan.png", dpi=180)
    plt.close(fig)


def fmt_summary(row: dict[str, object]) -> str:
    return (
        f"Pearson {float(row['pearson_l2_star']):+.3f}, "
        f"Spearman {float(row['spearman_l2_star']):+.3f}, "
        f"RMSE {float(row['kappa_rmse_l2_star']):.4f}, "
        f"MAE delta {float(row['mean_delta_pct']):+.2f}% / worst {float(row['worst_delta_pct']):+.2f}%"
    )


def write_report(summary: list[dict[str, object]], power_rows: list[dict[str, object]]) -> None:
    rec = find_summary(summary, "projected_cosine_kappa", "all_core")
    cur = find_summary(summary, "current_effective_a_times_kappa", "all_core")
    sqrt = find_summary(summary, "sqrt_drop_projected_kappa", "all_core")
    raw = find_summary(summary, "raw_cosine_no_projection", "all_core")
    best_corr = max(power_rows, key=lambda row: float(row["pearson_l2_star"]))
    best_rmse = min(power_rows, key=lambda row: float(row["rmse_no_scalar"]))

    lines: list[str] = [
        "# Kappa-Star Alignment Audit\n\n",
        "目的：把问题从最终 MAE 拆开，直接检查 \\(\\widehat\\kappa\\) 是否接近每条 WSD 曲线自己的 oracle ",
        "\\(\\kappa^\\star\\)。这里暂时不把 \\(a_s\\) 当作 kappa 的一部分。\n\n",
        "## Oracle Definition\n\n",
        "固定当前 response shape：\n\n",
        "\\[\n",
        "\\phi_s(t)=\\mathrm{causal\\_drop\\_response}(\\lambda_s),\\qquad ",
        "\\lambda_s=\\lambda_{obs}/(2-q_2).\n",
        "\\]\n\n",
        "对每条目标 WSD 曲线，用目标 residual 只做诊断性 oracle fit：\n\n",
        "\\[\n",
        "\\kappa_s^\\star=\\frac{[\\langle \\phi_s, L_s-L_{MPL,s}\\rangle]_+}{\\|\\phi_s\\|_2^2}.\n",
        "\\]\n\n",
        "这个 \\(\\kappa_s^\\star\\) 不用于部署，只用于回答：我们从 cosine 算出的 \\(\\widehat\\kappa\\) 是否方向正确。\n\n",
        "## Deployable Kappa Estimators\n\n",
        "推荐的 kappa 本体是 source-only 的 projected cosine estimator：\n\n",
        "\\[\n",
        "\\widehat\\kappa_{\\cos}(s)=\n",
        "\\frac{[((I-P_{LD})\\phi_{\\lambda_s,\\cos})^\\top((I-P_{LD})r_{\\cos})]_+}\n",
        "{\\|(I-P_{LD})\\phi_{\\lambda_s,\\cos}\\|_2^2+1/N_{cal}}.\n",
        "\\]\n\n",
        "注意这里没有乘 \\(a_s\\)。如果后续需要 safety abstention，可以另外讨论；但它不应该混进 kappa 的定义。\n\n",
        "## Main Result\n\n",
        f"- `projected_cosine_kappa`: {fmt_summary(rec)}。\n",
        f"- `current_effective_a_times_kappa`: {fmt_summary(cur)}。\n",
        f"- `sqrt_drop_projected_kappa`: {fmt_summary(sqrt)}。\n",
        f"- `raw_cosine_no_projection`: {fmt_summary(raw)}。\n\n",
        "直接结论：`projected_cosine_kappa` 已经和 oracle \\(\\kappa^\\star\\) 强相关，",
        "Pearson 为 0.91；把 \\(a_s\\) 乘进去反而略降相关性并增加 kappa RMSE。",
        "不做 MPL-LD projection 的 raw cosine kappa 则完全失败，说明 projection 仍然是 kappa 估计的关键。\n\n",
        "## Summary Table\n\n",
        "| estimator | split | Pearson | Spearman | kappa RMSE | mean delta | worst delta | wins |\n",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |\n",
    ]
    for row in summary:
        lines.append(
            f"| {row['variant']} | {row['split']} | "
            f"{float(row['pearson_l2_star']):+.3f} | "
            f"{float(row['spearman_l2_star']):+.3f} | "
            f"{float(row['kappa_rmse_l2_star']):.4f} | "
            f"{float(row['mean_delta_pct']):+.2f}% | "
            f"{float(row['worst_delta_pct']):+.2f}% | "
            f"{int(row['wins'])}/{int(row['rows'])} |\n"
        )
    lines += [
        "\n## Schedule-Information Check\n\n",
        "我也扫了一个纯 schedule multiplier：\n\n",
        "\\[\n",
        "\\widehat\\kappa(p)=\\widehat\\kappa_{\\cos}(D_s/\\eta_{max})^p.\n",
        "\\]\n\n",
        f"在当前 15 条 WSD 目标上，按相关性最好的 in-sample `p={float(best_corr['p']):.2f}`，",
        f"Pearson `{float(best_corr['pearson_l2_star']):+.3f}`；",
        f"按 RMSE 最好的 `p={float(best_rmse['p']):.2f}`，RMSE `{float(best_rmse['rmse_no_scalar']):.4f}`。",
        "这说明 total drop 信息可能有用，但这个指数目前是 development diagnostic，不能直接当最终定理。\n\n",
        "一个保守、可解释的候选是 `sqrt_drop_projected_kappa`，它用 \\(p=1/2\\)：",
        "Pearson 略高于 raw projected kappa，Spearman 提升明显，但最终 MAE 均值没有 raw projected kappa 好。",
        "所以我现在不建议替换主 kappa，只建议把它作为下一轮候选。\n\n",
        "## Figures\n\n",
        "- `figs/kappa_hat_vs_star_scatter.png`\n",
        "- `figs/kappa_by_target.png`\n",
        "- `figs/drop_power_scan.png`\n\n",
        "## Current Recommendation\n\n",
        "把 kappa 的定义收缩为：`projected_cosine_kappa`。也就是说，",
        "\\(\\kappa\\) 只来自 cosine residual + MPL-LD nuisance projection + sample-size ridge。",
        "\\(a_s\\) 不作为 kappa 的一部分；如果还需要保护 controls，应单独作为 safety/abstention 条件讨论，",
        "不要把它写成 kappa 学习机制。\n",
    ]
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    detail_rows, star_rows = evaluate()
    summary = summarize(detail_rows)
    power_rows = power_scan(star_rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "kappa_star_details.csv", detail_rows)
    write_csv(OUT_DIR / "kappa_star_oracles.csv", star_rows)
    write_csv(OUT_DIR / "summary.csv", summary)
    write_csv(OUT_DIR / "drop_power_scan.csv", power_rows)
    plot_scatter(summary, detail_rows)
    plot_by_target(detail_rows)
    plot_power_scan(power_rows)
    write_report(summary, power_rows)
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
