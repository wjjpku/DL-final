#!/usr/bin/env python3
"""Ablate the MPL-LD tangent projection in the current q2 half-life formula."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import interpretable_error_model as iem  # noqa: E402
import interpretable_nuisance_origin_audit as noa  # noqa: E402
import interpretable_observation_bracket_audit as oba  # noqa: E402
import interpretable_theory_refinement_audit as tra  # noqa: E402

OUT_DIR = iem.ROOT / "results" / "mpl_ld_projection_ablation"
FIT_START = tra.FIT_START
Q_RULE = "q2"
LAMBDA_RULE = "halflife"
LOCALITY = "support_projection"

VARIANTS = [
    {
        "variant": "current_with_mplld_projection",
        "projection": "mpl_ld4",
        "role": "current_formula",
    },
    {
        "variant": "current_without_mplld_projection",
        "projection": "none",
        "role": "projection_ablation",
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


def evaluate() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    detail_rows: list[dict[str, object]] = []
    coef_rows: list[dict[str, object]] = []
    cache: dict[tuple[str, str], iem.CurvePack] = {}
    basis_cache: dict[tuple[str, str, int], np.ndarray] = {}
    seen_coef: set[tuple[str, str, str, str, str]] = set()

    for variant in VARIANTS:
        for train_scale in iem.SCALES:
            source = load_pack(train_scale, iem.TRAIN_CURVE, cache)
            for test_scale in iem.SCALES:
                for group, curve_name, label in noa.ALL_TARGETS:
                    target = load_pack(test_scale, curve_name, cache)
                    lam = tra.response_lambda(target.curve, Q_RULE, LAMBDA_RULE)
                    coef, info = oba.fit_coefficient(
                        source,
                        lam,
                        variant["projection"],
                        "sample_size_ridge",
                        FIT_START,
                        basis_cache,
                    )
                    factor = tra.locality_factor(target.curve, LOCALITY)
                    feature = iem.causal_drop_response(target.curve, lam)
                    pred = target.baseline + factor * coef * feature
                    corr_mae = iem.mae(target.curve.loss, pred)
                    delta = 100.0 * (corr_mae / target.base_mae - 1.0)
                    stats = tra.drop_stats(target.curve)
                    row = {
                        "variant": variant["variant"],
                        "role": variant["role"],
                        "projection": variant["projection"],
                        "q_rule": Q_RULE,
                        "lambda_rule": LAMBDA_RULE,
                        "locality": LOCALITY,
                        "fit_start": FIT_START,
                        "group": group,
                        "train_scale": train_scale,
                        "test_scale": test_scale,
                        "test_curve": curve_name,
                        "test_label": label,
                        "lambda": lam,
                        "coef": coef,
                        "locality_factor": factor,
                        "base_mae": target.base_mae,
                        "corr_mae": corr_mae,
                        "delta_pct": delta,
                        "win": int(corr_mae < target.base_mae),
                        "nonharm": int(delta <= 1e-12),
                        **{f"target_{key}": value for key, value in stats.items()},
                        **info,
                    }
                    detail_rows.append(row)

                    coef_key = (
                        variant["variant"],
                        train_scale,
                        test_scale,
                        curve_name,
                        f"{lam:.12g}",
                    )
                    if coef_key not in seen_coef:
                        seen_coef.add(coef_key)
                        coef_rows.append(
                            {
                                key: row[key]
                                for key in [
                                    "variant",
                                    "role",
                                    "projection",
                                    "q_rule",
                                    "lambda_rule",
                                    "locality",
                                    "fit_start",
                                    "train_scale",
                                    "test_scale",
                                    "test_curve",
                                    "test_label",
                                    "lambda",
                                    "coef",
                                    "locality_factor",
                                    "basis_dim",
                                    "n_cal",
                                    "ridge",
                                    "source_dot",
                                    "source_full_norm",
                                    "source_perp_norm",
                                    "source_retention",
                                    "denominator",
                                ]
                            }
                        )
    return detail_rows, coef_rows


def aggregate(rows: list[dict[str, object]], variant: str, group: str, split: str) -> dict[str, object]:
    deltas = np.array([float(row["delta_pct"]) for row in rows], dtype=np.float64)
    return {
        "variant": variant,
        "group": group,
        "split": split,
        "rows": len(rows),
        "mean_delta": float(np.mean(deltas)),
        "median_delta": float(np.median(deltas)),
        "worst_delta": float(np.max(deltas)),
        "best_delta": float(np.min(deltas)),
        "wins": int(np.sum(deltas < 0.0)),
        "nonharm": int(np.sum(deltas <= 1e-12)),
    }


def summarize(detail_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for variant in [item["variant"] for item in VARIANTS]:
        rows_v = [row for row in detail_rows if row["variant"] == variant]
        for group in ["core_wsd", "extra_control"]:
            rows_g = [row for row in rows_v if row["group"] == group]
            splits = {
                "all": rows_g,
                "same_scale": [row for row in rows_g if row["train_scale"] == row["test_scale"]],
                "cross_scale": [row for row in rows_g if row["train_scale"] != row["test_scale"]],
            }
            for split, rows in splits.items():
                if rows:
                    out.append(aggregate(rows, variant, group, split))
    return out


def find(summary: list[dict[str, object]], variant: str, group: str, split: str) -> dict[str, object]:
    for row in summary:
        if row["variant"] == variant and row["group"] == group and row["split"] == split:
            return row
    raise KeyError((variant, group, split))


def fmt(row: dict[str, object]) -> str:
    return (
        f"{float(row['mean_delta']):+.2f}% / {float(row['worst_delta']):+.2f}% / "
        f"{int(row['wins'])}/{int(row['rows'])}"
    )


def coef_stats(coef_rows: list[dict[str, object]], variant: str) -> dict[str, float]:
    rows = [row for row in coef_rows if row["variant"] == variant]
    coefs = np.array([float(row["coef"]) for row in rows], dtype=np.float64)
    ret = np.array([float(row["source_retention"]) for row in rows], dtype=np.float64)
    perp = np.array([float(row["source_perp_norm"]) for row in rows], dtype=np.float64)
    full = np.array([float(row["source_full_norm"]) for row in rows], dtype=np.float64)
    return {
        "coef_mean": float(np.mean(coefs)),
        "coef_median": float(np.median(coefs)),
        "coef_max": float(np.max(coefs)),
        "retention_mean": float(np.mean(ret)),
        "retention_median": float(np.median(ret)),
        "perp_norm_mean": float(np.mean(perp)),
        "full_norm_mean": float(np.mean(full)),
    }


def write_report(summary: list[dict[str, object]], coef_rows: list[dict[str, object]]) -> None:
    with_proj = "current_with_mplld_projection"
    no_proj = "current_without_mplld_projection"

    lines: list[str] = []
    lines += [
        "# MPL-LD Tangent Projection Ablation\n\n",
        "本消融只检查一个问题：在当前 q2 half-life 公式里，是否必须先取 MPL-LD 梯度矩阵并做正交投影。\n\n",
        "除这一项外，所有设定保持一致：cosine 作为 source，`fit_start=8000`，`q_rule=q2`，`lambda_rule=halflife`，",
        "`locality=support_projection`，ridge 为 `1/N_cal`，目标曲线的 loss 只用于最终评价。\n\n",
        "## 被消融的操作\n\n",
        "当前带投影版本使用\n\n",
        "\\[\n",
        "x=(I-P_{LD})\\phi_{\\lambda_s,\\mathrm{cos}},\\qquad y=(I-P_{LD})r_{\\mathrm{cos}},\n",
        "\\]\n\n",
        "其中 \\(P_{LD}=Q_{LD}Q_{LD}^{\\top}\\)，\\(Q_{LD}\\) 来自 cosine 后缀区间上 MPL 参数 ",
        "\\((\\log B,\\log C,\\log\\beta,\\log\\gamma)\\) 的 finite-difference tangent matrix 的 QR 正交化。然后\n\n",
        "\\[\n",
        "\\widehat\\kappa_s=\\frac{[x^{\\top}y]_+}{\\|x\\|_2^2+1/N_{cal}}.\n",
        "\\]\n\n",
        "无投影消融把上式替换为\n\n",
        "\\[\n",
        "x=\\phi_{\\lambda_s,\\mathrm{cos}},\\qquad y=r_{\\mathrm{cos}},\n",
        "\\]\n\n",
        "其余 \\(\\widehat\\kappa_s\\) 的估计式完全不变。\n\n",
        "最终预测仍然是\n\n",
        "\\[\n",
        "\\widehat L_s(t)=L_{\\mathrm{MPL},s}(t)+a_s\\widehat\\kappa_s\\phi_{\\lambda_s,s}(t),\n",
        "\\]\n\n",
        "其中 \\(q_s=\\sum_t(d_t/D)^2\\)，\\(\\lambda_s=\\lambda_{obs}/(2-q_s)\\)，",
        "\\(a_s=1-\\mathrm{support\\_span}/\\mathrm{post\\_warmup\\_horizon}\\)。\n\n",
        "## 结果\n\n",
        "| variant | split | group | mean / worst / wins |\n",
        "| --- | --- | --- | --- |\n",
    ]

    for variant in [with_proj, no_proj]:
        for group in ["core_wsd", "extra_control"]:
            for split in ["same_scale", "cross_scale", "all"]:
                row = find(summary, variant, group, split)
                lines.append(f"| {variant} | {split} | {group} | {fmt(row)} |\n")

    with_stats = coef_stats(coef_rows, with_proj)
    none_stats = coef_stats(coef_rows, no_proj)
    lines += [
        "\n## 系数诊断\n\n",
        "| variant | mean coef | median coef | max coef | mean source retention | median source retention |\n",
        "| --- | ---: | ---: | ---: | ---: | ---: |\n",
        (
            f"| {with_proj} | {with_stats['coef_mean']:.6g} | {with_stats['coef_median']:.6g} | "
            f"{with_stats['coef_max']:.6g} | {with_stats['retention_mean']:.6g} | "
            f"{with_stats['retention_median']:.6g} |\n"
        ),
        (
            f"| {no_proj} | {none_stats['coef_mean']:.6g} | {none_stats['coef_median']:.6g} | "
            f"{none_stats['coef_max']:.6g} | {none_stats['retention_mean']:.6g} | "
            f"{none_stats['retention_median']:.6g} |\n"
        ),
        "\n## 结论\n\n",
        "这个投影是必要的。不开投影时，cosine residual 中可被 MPL-LD 参数微调解释的平滑结构会直接进入 ",
        "\\(\\widehat\\kappa_s\\)，导致校正项把 MPL 本身的低维拟合误差也当成 schedule-response 误差。",
        "因此跨到 WSD 时不是小幅退化，而是系统性失败。\n\n",
        "带投影版本虽然只保留 source response feature 中很小的一部分正交能量，但它保留的是 MPL-LD 低维重拟合不能解释的残差方向。",
        "这正是我们希望 \\(\\kappa\\) 学到的部分：与 learning-rate drop 的滞后响应有关，而不是与 MPL 参数偏移有关。\n",
    ]

    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    detail_rows, coef_rows = evaluate()
    summary = summarize(detail_rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "details.csv", detail_rows)
    write_csv(OUT_DIR / "coefficients.csv", coef_rows)
    write_csv(OUT_DIR / "summary.csv", summary)
    write_report(summary, coef_rows)
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
