#!/usr/bin/env python3
"""Theory-guided refinements of the observation-bracket MPL-LD estimator.

This audit keeps the core architecture fixed:

    L_hat = L_MPL + a_s * kappa_hat_s * phi_{lambda_s,s}.

It only tests schedule-only replacements for the two least theoretical parts
of the current formula:

1. Drop concentration for lambda_s.
2. The locality factor a_s.

No target loss is used to fit kappa or choose per-row parameters.
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import interpretable_error_model as iem  # noqa: E402
import interpretable_nuisance_origin_audit as noa  # noqa: E402
import interpretable_observation_bracket_audit as oba  # noqa: E402

OUT_DIR = iem.ROOT / "results" / "interpretable_theory_refinement"
FIT_START = 8000

VARIANTS = [
    {
        "variant": "current_qinf_support_projection",
        "q_rule": "qinf",
        "lambda_rule": "rate",
        "locality": "support_projection",
        "role": "current_formula_reinterpreted",
    },
    {
        "variant": "hhi_q2_support_projection",
        "q_rule": "q2",
        "lambda_rule": "rate",
        "locality": "support_projection",
        "role": "effective_drop_count_q",
    },
    {
        "variant": "hhi_q2_halflife_support_projection",
        "q_rule": "q2",
        "lambda_rule": "halflife",
        "locality": "support_projection",
        "role": "linear_half_life_interpolation",
    },
    {
        "variant": "hhi_q2_density_projection",
        "q_rule": "q2",
        "lambda_rule": "rate",
        "locality": "density_projection",
        "role": "full_density_projection_boundary",
    },
    {
        "variant": "hhi_q2_no_locality",
        "q_rule": "q2",
        "lambda_rule": "rate",
        "locality": "none",
        "role": "no_boundary_ablation",
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


def positive_drops(curve: iem.Curve) -> np.ndarray:
    eta = curve.lrs.astype(np.float64)
    drop = np.zeros_like(eta)
    drop[1:] = np.maximum(eta[:-1] - eta[1:], 0.0)
    return drop


def drop_stats(curve: iem.Curve) -> dict[str, float]:
    drop = positive_drops(curve)
    total = float(np.sum(drop))
    horizon = float(max(len(drop) - iem.WARMUP, 1))
    idx = np.flatnonzero(drop > 1e-18)
    support_span = float(idx[-1] - idx[0] + 2) if idx.size else 0.0
    if total <= 1e-18:
        return {
            "drop_total": 0.0,
            "qinf": 0.0,
            "q2": 0.0,
            "effective_drop_count": 0.0,
            "support_span": support_span,
            "post_warmup_horizon": horizon,
            "support_projection": 0.0,
            "density_projection": 0.0,
        }

    p_pos = drop[drop > 1e-18] / total
    qinf = float(np.max(p_pos))
    q2 = float(np.sum(p_pos * p_pos))
    support_projection = max(0.0, 1.0 - support_span / horizon)

    density = drop[iem.WARMUP :] / total
    uniform = np.ones_like(density) / max(len(density), 1)
    denom = float(np.dot(density, density))
    if denom > 1e-18:
        projection = uniform * (float(np.dot(uniform, density)) / float(np.dot(uniform, uniform)))
        residual = density - projection
        density_projection = float(np.dot(residual, residual) / denom)
    else:
        density_projection = 0.0

    return {
        "drop_total": total,
        "qinf": qinf,
        "q2": q2,
        "effective_drop_count": float(1.0 / q2) if q2 > 1e-18 else 0.0,
        "support_span": support_span,
        "post_warmup_horizon": horizon,
        "support_projection": support_projection,
        "density_projection": density_projection,
    }


def lambda_obs(curve: iem.Curve) -> float:
    return math.log(2.0) / (iem.PEAK_LR * iem.modal_observation_interval(curve))


def response_lambda(curve: iem.Curve, q_rule: str, lambda_rule: str) -> float:
    stats = drop_stats(curve)
    q = float(stats[q_rule])
    obs = lambda_obs(curve)
    if lambda_rule == "rate":
        return obs * (1.0 + q) / 2.0
    if lambda_rule == "halflife":
        return obs / max(2.0 - q, 1e-12)
    raise ValueError(f"unknown lambda rule: {lambda_rule}")


def locality_factor(curve: iem.Curve, mode: str) -> float:
    if mode == "none":
        return 1.0
    stats = drop_stats(curve)
    if mode == "support_projection":
        return float(stats["support_projection"])
    if mode == "density_projection":
        return float(stats["density_projection"])
    raise ValueError(f"unknown locality mode: {mode}")


def load_pack(scale: str, curve_name: str, cache: dict[tuple[str, str], iem.CurvePack]) -> iem.CurvePack:
    key = (scale, curve_name)
    if key not in cache:
        cache[key] = oba.load_pack(scale, curve_name)
    return cache[key]


def evaluate() -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    detail_rows: list[dict[str, object]] = []
    coef_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    cache: dict[tuple[str, str], iem.CurvePack] = {}
    basis_cache: dict[tuple[str, str, int], np.ndarray] = {}
    seen_coef: set[tuple[str, str, str, str, str]] = set()

    for group, curve_name, label in noa.ALL_TARGETS:
        for scale in iem.SCALES:
            curve = load_pack(scale, curve_name, cache).curve
            stats = drop_stats(curve)
            diagnostic_rows.append(
                {
                    "group": group,
                    "scale": scale,
                    "curve": curve_name,
                    "label": label,
                    **stats,
                }
            )

    for variant in VARIANTS:
        for train_scale in iem.SCALES:
            source = load_pack(train_scale, iem.TRAIN_CURVE, cache)
            for test_scale in iem.SCALES:
                for group, curve_name, label in noa.ALL_TARGETS:
                    target = load_pack(test_scale, curve_name, cache)
                    lam = response_lambda(target.curve, variant["q_rule"], variant["lambda_rule"])
                    coef, info = oba.fit_coefficient(
                        source,
                        lam,
                        "mpl_ld4",
                        "sample_size_ridge",
                        FIT_START,
                        basis_cache,
                    )
                    factor = locality_factor(target.curve, variant["locality"])
                    feature = iem.causal_drop_response(target.curve, lam)
                    pred = target.baseline + factor * coef * feature
                    corr_mae = iem.mae(target.curve.loss, pred)
                    delta = 100.0 * (corr_mae / target.base_mae - 1.0)
                    stats = drop_stats(target.curve)
                    row = {
                        "variant": variant["variant"],
                        "role": variant["role"],
                        "q_rule": variant["q_rule"],
                        "lambda_rule": variant["lambda_rule"],
                        "locality": variant["locality"],
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
    return detail_rows, coef_rows, diagnostic_rows


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


def diagnostic_table(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    labels = list(dict.fromkeys(str(row["label"]) for row in rows))
    for label in labels:
        sub = [row for row in rows if row["label"] == label]
        out.append(
            {
                "label": label,
                "group": sub[0]["group"],
                "qinf": float(np.mean([float(row["qinf"]) for row in sub])),
                "q2": float(np.mean([float(row["q2"]) for row in sub])),
                "effective_drop_count": float(np.mean([float(row["effective_drop_count"]) for row in sub])),
                "support_span": float(np.mean([float(row["support_span"]) for row in sub])),
                "support_projection": float(np.mean([float(row["support_projection"]) for row in sub])),
                "density_projection": float(np.mean([float(row["density_projection"]) for row in sub])),
            }
        )
    return out


def write_report(
    summary: list[dict[str, object]],
    diagnostics: list[dict[str, object]],
) -> None:
    lines: list[str] = []
    lines += [
        "# Interpretable Theory Refinement Audit\n\n",
        "本 audit 不改变核心架构，只检查两个 schedule-only 规则是否可以更理论化：\n\n",
        "\\[\n",
        "\\widehat L_s(t)=L_{\\mathrm{MPL},s}(t)+a_s\\widehat\\kappa_s\\phi_{\\lambda_s,s}(t).\n",
        "\\]\n\n",
        "其中 \\(\\widehat\\kappa_s\\) 仍然只从 cosine residual 经过 MPL-LD tangent projection 后的一维非负投影得到。目标 loss 只用于最后评价。\n\n",
        "## 理论修正 1：drop concentration\n\n",
        "令 \\(d_t=[\\eta_{t-1}-\\eta_t]_+\\)，\\(D=\\sum_t d_t\\)，\\(p_t=d_t/D\\)。旧公式使用\n\n",
        "\\[\n",
        "q_\\infty=\\|d\\|_\\infty/\\|d\\|_1=\\max_t p_t.\n",
        "\\]\n\n",
        "一个更有统计解释的替代是 Herfindahl concentration：\n\n",
        "\\[\n",
        "q_2=\\sum_t p_t^2=1/n_{\\mathrm{eff}}.\n",
        "\\]\n\n",
        "\\(q_2\\) 可以解释为 drop 分布的 effective atom count：单步 drop 时 \\(q_2=1\\)，均匀分布在 \\(n\\) 个 step 上时 \\(q_2=1/n\\)。\n",
        "因此 response half-life 可以写成\n\n",
        "\\[\n",
        "H_s=(2-q_2)\\Delta_{\\mathrm{obs}},\n",
        "\\qquad\n",
        "\\lambda_s=\\frac{\\lambda_{\\mathrm{obs}}}{2-q_2}.\n",
        "\\]\n\n",
        "这保持 observation bracket：diffuse drop 约为 two-observation half-life，single-step drop 为 one-observation half-life。\n\n",
        "## 理论修正 2：locality factor\n\n",
        "当前使用\n\n",
        "\\[\n",
        "a_s=\\mathbf{1}\\{D>0\\}\\left[1-\\frac{\\ell_s}{T_s-W}\\right]_+.\n",
        "\\]\n\n",
        "这个项可以从投影解释出来，而不是把它叫作 gate。设 post-warmup horizon 为 \\(H=T_s-W\\)，\n",
        "局部 drop support 上的均匀密度为 \\(m_t=\\mathbf{1}_{t\\in\\mathrm{supp}(d)}/\\ell_s\\)，全局 diffuse mode 为 \\(u_t=1/H\\)。\n",
        "把局部 forcing 投影到 diffuse mode 的正交补上：\n\n",
        "\\[\n",
        "\\frac{\\|(I-P_u)m\\|_2^2}{\\|m\\|_2^2}=1-\\frac{\\ell_s}{H}.\n",
        "\\]\n\n",
        "所以当前 \\(a_s\\) 可解释为：去掉 full-horizon adiabatic/diffuse forcing 后，局部 LR-drop forcing 保留下来的能量比例。\n",
        "这也解释了为什么 full-horizon cosine control 应该被压到 0，而 WSD-con single drop 接近 1。\n\n",
        "作为对照，本 audit 还测试了更细的 density projection：\n\n",
        "\\[\n",
        "a_s^{\\mathrm{density}}=\\frac{\\|(I-P_u)p\\|_2^2}{\\|p\\|_2^2}.\n",
        "\\]\n\n",
        "它更忠实于 drop density，但会把 cosine 的平滑非均匀下降也看成一部分 local signal；实验显示这对 controls 不够保守。\n\n",
        "## Schedule Diagnostics\n\n",
        "| curve | group | q_inf | q_2 | n_eff | support span | support proj | density proj |\n",
        "|---|---|---:|---:|---:|---:|---:|---:|\n",
    ]
    for row in diagnostics:
        lines.append(
            f"| {row['label']} | {row['group']} | {float(row['qinf']):.6f} | "
            f"{float(row['q2']):.6f} | {float(row['effective_drop_count']):.1f} | "
            f"{float(row['support_span']):.0f} | {float(row['support_projection']):.4f} | "
            f"{float(row['density_projection']):.4f} |\n"
        )

    lines += [
        "\n## Result Summary\n\n",
        "| variant | WSD same-scale | WSD cross-scale | controls same-scale | reading |\n",
        "|---|---:|---:|---:|---|\n",
    ]
    readings = {
        "current_qinf_support_projection": "当前公式；现在可解释为 support projection。",
        "hhi_q2_support_projection": "更可解释的 q2；结果应与当前接近。",
        "hhi_q2_halflife_support_projection": "half-life 线性插值；检查 bracket 解释敏感性。",
        "hhi_q2_density_projection": "更细 density projection；检查是否伤害 controls。",
        "hhi_q2_no_locality": "无边界负控。",
    }
    for variant in [item["variant"] for item in VARIANTS]:
        wsd_same = find(summary, variant, "core_wsd", "same_scale")
        wsd_cross = find(summary, variant, "core_wsd", "cross_scale")
        ctrl_same = find(summary, variant, "extra_control", "same_scale")
        lines.append(
            f"| {variant} | {fmt(wsd_same)} | {fmt(wsd_cross)} | {fmt(ctrl_same)} | "
            f"{readings[variant]} |\n"
        )

    best = find(summary, "hhi_q2_halflife_support_projection", "core_wsd", "same_scale")
    ctrl = find(summary, "hhi_q2_halflife_support_projection", "extra_control", "same_scale")
    lines += [
        "\n## Decision\n\n",
        "推荐把公式解释更新为 **q2 concentration + half-life bracket + support-projection locality**：\n\n",
        "\\[\n",
        "q_s=\\sum_t\\left(\\frac{d_t}{\\sum_u d_u}\\right)^2,\n",
        "\\qquad\n",
        "H_s=(2-q_s)\\Delta_{\\mathrm{obs}},\n",
        "\\qquad\n",
        "\\lambda_s=\\frac{\\log 2}{\\eta_{\\max}H_s}\n",
        "=\\frac{\\lambda_{\\mathrm{obs}}}{2-q_s},\n",
        "\\]\n\n",
        "\\[\n",
        "a_s=\\mathbf{1}\\{D_s>0\\}\\frac{\\|(I-P_u)m_s\\|_2^2}{\\|m_s\\|_2^2}\n",
        "=\\mathbf{1}\\{D_s>0\\}\\left[1-\\frac{\\ell_s}{T_s-W}\\right]_+.\n",
        "\\]\n\n",
        f"在当前数据上，该版本 WSD same-scale 为 `{fmt(best)}`，controls same-scale 为 `{fmt(ctrl)}`。\n",
        "它没有增加 residual-fitted 参数，仍然只有一个 \\(\\widehat\\kappa_s\\)。\n\n",
        "解释上，\\(q_2\\) 比 \\(q_\\infty\\) 更像 effective support size；直接插值 half-life 也比插值 rate 更符合 observation-bracket 叙事。\n",
        "\\(a_s\\) 不再是经验 gate，\n",
        "而是 local forcing 去掉 diffuse adiabatic mode 后的能量保留率。density projection 虽然自然，\n",
        "但会保留 cosine 的平滑非均匀下降，从 controls 看不够保守，因此暂不作为主公式。\n",
    ]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    detail_rows, coef_rows, diagnostic_rows = evaluate()
    summary_rows = summarize(detail_rows)
    diagnostics = diagnostic_table(diagnostic_rows)
    write_csv(OUT_DIR / "details.csv", detail_rows)
    write_csv(OUT_DIR / "coefficients.csv", coef_rows)
    write_csv(OUT_DIR / "summary.csv", summary_rows)
    write_csv(OUT_DIR / "schedule_diagnostics.csv", diagnostics)
    write_report(summary_rows, diagnostics)
    target = find(summary_rows, "hhi_q2_halflife_support_projection", "core_wsd", "same_scale")
    print(
        "hhi_q2_halflife_support_projection same-scale WSD: "
        f"{float(target['mean_delta']):+.2f}% / worst {float(target['worst_delta']):+.2f}% / "
        f"{int(target['wins'])}/{int(target['rows'])}"
    )


if __name__ == "__main__":
    main()
