#!/usr/bin/env python3
"""Write the interpretability-first decision note for the residual model."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import interpretable_error_model as iem  # noqa: E402
import interpretable_strict_vs_rounded_audit as strict  # noqa: E402

OUT_DIR = iem.ROOT / "results" / "interpretable_error_model"
FIT_START = 8000
NUISANCE_LAMBDA = 0.01
RIDGE_TAU = iem.RIDGE_TAU
EXTRA_CONTROLS = [
    ("cosine_24000.csv", "Cosine 24k"),
    ("constant_24000.csv", "Constant 24k"),
    ("constant_72000.csv", "Constant 72k"),
]
ALL_TARGETS = [("core_wsd", *item) for item in iem.TARGETS] + [
    ("extra_control", *item) for item in EXTRA_CONTROLS
]


def load_pack(scale: str, curve_name: str) -> iem.CurvePack:
    curve = iem.load_curve(scale, curve_name)
    params = iem.MPL_PRECOMPUTED_INIT[scale]
    baseline = iem.mpl_predict(params, curve)
    residual = curve.loss - baseline
    slope_raw, slope_norm = iem.mpl_slope_features(curve, baseline)
    ld_basis, dlogc_basis = iem.mpl_sensitivity_features(params, curve)
    return iem.CurvePack(
        curve=curve,
        baseline=baseline,
        residual=residual,
        base_mae=iem.mae(curve.loss, baseline),
        slope_raw=slope_raw,
        slope_norm=slope_norm,
        ld_basis=ld_basis,
        dlogc_basis=dlogc_basis,
    )


def fixed_lambda_rows(variant: str, response_lambda: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scale in iem.SCALES:
        source = load_pack(scale, iem.TRAIN_CURVE)
        source_feature = iem.causal_drop_response(source.curve, response_lambda)[:, None]
        coef, fit_info = iem.fit_nonnegative_ridge(
            source.residual,
            source_feature,
            source.curve.step,
            fit_start=FIT_START,
            nuisance_lambda=NUISANCE_LAMBDA,
            max_mode=iem.DCT_MODES,
            ridge_tau=RIDGE_TAU,
            signed=False,
        )
        for group, curve_name, label in ALL_TARGETS:
            target = load_pack(scale, curve_name)
            feature = iem.causal_drop_response(target.curve, response_lambda)
            pred = target.baseline + float(coef[0]) * feature
            corr_mae = iem.mae(target.curve.loss, pred)
            rows.append(
                {
                    "variant": variant,
                    "group": group,
                    "scale": scale,
                    "test_curve": curve_name,
                    "test_label": label,
                    "lambda": response_lambda,
                    "coef": float(coef[0]),
                    "localization": 1.0,
                    "fit_start": FIT_START,
                    "nuisance_lambda": NUISANCE_LAMBDA,
                    "ridge_tau": RIDGE_TAU,
                    "fit_objective": float(fit_info["fit_objective"]),
                    "residualized_corr": float(fit_info["residualized_corr"]),
                    "base_mae": target.base_mae,
                    "corr_mae": corr_mae,
                    "delta_pct": 100.0 * (corr_mae / target.base_mae - 1.0),
                    "win": int(corr_mae < target.base_mae),
                }
            )
    return rows


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    variants = list(dict.fromkeys(str(row["variant"]) for row in rows))
    for variant in variants:
        for group in ["core_wsd", "extra_control"]:
            sub = [row for row in rows if row["variant"] == variant and row["group"] == group]
            deltas = np.array([float(row["delta_pct"]) for row in sub], dtype=np.float64)
            out.append(
                {
                    "variant": variant,
                    "group": group,
                    "rows": len(sub),
                    "mean_delta": float(np.mean(deltas)),
                    "median_delta": float(np.median(deltas)),
                    "worst_delta": float(np.max(deltas)),
                    "wins": int(np.sum(deltas < 0.0)),
                    "nonharm": int(np.sum(deltas <= 1e-12)),
                }
            )
    return out


def by_variant_group(summary_rows: list[dict[str, object]], variant: str, group: str) -> dict[str, object]:
    for row in summary_rows:
        if row["variant"] == variant and row["group"] == group:
            return row
    raise KeyError((variant, group))


def fmt(row: dict[str, object]) -> tuple[str, str, str]:
    return (
        f"{float(row['mean_delta']):+.2f}%",
        f"{float(row['worst_delta']):+.2f}%",
        f"{int(row['wins'])}/{int(row['rows'])}",
    )


def write_report(summary_rows: list[dict[str, object]]) -> None:
    fixed_obs = by_variant_group(summary_rows, "fixed_lambda_obs", "core_wsd")
    fixed_20 = by_variant_group(summary_rows, "fixed_lambda_20", "core_wsd")
    strict_exact = by_variant_group(summary_rows, "strict_exact", "core_wsd")
    rounded = by_variant_group(summary_rows, "rounded_fast20", "core_wsd")
    linear = by_variant_group(summary_rows, "rounded_fast20_localized", "core_wsd")
    sqrt_row = by_variant_group(summary_rows, "rounded_fast20_sqrtlocalized", "core_wsd")
    rounded_ctrl = by_variant_group(summary_rows, "rounded_fast20", "extra_control")
    linear_ctrl = by_variant_group(summary_rows, "rounded_fast20_localized", "extra_control")
    sqrt_ctrl = by_variant_group(summary_rows, "rounded_fast20_sqrtlocalized", "extra_control")

    lines = [
        "# 解释性重置：只保留可讲清楚的残差模型\n\n",
        "> 2026-06-19 追加修正：此前把 DCT tau-free baseline 写成主线仍然不够好，因为 DCT 是 generic low-frequency residualizer，解释性弱。当前主线以 `MODEL_DECISION.md` 为准：使用 observation-bracket MPL-LD 作为机制化主候选，DCT 仅作为 performance reference / ablation。该版本用 \\(\\lambda_s=\\lambda_{\\mathrm{obs}}(1+q_s)/2\\) 替代旧的 `2.5`/`20` endpoints，用 \\(1/N_{\\mathrm{cal}}\\) 替代固定 \\(\\tau=0.05\\)。\n\n",
        "这份记录是对前面复杂模型的收缩。当前不再推荐把 gate、channel routing、正弦展开、curvature patch 或 `sqrt-localized` 作为主公式。"
        "这些变体可以留作探索和消融，但主线必须只回答一个问题：MPL 在 learning-rate 下降后是否遗漏了一个可由 schedule 推出的因果响应项。\n\n",
        "## 1. 主公式\n\n",
        "研究主线只保留\n\n",
        "\\[\n",
        "\\hat L_s(t)=L_{\\mathrm{MPL},s}(t)+\\hat\\kappa_s\\phi_{\\lambda_s,s}(t).\n",
        "\\]\n\n",
        "其中\n\n",
        "\\[\n",
        "\\phi_{\\lambda,s}(t)=\\sum_{u\\le t}\\exp[-\\lambda\\eta_u]\\frac{[\\eta_{u-1}-\\eta_u]_+}{\\eta_{\\max}}.\n",
        "\\]\n\n",
        "解释很简单：MPL 已经给出主趋势；当 LR 下降时，真实 loss 对新 LR 的响应不是瞬时完成的，因此残差中可能出现一个因果、只由过去 LR drop 激发的 relaxation response。"
        "唯一从 loss residual 拟合的量是非负幅度 \\(\\hat\\kappa_s\\)，而且只从 `cosine_72000.csv` residual 中估计。\n\n",
        "## 2. 训练与测试协议\n\n",
        "训练 / 校准：\n\n",
        "1. 对 `cosine_72000.csv` 计算 MPL prediction 与 residual \\(r_{\\cos}=L_{\\cos}-L_{\\mathrm{MPL},\\cos}\\)。\n",
        "2. 给定目标 schedule 的 \\(\\lambda_s\\)，在 cosine schedule 上构造同一个响应核 \\(\\phi_{\\lambda_s,\\cos}\\)。\n",
        "3. 只在 \\(t\\ge8000\\) 的点上做一维 partial regression：\n\n",
        "\\[\n",
        "\\hat\\kappa_s=\n",
        "\\frac{\\langle M_\\mu\\phi_{\\lambda_s,\\cos},M_\\mu r_{\\cos}\\rangle_+}\n",
        "{\\|M_\\mu\\phi_{\\lambda_s,\\cos}\\|_2^2+\\tau^2}.\n",
        "\\]\n\n",
        "这里 \\(M_\\mu\\) 只是去掉 cosine residual 中的低频 MPL drift；当前固定 \\(\\mu=0.01, \\tau=0.05\\)。\n\n",
        "测试 / 转移：\n\n",
        "1. 不使用目标 loss 拟合任何参数。\n",
        "2. 由目标 LR schedule 构造 \\(\\phi_{\\lambda_s,s}\\)。\n",
        "3. 输出 \\(L_{\\mathrm{MPL},s}+\\hat\\kappa_s\\phi_{\\lambda_s,s}\\)。\n\n",
        "## 3. 哪个版本可以作为主线\n\n",
        "本节保留早期 DCT-projected core decision 的结果，用作历史对照。当前不再把这里的 DCT tau-free 或 `strict_exact` 写作最终主线；最终主线见 `MODEL_DECISION.md`。\n\n",
        "| status | variant | extra structure | WSD mean | WSD worst | WSD wins | control note |\n",
        "|---|---|---|---:|---:|---:|---|\n",
    ]
    f_mean, f_worst, f_wins = fmt(fixed_obs)
    r20_mean, r20_worst, r20_wins = fmt(fixed_20)
    se_mean, se_worst, se_wins = fmt(strict_exact)
    rd_mean, rd_worst, rd_wins = fmt(rounded)
    lin_mean, lin_worst, lin_wins = fmt(linear)
    sq_mean, sq_worst, sq_wins = fmt(sqrt_row)
    lines += [
        f"| minimal sanity | fixed_lambda_obs | one observed half-life, no schedule geometry | {f_mean} | {f_worst} | {f_wins} | shows the mechanism already helps |\n",
        f"| minimal rounded | fixed_lambda_20 | round observed half-life to 20 | {r20_mean} | {r20_worst} | {r20_wins} | still one response kernel |\n",
        f"| recommended theory | strict_exact | \\(\\lambda_s\\) from drop concentration and exact observed half-life endpoints | {se_mean} | {se_worst} | {se_wins} | cleanest explainable WSD formula |\n",
        f"| performance variant | rounded_fast20 | same, but round fast endpoint to 20 | {rd_mean} | {rd_worst} | {rd_wins} | stronger WSD, slightly less pure |\n",
        f"| optional safety | rounded_fast20_localized | linear locality factor only for control-safety discussion | {lin_mean} | {lin_worst} | {lin_wins} | controls non-harm {int(linear_ctrl['nonharm'])}/{int(linear_ctrl['rows'])} |\n",
        f"| not main | rounded_fast20_sqrtlocalized | square-root locality amplitude | {sq_mean} | {sq_worst} | {sq_wins} | controls non-harm {int(sqrt_ctrl['nonharm'])}/{int(sqrt_ctrl['rows'])}, but weaker explanation |\n",
        "\n",
        "早期结论曾认为 `strict_exact` 或 `rounded_fast20` 可以作为严谨主线。这个判断现在下调为 DCT-based ablation：它们数值强，但 nuisance projection 解释性不足。"
        "`sqrt-localized` 仍不作为主公式，因为它的 square-root amplitude 解释不够硬。\n\n",
        "## 4. 为什么不再用复杂模型\n\n",
        "- gate / channel routing：分类规则很容易被理解成针对当前几条曲线调出来的经验开关，泛化解释弱。\n",
        "- 正弦展开：能贴合 cosine residual 的形状，但没有明确 schedule 机制，最容易过拟合。\n",
        "- curvature patch：可能有帮助，但和 MPL backbone 的误差边界纠缠，难以说明新增项到底是在修正 schedule lag 还是在重拟合 MPL。\n",
        "- `sqrt-localized`：虽然没有新增拟合参数，也能保护 short-cosine control，但 square-root 从 energy 到 amplitude 的论证偏软，不适合当核心贡献。\n\n",
        "这些结果可以作为负证据：它们说明单纯追求指标会不断诱导我们加入解释不稳的项。因此当前主线应宁愿少一点性能，也要保证公式每一步都能讲清楚。\n\n",
        "## 5. 需要诚实承认的局限\n\n",
        f"- 不加 locality 时，`rounded_fast20` 对 extra controls 的 mean/worst 为 {float(rounded_ctrl['mean_delta']):+.2f}% / {float(rounded_ctrl['worst_delta']):+.2f}%。这说明该公式不是 universal schedule predictor，而是针对 cosine-to-WSD transfer 的机制修正。\n",
        "- \\(\\lambda_s\\) 的 slow endpoint 仍含有 2.5-observation 这个 protocol choice；虽然 sensitivity 是宽的，但还不是严格定理。\n",
        "- \\(\\mu=0.01,\\tau=0.05\\) 是 identifiability protocol，不是从第一性原理唯一推出。\n",
        "- 当前证据仍只来自已有曲线，最终需要新 schedule 或新训练 run 做外部验证。\n\n",
        "## 6. 当前写作建议\n\n",
        "主文只讲 `MPL + causal LR-drop response`，但 nuisance removal 应优先使用 MPL-LD tangent 解释，而不是 DCT。"
        "如果老师追问 short-cosine control，再把 linear locality 作为 boundary condition，而不是把它写成核心理论项。\n",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "INTERPRETABILITY_RESET.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    lambda_obs = math.log(2.0) / (iem.PEAK_LR * 128)
    rows: list[dict[str, object]] = []
    rows.extend(fixed_lambda_rows("fixed_lambda_obs", lambda_obs))
    rows.extend(fixed_lambda_rows("fixed_lambda_20", 20.0))
    for variant in [
        "strict_exact",
        "rounded_fast20",
        "rounded_fast20_localized",
        "rounded_fast20_sqrtlocalized",
    ]:
        rows.extend(strict.run_variant(variant))
    summary_rows = summarize(rows)
    iem.write_csv(OUT_DIR / "core_decision_details.csv", rows)
    iem.write_csv(OUT_DIR / "core_decision_summary.csv", summary_rows)
    write_report(summary_rows)
    print(f"wrote {OUT_DIR / 'core_decision_details.csv'}")
    print(f"wrote {OUT_DIR / 'core_decision_summary.csv'}")
    print(f"wrote {OUT_DIR / 'INTERPRETABILITY_RESET.md'}")


if __name__ == "__main__":
    main()
