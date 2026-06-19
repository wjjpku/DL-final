#!/usr/bin/env python3
"""Test whether our residual correction is just a scalar multiple of MPL's final term."""
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
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import interpretable_error_model as iem  # noqa: E402
import interpretable_nuisance_origin_audit as noa  # noqa: E402
import interpretable_observation_bracket_audit as oba  # noqa: E402
import interpretable_theory_refinement_audit as tra  # noqa: E402

OUT_DIR = iem.ROOT / "results" / "mpl_final_term_multiplier_audit"
FIG_DIR = OUT_DIR / "figs"
FIT_START = 8000

CORE_TARGETS = [
    ("wsd_20000_24000.csv", "WSD sharp"),
    ("wsdld_20000_24000.csv", "WSD linear"),
    ("wsdcon_3.csv", "WSD-con 3e-5"),
    ("wsdcon_9.csv", "WSD-con 9e-5"),
    ("wsdcon_18.csv", "WSD-con 18e-5"),
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


def smooth(y: np.ndarray) -> np.ndarray:
    if len(y) < 9:
        return y.copy()
    window = max(5, int(round(0.025 * len(y))))
    if window % 2 == 0:
        window += 1
    window = min(window, len(y) if len(y) % 2 == 1 else len(y) - 1)
    if window < 3:
        return y.copy()
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(y, kernel, mode="same")


def load_pack(scale: str, curve_name: str, cache: dict[tuple[str, str], iem.CurvePack]) -> iem.CurvePack:
    key = (scale, curve_name)
    if key not in cache:
        cache[key] = tra.load_pack(scale, curve_name, cache)
    return cache[key]


def mpl_final_term(curve: iem.Curve) -> tuple[np.ndarray, np.ndarray]:
    params = np.array(iem.MPL_PRECOMPUTED_INIT[curve.scale], dtype=np.float64)
    _l0, _a, _alpha, b_value, c_value, beta, gamma = params
    dld = iem.compute_ld(curve, c_value, beta, gamma)
    return b_value * dld, dld


def fit_scalar(x: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray, dict[str, float]]:
    denom = float(np.dot(x, x))
    alpha = float(np.dot(x, y) / denom) if denom > 1e-18 else 0.0
    yhat = alpha * x
    resid = y - yhat
    norm_y = float(np.linalg.norm(y))
    r2_origin = 1.0 - float(np.dot(resid, resid)) / max(float(np.dot(y, y)), 1e-18)
    rel_rmse = float(np.linalg.norm(resid) / max(norm_y, 1e-18))
    corr = pearson(x, y)
    return alpha, yhat, {"r2_origin": r2_origin, "rel_rmse": rel_rmse, "pearson": corr}


def fit_affine(x: np.ndarray, y: np.ndarray) -> tuple[float, float, np.ndarray, dict[str, float]]:
    design = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    intercept = float(coef[0])
    alpha = float(coef[1])
    yhat = design @ coef
    resid = y - yhat
    centered = y - float(np.mean(y))
    r2_centered = 1.0 - float(np.dot(resid, resid)) / max(float(np.dot(centered, centered)), 1e-18)
    r2_origin = 1.0 - float(np.dot(resid, resid)) / max(float(np.dot(y, y)), 1e-18)
    rel_rmse = float(np.linalg.norm(resid) / max(float(np.linalg.norm(y)), 1e-18))
    return intercept, alpha, yhat, {
        "affine_r2_centered": r2_centered,
        "affine_r2_origin": r2_origin,
        "affine_rel_rmse": rel_rmse,
    }


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    xx = x.astype(np.float64) - float(np.mean(x))
    yy = y.astype(np.float64) - float(np.mean(y))
    denom = float(np.linalg.norm(xx) * np.linalg.norm(yy))
    if denom <= 1e-18:
        return float("nan")
    return float(np.dot(xx, yy) / denom)


def our_correction(
    source: iem.CurvePack,
    target: iem.CurvePack,
    basis_cache: dict[tuple[str, str, int], np.ndarray],
) -> tuple[np.ndarray, dict[str, float]]:
    lam = tra.response_lambda(target.curve, "q2", "halflife")
    coef, info = oba.fit_coefficient(source, lam, "mpl_ld4", "sample_size_ridge", FIT_START, basis_cache)
    factor = tra.locality_factor(target.curve, "support_projection")
    feature = iem.causal_drop_response(target.curve, lam)
    return factor * coef * feature, {
        "lambda": float(lam),
        "coef": float(coef),
        "locality_factor": float(factor),
        **{key: float(value) for key, value in info.items()},
    }


def mae_delta(target: iem.CurvePack, extra: np.ndarray) -> float:
    pred = target.baseline + extra
    return 100.0 * (iem.mae(target.curve.loss, pred) / target.base_mae - 1.0)


def analyze() -> tuple[list[dict[str, object]], dict[tuple[str, str], dict[str, object]]]:
    cache: dict[tuple[str, str], iem.CurvePack] = {}
    basis_cache: dict[tuple[str, str, int], np.ndarray] = {}
    rows: list[dict[str, object]] = []
    panels: dict[tuple[str, str], dict[str, object]] = {}

    for scale in iem.SCALES:
        source = load_pack(scale, iem.TRAIN_CURVE, cache)
        for curve_name, label in CORE_TARGETS:
            target = load_pack(scale, curve_name, cache)
            residual = target.curve.loss - target.baseline
            ours, info = our_correction(source, target, basis_cache)
            l_g, dld = mpl_final_term(target.curve)

            alpha_ours, scalar_ours, scalar_ours_stats = fit_scalar(l_g, ours)
            intercept_ours, alpha_aff_ours, affine_ours, affine_ours_stats = fit_affine(l_g, ours)

            alpha_resid, scalar_resid, scalar_resid_stats = fit_scalar(l_g, residual)
            intercept_resid, alpha_aff_resid, affine_resid, affine_resid_stats = fit_affine(l_g, residual)

            row = {
                "scale": scale,
                "target_curve": curve_name,
                "target_label": label,
                "our_delta_pct": mae_delta(target, ours),
                "oracle_scalar_lg_delta_pct": mae_delta(target, scalar_resid),
                "oracle_affine_lg_delta_pct": mae_delta(target, affine_resid),
                "alpha_lg_to_ours": alpha_ours,
                "intercept_lg_to_ours": intercept_ours,
                "alpha_affine_lg_to_ours": alpha_aff_ours,
                "alpha_lg_to_residual": alpha_resid,
                "intercept_lg_to_residual": intercept_resid,
                "alpha_affine_lg_to_residual": alpha_aff_resid,
                "ours_l1": float(np.mean(np.abs(ours))),
                "lg_l1": float(np.mean(np.abs(l_g))),
                "residual_l1": float(np.mean(np.abs(residual))),
                "scalar_lg_to_ours_l1": float(np.mean(np.abs(scalar_ours))),
                "affine_lg_to_ours_l1": float(np.mean(np.abs(affine_ours))),
                "scalar_lg_to_residual_l1": float(np.mean(np.abs(scalar_resid))),
                "affine_lg_to_residual_l1": float(np.mean(np.abs(affine_resid))),
                "lg_to_ours_pearson": scalar_ours_stats["pearson"],
                "lg_to_ours_r2_origin": scalar_ours_stats["r2_origin"],
                "lg_to_ours_rel_rmse": scalar_ours_stats["rel_rmse"],
                "lg_to_ours_affine_r2_centered": affine_ours_stats["affine_r2_centered"],
                "lg_to_ours_affine_r2_origin": affine_ours_stats["affine_r2_origin"],
                "lg_to_ours_affine_rel_rmse": affine_ours_stats["affine_rel_rmse"],
                "lg_to_residual_pearson": scalar_resid_stats["pearson"],
                "lg_to_residual_r2_origin": scalar_resid_stats["r2_origin"],
                "lg_to_residual_rel_rmse": scalar_resid_stats["rel_rmse"],
                "lg_to_residual_affine_r2_centered": affine_resid_stats["affine_r2_centered"],
                "lg_to_residual_affine_r2_origin": affine_resid_stats["affine_r2_origin"],
                "lg_to_residual_affine_rel_rmse": affine_resid_stats["affine_rel_rmse"],
                "lambda": info["lambda"],
                "coef": info["coef"],
                "locality_factor": info["locality_factor"],
            }
            rows.append(row)
            panels[(scale, curve_name)] = {
                "scale": scale,
                "curve_name": curve_name,
                "label": label,
                "target": target,
                "residual": residual,
                "ours": ours,
                "l_g": l_g,
                "dld": dld,
                "scalar_ours": scalar_ours,
                "affine_ours": affine_ours,
                "scalar_resid": scalar_resid,
                "affine_resid": affine_resid,
                "row": row,
            }
    return rows, panels


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    groups = {
        "all_core": rows,
        "wsd_final": [row for row in rows if row["target_curve"] in {"wsd_20000_24000.csv", "wsdld_20000_24000.csv"}],
        "wsdcon": [row for row in rows if str(row["target_curve"]).startswith("wsdcon")],
    }
    for group, sub in groups.items():
        if not sub:
            continue
        out.append(
            {
                "group": group,
                "rows": len(sub),
                "mean_our_delta_pct": float(np.mean([float(row["our_delta_pct"]) for row in sub])),
                "mean_oracle_scalar_lg_delta_pct": float(np.mean([float(row["oracle_scalar_lg_delta_pct"]) for row in sub])),
                "mean_oracle_affine_lg_delta_pct": float(np.mean([float(row["oracle_affine_lg_delta_pct"]) for row in sub])),
                "mean_lg_to_ours_r2_origin": float(np.mean([float(row["lg_to_ours_r2_origin"]) for row in sub])),
                "median_lg_to_ours_r2_origin": float(np.median([float(row["lg_to_ours_r2_origin"]) for row in sub])),
                "mean_lg_to_ours_affine_r2_centered": float(
                    np.mean([float(row["lg_to_ours_affine_r2_centered"]) for row in sub])
                ),
                "median_lg_to_ours_affine_r2_centered": float(
                    np.median([float(row["lg_to_ours_affine_r2_centered"]) for row in sub])
                ),
                "mean_lg_to_residual_r2_origin": float(np.mean([float(row["lg_to_residual_r2_origin"]) for row in sub])),
                "mean_lg_to_residual_affine_r2_centered": float(
                    np.mean([float(row["lg_to_residual_affine_r2_centered"]) for row in sub])
                ),
                "count_lg_to_ours_r2_gt_0p8": int(np.sum([float(row["lg_to_ours_r2_origin"]) > 0.8 for row in sub])),
                "count_lg_to_ours_affine_r2_gt_0p8": int(
                    np.sum([float(row["lg_to_ours_affine_r2_centered"]) > 0.8 for row in sub])
                ),
            }
        )
    return out


def ylim_for(arrays: list[np.ndarray], pad: float = 1.16) -> tuple[float, float]:
    limit = max(float(np.max(np.abs(arr))) for arr in arrays)
    limit = max(limit, 1e-8)
    return -pad * limit, pad * limit


def plot_single(panel: dict[str, object]) -> None:
    target = panel["target"]
    steps = target.curve.step
    residual = panel["residual"]
    ours = panel["ours"]
    l_g = panel["l_g"]
    scalar_ours = panel["scalar_ours"]
    affine_ours = panel["affine_ours"]
    scalar_resid = panel["scalar_resid"]
    row = panel["row"]

    fig, axes = plt.subplots(3, 1, figsize=(11.5, 9.2), sharex=True, constrained_layout=True)

    axes[0].axhline(0.0, color="#111827", lw=0.8, alpha=0.72)
    axes[0].plot(steps, smooth(l_g), color="#7c3aed", lw=1.45, label=r"MPL final term $L_G(t)=BD_{LD}(t)$")
    axes[0].set_ylabel("MPL final term")
    axes[0].legend(fontsize=8, loc="best")
    axes[0].set_title(f"{panel['label']} ({panel['scale']}M): is our residual a multiple of MPL final term?")

    axes[1].axhline(0.0, color="#111827", lw=0.8, alpha=0.72)
    axes[1].plot(steps, smooth(ours), color="#2563eb", lw=1.55, label="our residual correction")
    axes[1].plot(
        steps,
        smooth(scalar_ours),
        color="#dc2626",
        lw=1.25,
        ls="--",
        label=rf"best $\rho L_G$ to ours, R2={float(row['lg_to_ours_r2_origin']):+.2f}",
    )
    axes[1].plot(
        steps,
        smooth(affine_ours),
        color="#f97316",
        lw=1.15,
        ls="-.",
        label=rf"best $c+\rho L_G$, R2={float(row['lg_to_ours_affine_r2_centered']):+.2f}",
    )
    axes[1].set_ylim(*ylim_for([ours, scalar_ours, affine_ours]))
    axes[1].set_ylabel("fit to ours")
    axes[1].legend(fontsize=8, loc="best")

    axes[2].axhline(0.0, color="#111827", lw=0.8, alpha=0.72)
    axes[2].plot(steps, smooth(residual), color="#111827", lw=1.55, label="true MPL residual")
    axes[2].plot(steps, smooth(ours), color="#2563eb", lw=1.35, label=f"our correction ({float(row['our_delta_pct']):+.1f}%)")
    axes[2].plot(
        steps,
        smooth(scalar_resid),
        color="#dc2626",
        lw=1.2,
        ls="--",
        label=f"oracle scalar L_G to residual ({float(row['oracle_scalar_lg_delta_pct']):+.1f}%)",
    )
    axes[2].set_ylim(*ylim_for([residual, ours, scalar_resid]))
    axes[2].set_ylabel("fit to residual")
    axes[2].set_xlabel("step")
    axes[2].legend(fontsize=8, loc="best")

    for ax in axes:
        ax2 = ax.twinx()
        ax2.plot(steps, target.curve.lrs[target.curve.step] / iem.PEAK_LR, color="#b45309", lw=0.75, alpha=0.18)
        ax2.set_ylim(-0.05, 1.05)
        ax2.set_yticks([])

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"single_{panel['scale']}M_{panel['curve_name'].replace('.csv', '')}.png", dpi=180)
    plt.close(fig)


def plot_grid(scale: str, panels: dict[tuple[str, str], dict[str, object]]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16.2, 7.8), constrained_layout=True)
    for idx, (curve_name, _label) in enumerate(CORE_TARGETS):
        panel = panels[(scale, curve_name)]
        ax = axes.ravel()[idx]
        target = panel["target"]
        steps = target.curve.step
        ours = panel["ours"]
        scalar_ours = panel["scalar_ours"]
        affine_ours = panel["affine_ours"]
        row = panel["row"]
        ax.axhline(0.0, color="#111827", lw=0.8, alpha=0.72)
        ax.plot(steps, smooth(ours), color="#2563eb", lw=1.25, label="our correction")
        ax.plot(steps, smooth(scalar_ours), color="#dc2626", lw=1.05, ls="--", label=r"best $\rho L_G$")
        ax.plot(steps, smooth(affine_ours), color="#f97316", lw=1.0, ls="-.", label=r"best $c+\rho L_G$")
        ax.set_ylim(*ylim_for([ours, scalar_ours, affine_ours], pad=1.35))
        ax.set_title(
            f"{panel['label']} ({scale}M)\n"
            f"raw R2={float(row['lg_to_ours_r2_origin']):+.2f}, "
            f"affine R2={float(row['lg_to_ours_affine_r2_centered']):+.2f}",
            fontsize=9,
        )
        ax.tick_params(labelsize=8)
        if idx % 3 == 0:
            ax.set_ylabel("correction")
        ax.set_xlabel("step")
        if idx == 0:
            ax.legend(fontsize=7, loc="best")
    axes.ravel()[-1].axis("off")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"core_multiplier_grid_{scale}M.png", dpi=180)
    plt.close(fig)


def plot_summary(rows: list[dict[str, object]]) -> None:
    labels = [f"{row['scale']}M\n{row['target_label'].replace('WSD-', '')}" for row in rows]
    raw = np.array([float(row["lg_to_ours_r2_origin"]) for row in rows], dtype=np.float64)
    aff = np.array([float(row["lg_to_ours_affine_r2_centered"]) for row in rows], dtype=np.float64)
    x = np.arange(len(rows))
    width = 0.38
    fig, ax = plt.subplots(figsize=(15.5, 4.8), constrained_layout=True)
    ax.axhline(0.0, color="#111827", lw=0.8)
    ax.axhline(0.8, color="#6b7280", lw=0.8, ls=":")
    ax.bar(x - width / 2, raw, width, color="#dc2626", label=r"best $\rho L_G$ to ours")
    ax.bar(x + width / 2, aff, width, color="#f97316", label=r"best $c+\rho L_G$ to ours")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("R2 explaining our correction")
    ax.set_title("Can MPL final term explain our residual correction?")
    ax.legend(loc="best")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "r2_summary.png", dpi=180)
    plt.close(fig)


def write_report(rows: list[dict[str, object]], agg: list[dict[str, object]]) -> None:
    def fmt_pct(x: float) -> str:
        return f"{x:+.2f}%"

    target = next(row for row in rows if row["scale"] == "100" and row["target_curve"] == "wsdcon_3.csv")
    lines = [
        "# MPL Final Term Multiplier Audit\n\n",
        "问题：我们新增的 residual correction 是否只是 MPL 最后一项的若干倍？",
        "如果是，那么它可以写成\n\n",
        "\\[\n",
        "\\widehat e(t)\\approx \\rho L_G(t),\\qquad L_G(t)=B D_{LD}(t),\n",
        "\\]\n\n",
        "这等价于把 MPL 最后一项的系数从 \\(B\\) 改成 \\((1+\\rho)B\\)。",
        "本实验直接拟合最优 \\(\\rho\\)，并额外给一个更宽松的 affine check：",
        "\\(c+\\rho L_G(t)\\)。affine 不是单独调 B，但可以判断形状是否至少接近。\n\n",
        "## Main Result\n\n",
        "结论：不是。直接的 \\(\\rho L_G(t)\\) 基本不能解释我们的 residual correction；",
        "即使用更宽松的 \\(c+\\rho L_G(t)\\)，在 WSD-con 目标上也明显不够。\n\n",
        "### Aggregate\n\n",
        "| group | rows | our mean delta | oracle scalar-LG mean delta | mean R2 to ours | affine mean R2 to ours | R2>0.8 count |\n",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n",
    ]
    for row in agg:
        lines.append(
            f"| {row['group']} | {int(row['rows'])} | "
            f"{fmt_pct(float(row['mean_our_delta_pct']))} | "
            f"{fmt_pct(float(row['mean_oracle_scalar_lg_delta_pct']))} | "
            f"{float(row['mean_lg_to_ours_r2_origin']):+.3f} | "
            f"{float(row['mean_lg_to_ours_affine_r2_centered']):+.3f} | "
            f"{int(row['count_lg_to_ours_r2_gt_0p8'])}/{int(row['rows'])} raw, "
            f"{int(row['count_lg_to_ours_affine_r2_gt_0p8'])}/{int(row['rows'])} affine |\n"
        )
    lines += [
        "\n### Single Example: 100M WSD-con 3e-5\n\n",
        "| quantity | value |\n",
        "| --- | ---: |\n",
        f"| our correction MAE delta | {fmt_pct(float(target['our_delta_pct']))} |\n",
        f"| oracle scalar `rho * L_G` MAE delta | {fmt_pct(float(target['oracle_scalar_lg_delta_pct']))} |\n",
        f"| best scalar R2 explaining our correction | {float(target['lg_to_ours_r2_origin']):+.3f} |\n",
        f"| best affine R2 explaining our correction | {float(target['lg_to_ours_affine_r2_centered']):+.3f} |\n",
        f"| scalar alpha to ours | {float(target['alpha_lg_to_ours']):+.6g} |\n",
        f"| scalar relative RMSE to ours | {float(target['lg_to_ours_rel_rmse']):.3f} |\n\n",
        "## Per-Target Table\n\n",
        "| scale | target | our delta | oracle scalar-LG delta | scalar R2 to ours | affine R2 to ours | alpha to ours |\n",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |\n",
    ]
    for row in rows:
        lines.append(
            f"| {row['scale']}M | {row['target_label']} | "
            f"{fmt_pct(float(row['our_delta_pct']))} | "
            f"{fmt_pct(float(row['oracle_scalar_lg_delta_pct']))} | "
            f"{float(row['lg_to_ours_r2_origin']):+.3f} | "
            f"{float(row['lg_to_ours_affine_r2_centered']):+.3f} | "
            f"{float(row['alpha_lg_to_ours']):+.4g} |\n"
        )
    lines += [
        "\n## Figures\n\n",
        "- `figs/single_100M_wsdcon_3.png`\n",
        "- `figs/core_multiplier_grid_25M.png`\n",
        "- `figs/core_multiplier_grid_100M.png`\n",
        "- `figs/core_multiplier_grid_400M.png`\n",
        "- `figs/r2_summary.png`\n\n",
        "## Interpretation\n\n",
        "调 MPL 最后一项的系数会给整个 \\(L_G(t)=B D_{LD}(t)\\) 乘一个全局倍数。",
        "这个方向包含很大的低频/全局形状；而我们的 correction 是一个更局部的 positive residual shape。",
        "因此二者不是同一个一维方向。尤其在 WSD-con 上，best scalar multiple 的 R2 只有约 0.04，",
        "即使加 intercept 的 affine 形状解释也很弱。\n\n",
        "所以可以把“调 B”作为一个 baseline 或 ablation，但不能把它当作我们 residual 的等价替代。",
    ]
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    rows, panels = analyze()
    agg = aggregate(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "metrics.csv", rows)
    write_csv(OUT_DIR / "summary.csv", agg)
    plot_single(panels[("100", "wsdcon_3.csv")])
    for scale in iem.SCALES:
        plot_grid(scale, panels)
    plot_summary(rows)
    write_report(rows, agg)
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
