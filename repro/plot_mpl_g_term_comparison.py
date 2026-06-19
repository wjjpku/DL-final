#!/usr/bin/env python3
"""Compare MPL's final G-term with the residual correction."""
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

OUT_DIR = iem.ROOT / "results" / "mpl_g_term_comparison"
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


def mpl_g_term(curve: iem.Curve) -> np.ndarray:
    params = np.array(iem.MPL_PRECOMPUTED_INIT[curve.scale], dtype=np.float64)
    _l0, _a, _alpha, b_value, c_value, beta, gamma = params
    return b_value * iem.compute_ld(curve, c_value, beta, gamma)


def first_lr_drop_step(curve: iem.Curve) -> int:
    diff = np.diff(curve.lrs)
    idx = np.flatnonzero(diff < -1e-18)
    return int(idx[0] + 1) if idx.size else int(curve.step[0])


def value_before_or_at(steps: np.ndarray, values: np.ndarray, step: int) -> float:
    idx = np.flatnonzero(steps <= step)
    if idx.size:
        return float(values[int(idx[-1])])
    return float(values[0])


def normalize_like(y: np.ndarray, ref: np.ndarray) -> np.ndarray:
    scale = float(np.max(np.abs(ref)))
    denom = float(np.max(np.abs(y)))
    if denom <= 1e-18 or scale <= 1e-18:
        return np.zeros_like(y)
    return y * (scale / denom)


def pearson_after_drop(x: np.ndarray, y: np.ndarray, steps: np.ndarray, drop_step: int) -> float:
    mask = steps >= drop_step
    xx = x[mask].astype(np.float64)
    yy = y[mask].astype(np.float64)
    if len(xx) < 3:
        return float("nan")
    xx = xx - float(np.mean(xx))
    yy = yy - float(np.mean(yy))
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
    correction = factor * coef * feature
    return correction, {
        "lambda": float(lam),
        "coef": float(coef),
        "locality_factor": float(factor),
        **{key: float(value) for key, value in info.items()},
    }


def analyze() -> tuple[list[dict[str, object]], dict[tuple[str, str], dict[str, object]]]:
    cache: dict[tuple[str, str], iem.CurvePack] = {}
    basis_cache: dict[tuple[str, str, int], np.ndarray] = {}
    rows: list[dict[str, object]] = []
    panels: dict[tuple[str, str], dict[str, object]] = {}

    for scale in iem.SCALES:
        source = load_pack(scale, iem.TRAIN_CURVE, cache)
        for curve_name, label in CORE_TARGETS:
            target = load_pack(scale, curve_name, cache)
            steps = target.curve.step
            residual = target.curve.loss - target.baseline
            correction, info = our_correction(source, target, basis_cache)
            g_raw = mpl_g_term(target.curve)
            drop_step = first_lr_drop_step(target.curve)
            g_ref = value_before_or_at(steps, g_raw, drop_step)
            g_delta = g_raw - g_ref
            g_drop_magnitude = -g_delta
            g_shape_scaled = normalize_like(g_drop_magnitude, residual)
            corr_mae = iem.mae(target.curve.loss, target.baseline + correction)
            delta_pct = 100.0 * (corr_mae / target.base_mae - 1.0)

            row = {
                "scale": scale,
                "target_curve": curve_name,
                "target_label": label,
                "drop_step": drop_step,
                "base_mae": target.base_mae,
                "corr_mae": corr_mae,
                "delta_pct": delta_pct,
                "lambda": info["lambda"],
                "coef": info["coef"],
                "locality_factor": info["locality_factor"],
                "g_raw_min": float(np.min(g_raw)),
                "g_raw_max": float(np.max(g_raw)),
                "g_delta_min": float(np.min(g_delta)),
                "g_delta_max": float(np.max(g_delta)),
                "g_drop_l1": float(np.mean(np.abs(g_drop_magnitude))),
                "g_drop_max": float(np.max(g_drop_magnitude)),
                "residual_l1": float(np.mean(np.abs(residual))),
                "residual_max": float(np.max(residual)),
                "our_correction_l1": float(np.mean(np.abs(correction))),
                "our_correction_max": float(np.max(correction)),
                "g_drop_to_residual_l1": float(np.mean(np.abs(g_drop_magnitude)) / max(np.mean(np.abs(residual)), 1e-18)),
                "our_to_residual_l1": float(np.mean(np.abs(correction)) / max(np.mean(np.abs(residual)), 1e-18)),
                "corr_gdrop_residual_after_drop": pearson_after_drop(g_drop_magnitude, residual, steps, drop_step),
                "corr_our_residual_after_drop": pearson_after_drop(correction, residual, steps, drop_step),
            }
            rows.append(row)
            panels[(scale, curve_name)] = {
                "scale": scale,
                "curve_name": curve_name,
                "label": label,
                "target": target,
                "steps": steps,
                "residual": residual,
                "correction": correction,
                "g_raw": g_raw,
                "g_delta": g_delta,
                "g_drop_magnitude": g_drop_magnitude,
                "g_shape_scaled": g_shape_scaled,
                "drop_step": drop_step,
                "row": row,
            }
    return rows, panels


def ylim_for(arrays: list[np.ndarray], pad: float = 1.12) -> tuple[float, float]:
    limit = max(float(np.max(np.abs(a))) for a in arrays)
    limit = max(limit, 1e-8)
    return -pad * limit, pad * limit


def plot_single(panel: dict[str, object]) -> None:
    steps = panel["steps"]
    target = panel["target"]
    residual = panel["residual"]
    correction = panel["correction"]
    g_raw = panel["g_raw"]
    g_delta = panel["g_delta"]
    g_drop = panel["g_drop_magnitude"]
    g_scaled = panel["g_shape_scaled"]
    row = panel["row"]

    fig, axes = plt.subplots(3, 1, figsize=(11.2, 9.2), sharex=True, constrained_layout=True)

    axes[0].axhline(0.0, color="#111827", lw=0.8, alpha=0.7)
    axes[0].plot(steps, smooth(g_raw), color="#7c3aed", lw=1.55, label=r"raw MPL final term $B D_{LD}(t)$")
    axes[0].plot(steps, smooth(g_delta), color="#a855f7", lw=1.25, ls="--", label=r"$\Delta B D_{LD}(t)$ from drop start")
    axes[0].set_ylabel("MPL G-term")
    axes[0].legend(fontsize=8, loc="best")
    axes[0].set_title(f"{panel['label']} ({panel['scale']}M): MPL final G-term vs our residual correction")

    axes[1].axhline(0.0, color="#111827", lw=0.8, alpha=0.7)
    axes[1].plot(steps, smooth(residual), color="#111827", lw=1.55, label="true MPL residual")
    axes[1].plot(steps, smooth(correction), color="#2563eb", lw=1.45, label=f"our correction ({float(row['delta_pct']):+.1f}%)")
    axes[1].plot(steps, smooth(g_drop), color="#7c3aed", lw=1.15, ls="--", label=r"$-\Delta B D_{LD}$ raw magnitude")
    axes[1].set_ylim(*ylim_for([residual, correction, g_drop]))
    axes[1].set_ylabel("loss units")
    axes[1].legend(fontsize=8, loc="best")

    axes[2].axhline(0.0, color="#111827", lw=0.8, alpha=0.7)
    axes[2].plot(steps, smooth(residual), color="#111827", lw=1.55, label="true MPL residual")
    axes[2].plot(steps, smooth(correction), color="#2563eb", lw=1.45, label="our correction")
    axes[2].plot(steps, smooth(g_scaled), color="#7c3aed", lw=1.25, ls="--", label=r"$-\Delta B D_{LD}$ scaled to residual max")
    axes[2].set_ylim(*ylim_for([residual, correction, g_scaled], pad=1.35))
    axes[2].set_ylabel("shape comparison")
    axes[2].set_xlabel("step")
    axes[2].legend(fontsize=8, loc="best")

    for ax in axes:
        ax.axvline(panel["drop_step"], color="#b45309", lw=0.8, alpha=0.28)
        ax2 = ax.twinx()
        ax2.plot(steps, target.curve.lrs[target.curve.step] / iem.PEAK_LR, color="#b45309", lw=0.75, alpha=0.18)
        ax2.set_ylim(-0.05, 1.05)
        ax2.set_yticks([])

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"single_{panel['scale']}M_{panel['curve_name'].replace('.csv', '')}.png", dpi=180)
    plt.close(fig)


def plot_core_grid(scale: str, panels: dict[tuple[str, str], dict[str, object]]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16.2, 7.8), constrained_layout=True)
    for idx, (curve_name, _label) in enumerate(CORE_TARGETS):
        panel = panels[(scale, curve_name)]
        ax = axes.ravel()[idx]
        steps = panel["steps"]
        residual = panel["residual"]
        correction = panel["correction"]
        g_scaled = panel["g_shape_scaled"]
        row = panel["row"]
        ax.axhline(0.0, color="#111827", lw=0.8, alpha=0.72)
        ax.plot(steps, smooth(residual), color="#111827", lw=1.25, label="MPL residual")
        ax.plot(steps, smooth(correction), color="#2563eb", lw=1.15, label="our correction")
        ax.plot(steps, smooth(g_scaled), color="#7c3aed", lw=1.05, ls="--", label=r"scaled $-\Delta B D_{LD}$")
        ax.axvline(panel["drop_step"], color="#b45309", lw=0.8, alpha=0.24)
        ax.set_ylim(*ylim_for([residual, correction, g_scaled], pad=1.35))
        ax.set_title(
            f"{panel['label']} ({scale}M)\n"
            f"corr(G,res)={float(row['corr_gdrop_residual_after_drop']):+.2f}, "
            f"corr(ours,res)={float(row['corr_our_residual_after_drop']):+.2f}",
            fontsize=9,
        )
        ax.tick_params(labelsize=8)
        if idx % 3 == 0:
            ax.set_ylabel("shape-scaled residual")
        ax.set_xlabel("step")
        if idx == 0:
            ax.legend(fontsize=7, loc="best")
    axes.ravel()[-1].axis("off")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"core_shape_grid_{scale}M.png", dpi=180)
    plt.close(fig)


def write_report(rows: list[dict[str, object]]) -> None:
    target_rows = [row for row in rows if row["scale"] == "100" and row["target_curve"] == "wsdcon_3.csv"]
    target = target_rows[0]
    lines = [
        "# MPL Final G-Term Comparison\n\n",
        "这里的 MPL 最后一项指代码中的\n\n",
        "\\[\n",
        "L_G(t)=B\\,D_{LD}(t),\\qquad ",
        "D_{LD}(t)=\\sum_{k\\le t}\\Delta\\eta_k\\,G\\!\\left(\\eta_k^{-\\gamma}(S(t)-S(k))\\right),\n",
        "\\]\n\n",
        "其中 \\(G(x)=1-(1+Cx)^{-\\beta}\\)。这项已经包含在 MPL baseline 里，不是我们新增的 residual correction。\n\n",
        "为了和正 residual 比较，图中还画了\n\n",
        "\\[\n",
        "-\\Delta L_G(t)=-\\bigl(L_G(t)-L_G(t_{drop})\\bigr),\n",
        "\\]\n\n",
        "它表示 MPL 最后一项在 LR drop 之后预测的 quasi-static loss decrease。我们的误差项则是\n\n",
        "\\[\n",
        "\\widehat e(t)=a_s\\widehat\\kappa_s\\phi_{\\lambda_s}(t).\n",
        "\\]\n\n",
        "## Key Single Example\n\n",
        "固定 `100M WSD-con 3e-5`：\n\n",
        "| quantity | value |\n",
        "| --- | ---: |\n",
        f"| our MAE delta | {float(target['delta_pct']):+.2f}% |\n",
        f"| raw `-Delta L_G` / true residual L1 | {float(target['g_drop_to_residual_l1']):.2f}x |\n",
        f"| our correction / true residual L1 | {float(target['our_to_residual_l1']):.2f}x |\n",
        f"| corr(`-Delta L_G`, residual) after drop | {float(target['corr_gdrop_residual_after_drop']):+.3f} |\n",
        f"| corr(our correction, residual) after drop | {float(target['corr_our_residual_after_drop']):+.3f} |\n\n",
        "## All Core Targets\n\n",
        "| scale | target | `-Delta L_G` / residual L1 | our / residual L1 | corr G-res | corr ours-res | our delta |\n",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |\n",
    ]
    for row in rows:
        lines.append(
            f"| {row['scale']}M | {row['target_label']} | "
            f"{float(row['g_drop_to_residual_l1']):.2f}x | "
            f"{float(row['our_to_residual_l1']):.2f}x | "
            f"{float(row['corr_gdrop_residual_after_drop']):+.2f} | "
            f"{float(row['corr_our_residual_after_drop']):+.2f} | "
            f"{float(row['delta_pct']):+.2f}% |\n"
        )
    lines += [
        "\n## Figures\n\n",
        "- `figs/single_100M_wsdcon_3.png`\n",
        "- `figs/core_shape_grid_25M.png`\n",
        "- `figs/core_shape_grid_100M.png`\n",
        "- `figs/core_shape_grid_400M.png`\n\n",
        "## Reading\n\n",
        "MPL 最后一项主要解释的是 LR schedule 变化导致的 quasi-static equilibrium movement：",
        "LR 降低后，它给出一个较大的、平滑的 loss decrease，并且这个 decrease 是 MPL baseline 的一部分。",
        "我们现在拟合的 residual correction 不是重复这件事，而是在 MPL 已经下降以后补一个正的 lag residual，",
        "表示真实 loss 没有立即跟上 MPL 的新 equilibrium。\n\n",
        "因此二者形状方向不同：`-Delta L_G` 更像累计的平滑 equilibrium shift；",
        "我们的误差项更像 cooldown 后的正向 transient / relaxation lag。",
    ]
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    rows, panels = analyze()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "metrics.csv", rows)
    plot_single(panels[("100", "wsdcon_3.csv")])
    for scale in iem.SCALES:
        plot_core_grid(scale, panels)
    write_report(rows)
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
