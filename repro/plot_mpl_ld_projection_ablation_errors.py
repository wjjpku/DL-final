#!/usr/bin/env python3
"""Plot with/without MPL-LD tangent projection residual predictions."""
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

OUT_DIR = iem.ROOT / "results" / "mpl_ld_projection_ablation"
FIG_DIR = OUT_DIR / "figs"
FIT_START = tra.FIT_START

PLOT_TARGETS = [
    ("core_wsd", "wsd_20000_24000.csv", "WSD sharp"),
    ("core_wsd", "wsdld_20000_24000.csv", "WSD linear"),
    ("core_wsd", "wsdcon_3.csv", "WSD-con 3e-5"),
    ("core_wsd", "wsdcon_9.csv", "WSD-con 9e-5"),
    ("core_wsd", "wsdcon_18.csv", "WSD-con 18e-5"),
]

METHODS = [
    {
        "key": "with_projection",
        "label": "with MPL-LD projection",
        "projection": "mpl_ld4",
        "color": "#2563eb",
        "linestyle": "-",
    },
    {
        "key": "without_projection",
        "label": "without projection",
        "projection": "none",
        "color": "#dc2626",
        "linestyle": "--",
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


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def load_pack(scale: str, curve_name: str, cache: dict[tuple[str, str], iem.CurvePack]) -> iem.CurvePack:
    key = (scale, curve_name)
    if key not in cache:
        cache[key] = tra.load_pack(scale, curve_name, cache)
    return cache[key]


def predict_correction(
    source: iem.CurvePack,
    target: iem.CurvePack,
    projection: str,
    basis_cache: dict[tuple[str, str, int], np.ndarray],
) -> tuple[np.ndarray, dict[str, float]]:
    lam = tra.response_lambda(target.curve, "q2", "halflife")
    coef, info = oba.fit_coefficient(
        source,
        lam,
        projection,
        "sample_size_ridge",
        FIT_START,
        basis_cache,
    )
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
        for group, curve_name, label in PLOT_TARGETS:
            target = load_pack(scale, curve_name, cache)
            residual = target.curve.loss - target.baseline
            panel: dict[str, object] = {
                "scale": scale,
                "group": group,
                "curve_name": curve_name,
                "label": label,
                "curve": target.curve,
                "residual": residual,
                "base_mae": target.base_mae,
                "methods": {},
            }
            row: dict[str, object] = {
                "scale": scale,
                "group": group,
                "test_curve": curve_name,
                "test_label": label,
                "mpl_mae": target.base_mae,
                "residual_l1": float(np.mean(np.abs(residual))),
                "residual_max_abs": float(np.max(np.abs(residual))),
            }
            for method in METHODS:
                correction, info = predict_correction(source, target, str(method["projection"]), basis_cache)
                remaining = residual - correction
                pred = target.baseline + correction
                method_mae = mae(target.curve.loss, pred)
                delta_pct = 100.0 * (method_mae / target.base_mae - 1.0)
                key = str(method["key"])
                row[f"{key}_mae"] = method_mae
                row[f"{key}_delta_pct"] = delta_pct
                row[f"{key}_coef"] = info["coef"]
                row[f"{key}_lambda"] = info["lambda"]
                row[f"{key}_locality_factor"] = info["locality_factor"]
                row[f"{key}_source_retention"] = info["source_retention"]
                row[f"{key}_correction_l1"] = float(np.mean(np.abs(correction)))
                row[f"{key}_correction_max_abs"] = float(np.max(np.abs(correction)))
                row[f"{key}_correction_to_residual_l1"] = float(
                    np.mean(np.abs(correction)) / max(np.mean(np.abs(residual)), 1e-18)
                )
                panel["methods"][key] = {
                    "label": method["label"],
                    "color": method["color"],
                    "linestyle": method["linestyle"],
                    "correction": correction,
                    "remaining": remaining,
                    "mae": method_mae,
                    "delta_pct": delta_pct,
                    "info": info,
                }
            rows.append(row)
            panels[(scale, curve_name)] = panel
    return rows, panels


def panel_ylim(arrays: list[np.ndarray], pad: float = 1.12) -> tuple[float, float]:
    limit = max(float(np.max(np.abs(arr))) for arr in arrays)
    limit = max(limit, 1e-7)
    return -pad * limit, pad * limit


def plot_prediction_scale(
    scale: str,
    panels: dict[tuple[str, str], dict[str, object]],
    path: Path,
    zoom: bool,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16.0, 7.8), constrained_layout=True)
    for idx, (_group, curve_name, _label) in enumerate(PLOT_TARGETS):
        panel = panels[(scale, curve_name)]
        ax = axes.ravel()[idx]
        curve = panel["curve"]
        steps = curve.step
        residual = panel["residual"]
        with_corr = panel["methods"]["with_projection"]["correction"]
        without_corr = panel["methods"]["without_projection"]["correction"]

        ax.axhline(0.0, color="#111827", lw=0.8, alpha=0.72)
        ax.plot(steps, smooth(residual), color="#111827", lw=1.35, label="true MPL residual")
        for method in METHODS:
            item = panel["methods"][str(method["key"])]
            ax.plot(
                steps,
                smooth(item["correction"]),
                color=item["color"],
                lw=1.25,
                linestyle=item["linestyle"],
                label=f"{item['label']} ({float(item['delta_pct']):+.1f}%)",
            )

        if zoom:
            ax.set_ylim(*panel_ylim([residual, with_corr], pad=1.35))
        else:
            ax.set_ylim(*panel_ylim([residual, with_corr, without_corr]))

        ax2 = ax.twinx()
        ax2.plot(steps, curve.lrs[curve.step] / iem.PEAK_LR, color="#b45309", lw=0.85, alpha=0.24)
        ax2.set_ylim(-0.05, 1.05)
        ax2.set_yticks([])

        suffix = "zoom" if zoom else "full"
        ax.set_title(f"{panel['label']} ({scale}M, {suffix})", fontsize=10)
        ax.set_xlabel("step", fontsize=8)
        if idx % 3 == 0:
            ax.set_ylabel("predicted error / residual", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=6.8, loc="best")
    axes.ravel()[-1].axis("off")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_remaining_scale(
    scale: str,
    panels: dict[tuple[str, str], dict[str, object]],
    path: Path,
    zoom: bool,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16.0, 7.8), constrained_layout=True)
    for idx, (_group, curve_name, _label) in enumerate(PLOT_TARGETS):
        panel = panels[(scale, curve_name)]
        ax = axes.ravel()[idx]
        curve = panel["curve"]
        steps = curve.step
        residual = panel["residual"]
        with_remaining = panel["methods"]["with_projection"]["remaining"]
        without_remaining = panel["methods"]["without_projection"]["remaining"]

        ax.axhline(0.0, color="#111827", lw=0.8, alpha=0.72)
        ax.plot(steps, smooth(residual), color="#111827", lw=1.35, label="MPL residual")
        for method in METHODS:
            item = panel["methods"][str(method["key"])]
            ax.plot(
                steps,
                smooth(item["remaining"]),
                color=item["color"],
                lw=1.25,
                linestyle=method["linestyle"],
                label=f"remaining {item['label']} ({float(item['delta_pct']):+.1f}%)",
            )

        if zoom:
            ax.set_ylim(*panel_ylim([residual, with_remaining], pad=1.35))
        else:
            ax.set_ylim(*panel_ylim([residual, with_remaining, without_remaining]))

        ax2 = ax.twinx()
        ax2.plot(steps, curve.lrs[curve.step] / iem.PEAK_LR, color="#b45309", lw=0.85, alpha=0.24)
        ax2.set_ylim(-0.05, 1.05)
        ax2.set_yticks([])

        suffix = "zoom" if zoom else "full"
        ax.set_title(f"{panel['label']} ({scale}M, {suffix})", fontsize=10)
        ax.set_xlabel("step", fontsize=8)
        if idx % 3 == 0:
            ax.set_ylabel("remaining residual", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=6.8, loc="best")
    axes.ravel()[-1].axis("off")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(rows: list[dict[str, object]]) -> None:
    lines = [
        "# Projection Ablation Error Curves\n\n",
        "这些图比较同一个 q2 half-life 公式在有无 MPL-LD tangent projection 时预测出的 residual correction。",
        "黑线是真实 MPL residual，即 \\(L-L_{MPL}\\)；蓝线是有投影的预测误差项；红虚线是无投影的预测误差项。",
        "每个面板里的百分比是加入该 correction 后相对 MPL MAE 的变化。\n\n",
        "## Figures\n\n",
    ]
    for scale in iem.SCALES:
        lines += [
            f"- `figs/predicted_residual_full_{scale}M.png`: 完整纵轴，能看到无投影过冲幅度。\n",
            f"- `figs/predicted_residual_zoom_{scale}M.png`: 以真实 residual 和有投影结果为主的放大图。\n",
            f"- `figs/remaining_residual_full_{scale}M.png`: 校正后剩余 residual，完整纵轴。\n",
            f"- `figs/remaining_residual_zoom_{scale}M.png`: 校正后剩余 residual，放大图。\n",
        ]
    lines += [
        "\n## Same-Scale Metrics\n\n",
        "| scale | target | with projection | without projection | no-proj / true residual L1 |\n",
        "| --- | --- | ---: | ---: | ---: |\n",
    ]
    for row in rows:
        lines.append(
            f"| {row['scale']}M | {row['test_label']} | "
            f"{float(row['with_projection_delta_pct']):+.2f}% | "
            f"{float(row['without_projection_delta_pct']):+.2f}% | "
            f"{float(row['without_projection_correction_to_residual_l1']):.2f}x |\n"
        )
    lines += [
        "\n## Reading\n\n",
        "无投影时，红线通常不是跟真实 residual 同量级的局部响应，而是把一个过大的、平滑的 cosine 残差方向迁移到了 WSD。",
        "这就是 summary 里 WSD 从全胜变成全败的直接视觉原因。\n",
    ]
    (OUT_DIR / "ERROR_CURVES.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    rows, panels = analyze()
    write_csv(OUT_DIR / "error_curve_metrics.csv", rows)
    for scale in iem.SCALES:
        plot_prediction_scale(scale, panels, FIG_DIR / f"predicted_residual_full_{scale}M.png", zoom=False)
        plot_prediction_scale(scale, panels, FIG_DIR / f"predicted_residual_zoom_{scale}M.png", zoom=True)
        plot_remaining_scale(scale, panels, FIG_DIR / f"remaining_residual_full_{scale}M.png", zoom=False)
        plot_remaining_scale(scale, panels, FIG_DIR / f"remaining_residual_zoom_{scale}M.png", zoom=True)
    write_report(rows)
    print(f"wrote {OUT_DIR / 'ERROR_CURVES.md'}")


if __name__ == "__main__":
    main()
