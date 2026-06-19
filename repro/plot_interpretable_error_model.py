#!/usr/bin/env python3
"""Plot residual-error curves for the current interpretable response models."""
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

OUT_DIR = iem.ROOT / "results" / "interpretable_error_model" / "error_comparison"
FIG_DIR = OUT_DIR / "figs"

PLOT_TARGETS = [
    ("core_wsd", "wsd_20000_24000.csv", "WSD sharp"),
    ("core_wsd", "wsdld_20000_24000.csv", "WSD linear"),
    ("core_wsd", "wsdcon_3.csv", "WSD-con 3e-5"),
    ("core_wsd", "wsdcon_9.csv", "WSD-con 9e-5"),
    ("core_wsd", "wsdcon_18.csv", "WSD-con 18e-5"),
    ("extra_control", "cosine_24000.csv", "Cosine 24k"),
]
METRIC_TARGETS = PLOT_TARGETS + [
    ("extra_control", "constant_24000.csv", "Constant 24k"),
    ("extra_control", "constant_72000.csv", "Constant 72k"),
]

METHODS = [
    {
        "key": "observation_bracket",
        "label": "Obs-bracket MPL-LD",
        "family": "observation_bracket",
        "response_rule": "observation_bracket",
        "nuisance": "mpl_ld4",
        "shrinkage": "sample_size_ridge",
        "locality": "linear",
        "color": "#2563eb",
        "linestyle": "-",
    },
    {
        "key": "old_mpl_ld",
        "label": "Old MPL-LD",
        "family": "old",
        "response_rule": "two_point_five_roundfast20",
        "nuisance": "mpl_ld4",
        "shrinkage": "ridge_tau_0p05",
        "locality": "linear",
        "color": "#059669",
        "linestyle": "--",
    },
    {
        "key": "dct_performance",
        "label": "DCT performance",
        "family": "old",
        "response_rule": "two_point_five_roundfast20",
        "nuisance": "dct_soft",
        "shrinkage": "ridge_tau_0p05",
        "locality": "linear",
        "color": "#dc2626",
        "linestyle": "-.",
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


def load_pack(scale: str, curve_name: str, cache: dict[tuple[str, str], iem.CurvePack]) -> iem.CurvePack:
    key = (scale, curve_name)
    if key not in cache:
        cache[key] = noa.load_pack(scale, curve_name)
    return cache[key]


def predict_method(
    source: iem.CurvePack,
    target: iem.CurvePack,
    method: dict[str, object],
    basis_cache: dict[tuple[str, str], np.ndarray],
    ob_basis_cache: dict[tuple[str, str, int], np.ndarray],
    selected_fit_start: int,
) -> tuple[np.ndarray, dict[str, float]]:
    if method.get("family") == "observation_bracket":
        lam = oba.response_lambda(target.curve, str(method["response_rule"]))
        coef, info = oba.fit_coefficient(
            source,
            lam,
            str(method["nuisance"]),
            str(method["shrinkage"]),
            selected_fit_start,
            ob_basis_cache,
        )
    else:
        lam = noa.response_lambda(target.curve, str(method["response_rule"]))
        coef, info = noa.fit_coefficient(
            source,
            lam,
            str(method["nuisance"]),
            str(method["shrinkage"]),
            basis_cache,
        )
    factor = iem.drop_localization_factor(target.curve) if method["locality"] == "linear" else 1.0
    feature = iem.causal_drop_response(target.curve, lam)
    pred = target.baseline + factor * coef * feature
    return pred, {
        "lambda": float(lam),
        "coef": float(coef),
        "locality_factor": float(factor),
        **{key: float(value) for key, value in info.items()},
    }


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def analyze() -> tuple[list[dict[str, object]], dict[tuple[str, str], dict[str, object]]]:
    cache: dict[tuple[str, str], iem.CurvePack] = {}
    basis_cache: dict[tuple[str, str], np.ndarray] = {}
    ob_basis_cache: dict[tuple[str, str, int], np.ndarray] = {}
    selected_fit_start = oba.select_fit_start(oba.fit_start_rule_rows(cache, ob_basis_cache))
    rows: list[dict[str, object]] = []
    panels: dict[tuple[str, str], dict[str, object]] = {}

    for scale in iem.SCALES:
        source = load_pack(scale, iem.TRAIN_CURVE, cache)
        for group, curve_name, label in METRIC_TARGETS:
            target = load_pack(scale, curve_name, cache)
            base_pred = target.baseline
            base_mae = target.base_mae
            row: dict[str, object] = {
                "scale": scale,
                "group": group,
                "test_curve": curve_name,
                "test_label": label,
                "mpl_mae": base_mae,
            }
            panel = {
                "scale": scale,
                "group": group,
                "curve_name": curve_name,
                "label": label,
                "curve": target.curve,
                "mpl_error": target.curve.loss - base_pred,
                "mpl_mae": base_mae,
                "methods": {},
            }
            for method in METHODS:
                pred, info = predict_method(source, target, method, basis_cache, ob_basis_cache, selected_fit_start)
                err = target.curve.loss - pred
                method_mae = mae(target.curve.loss, pred)
                key = str(method["key"])
                row[f"{key}_mae"] = method_mae
                row[f"{key}_delta_pct"] = 100.0 * (method_mae / base_mae - 1.0)
                row[f"{key}_signed_mean_error"] = float(np.mean(err))
                row[f"{key}_max_abs_error"] = float(np.max(np.abs(err)))
                row[f"{key}_coef"] = info["coef"]
                row[f"{key}_lambda"] = info["lambda"]
                row[f"{key}_locality_factor"] = info["locality_factor"]
                panel["methods"][key] = {
                    "error": err,
                    "mae": method_mae,
                    "delta_pct": row[f"{key}_delta_pct"],
                    "info": info,
                    "label": method["label"],
                    "color": method["color"],
                    "linestyle": method["linestyle"],
                }
            rows.append(row)
            if curve_name in {name for _, name, _ in PLOT_TARGETS}:
                panels[(scale, curve_name)] = panel
    return rows, panels


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for group in ["core_wsd", "extra_control", "all"]:
        sub = rows if group == "all" else [row for row in rows if row["group"] == group]
        if not sub:
            continue
        base: dict[str, object] = {"group": group, "rows": len(sub)}
        for method in METHODS:
            key = str(method["key"])
            deltas = np.array([float(row[f"{key}_delta_pct"]) for row in sub], dtype=np.float64)
            base[f"{key}_mean_delta"] = float(np.mean(deltas))
            base[f"{key}_median_delta"] = float(np.median(deltas))
            base[f"{key}_worst_delta"] = float(np.max(deltas))
            base[f"{key}_wins"] = int(np.sum(deltas < 0.0))
            base[f"{key}_nonharm"] = int(np.sum(deltas <= 1e-12))
        out.append(base)
    return out


def target_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for _, curve_name, label in METRIC_TARGETS:
        sub = [row for row in rows if row["test_curve"] == curve_name]
        entry: dict[str, object] = {"test_curve": curve_name, "test_label": label, "rows": len(sub)}
        for method in METHODS:
            key = str(method["key"])
            deltas = np.array([float(row[f"{key}_delta_pct"]) for row in sub], dtype=np.float64)
            entry[f"{key}_mean_delta"] = float(np.mean(deltas))
            entry[f"{key}_worst_delta"] = float(np.max(deltas))
            entry[f"{key}_wins"] = int(np.sum(deltas < 0.0))
            entry[f"{key}_nonharm"] = int(np.sum(deltas <= 1e-12))
        out.append(entry)
    return out


def plot_scale(scale: str, panels: dict[tuple[str, str], dict[str, object]], path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16.0, 7.6), constrained_layout=True)
    for idx, (_, curve_name, _label) in enumerate(PLOT_TARGETS):
        panel = panels[(scale, curve_name)]
        ax = axes.ravel()[idx]
        steps = panel["curve"].step
        errors = [panel["mpl_error"]] + [method["error"] for method in panel["methods"].values()]
        ylim = max(float(np.max(np.abs(err))) for err in errors)
        ylim = max(ylim, 1e-6)
        ax.axhline(0.0, color="#111111", lw=0.8, alpha=0.75)
        ax.plot(steps, smooth(panel["mpl_error"]), color="#111827", lw=1.35, label="MPL")
        for method in METHODS:
            data = panel["methods"][str(method["key"])]
            ax.plot(
                steps,
                smooth(data["error"]),
                color=str(method["color"]),
                lw=1.22,
                ls=str(method["linestyle"]),
                label=str(method["label"]),
            )
        ax2 = ax.twinx()
        ax2.plot(steps, panel["curve"].lrs[panel["curve"].step] / iem.PEAK_LR, color="#b45309", lw=0.8, alpha=0.22)
        ax2.set_ylim(-0.04, 1.05)
        ax2.tick_params(labelsize=7, colors="#92400e")
        title_parts = [
            f"{str(method['label']).split()[0]} {float(panel['methods'][str(method['key'])]['delta_pct']):+.1f}%"
            for method in METHODS
        ]
        ax.set_title(f"{panel['label']}\n" + " | ".join(title_parts), fontsize=9.2)
        ax.set_ylim(-1.12 * ylim, 1.12 * ylim)
        ax.grid(axis="y", alpha=0.2)
        ax.tick_params(labelsize=8)
        if idx in {0, 3}:
            ax.set_ylabel("true - prediction", fontsize=8.5)
        if idx == 0:
            ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.suptitle(f"{scale}M interpretable response error curves", fontsize=13)
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def plot_bar(summary: list[dict[str, object]], path: Path) -> None:
    rows = [row for row in target_summary_cache if row["test_curve"] not in {"constant_24000.csv", "constant_72000.csv"}]
    labels = [
        str(row["test_label"]).replace("WSD ", "WSD\n").replace("WSD-con ", "con\n").replace("Cosine ", "Cos\n")
        for row in rows
    ]
    x = np.arange(len(rows))
    width = 0.24
    fig, ax = plt.subplots(figsize=(13.0, 4.8), constrained_layout=True)
    ax.axhline(0.0, color="#111111", lw=0.9)
    offsets = np.linspace(-width, width, len(METHODS))
    for offset, method in zip(offsets, METHODS):
        key = str(method["key"])
        values = [float(row[f"{key}_mean_delta"]) for row in rows]
        ax.bar(x + offset, values, width, color=str(method["color"]), label=str(method["label"]))
    ax.set_xticks(x, labels)
    ax.set_ylabel("mean MAE change vs MPL (%)")
    ax.set_title("Interpretable response variants by target")
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def write_report(summary_rows: list[dict[str, object]], target_rows: list[dict[str, object]]) -> None:
    lines = [
        "# Interpretable Error-Comparison Figures\n\n",
        "Residual curves compare MPL, the current observation-bracket MPL-LD model, the previous fixed-tau MPL-LD reference, and the DCT performance extension.  All corrected variants fit one nonnegative coefficient from `cosine_72000.csv` only.\n\n",
        "## Aggregate\n\n",
        "| group | method | mean | worst | wins | non-harm |\n",
        "|---|---|---:|---:|---:|---:|\n",
    ]
    for row in summary_rows:
        for method in METHODS:
            key = str(method["key"])
            lines.append(
                f"| {row['group']} | {method['label']} | {float(row[f'{key}_mean_delta']):+.2f}% | "
                f"{float(row[f'{key}_worst_delta']):+.2f}% | {int(row[f'{key}_wins'])}/{int(row['rows'])} | "
                f"{int(row[f'{key}_nonharm'])}/{int(row['rows'])} |\n"
            )
    lines += [
        "\n## Per Target\n\n",
        "| target | observation-bracket mean/worst | old MPL-LD mean/worst | DCT perf mean/worst |\n",
        "|---|---:|---:|---:|\n",
    ]
    for row in target_rows:
        lines.append(
            f"| {row['test_label']} | "
            f"{float(row['observation_bracket_mean_delta']):+.2f}% / {float(row['observation_bracket_worst_delta']):+.2f}% | "
            f"{float(row['old_mpl_ld_mean_delta']):+.2f}% / {float(row['old_mpl_ld_worst_delta']):+.2f}% | "
            f"{float(row['dct_performance_mean_delta']):+.2f}% / {float(row['dct_performance_worst_delta']):+.2f}% |\n"
        )
    lines += [
        "\n## Figures\n\n",
    ]
    for scale in iem.SCALES:
        lines.append(f"- `{scale}M`: `figs/error_curves_{scale}M.png`\n")
    lines.append("- Target MAE summary: `figs/target_mae_summary.png`\n")
    lines += [
        "\n## Reading\n\n",
        "- The observation-bracket MPL-LD curve is the current mechanism-facing model: it removes the old fixed ridge and response-rate endpoints while improving every WSD-family row.\n",
        "- The old MPL-LD curve is retained as a reference to show that the newer observation-bracket rule improves the mechanism-native baseline.\n",
        "- The DCT performance extension is still a useful numerical reference, but its generic low-frequency nuisance basis should not be presented as the core explanation.\n",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


target_summary_cache: list[dict[str, object]] = []


def main() -> None:
    global target_summary_cache
    rows, panels = analyze()
    summary_rows = aggregate(rows)
    target_summary_cache = target_summary(rows)
    write_csv(OUT_DIR / "error_metrics.csv", rows)
    write_csv(OUT_DIR / "aggregate_metrics.csv", summary_rows)
    write_csv(OUT_DIR / "target_summary.csv", target_summary_cache)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for scale in iem.SCALES:
        plot_scale(scale, panels, FIG_DIR / f"error_curves_{scale}M.png")
    plot_bar(summary_rows, FIG_DIR / "target_mae_summary.png")
    write_report(summary_rows, target_summary_cache)
    print(f"wrote {OUT_DIR / 'error_metrics.csv'}")
    print(f"wrote {OUT_DIR / 'aggregate_metrics.csv'}")
    print(f"wrote {OUT_DIR / 'target_summary.csv'}")
    print(f"wrote {OUT_DIR / 'REPORT.md'}")
    print(f"wrote {FIG_DIR}")


if __name__ == "__main__":
    main()
