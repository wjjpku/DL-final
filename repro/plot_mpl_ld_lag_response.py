#!/usr/bin/env python3
"""Plot error curves for the MPL-LD finite-response candidates."""
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
import mpl_ld_lag_response_audit as lag  # noqa: E402

OUT_DIR = iem.ROOT / "results" / "mpl_ld_lag_response_audit"
FIG_DIR = OUT_DIR / "figs"

PLOT_TARGETS = [
    ("core_wsd", "wsd_20000_24000.csv", "WSD sharp"),
    ("core_wsd", "wsdld_20000_24000.csv", "WSD linear"),
    ("core_wsd", "wsdcon_3.csv", "WSD-con 3e-5"),
    ("core_wsd", "wsdcon_9.csv", "WSD-con 9e-5"),
    ("core_wsd", "wsdcon_18.csv", "WSD-con 18e-5"),
    ("extra_control", "cosine_24000.csv", "Cosine 24k control"),
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


def predict_fixed128(pack: iem.CurvePack) -> tuple[np.ndarray, dict[str, float]]:
    tau = 128.0
    factor = iem.drop_localization_factor(pack.curve)
    feature = lag.lag_feature(pack, tau, "cooldown")
    return pack.baseline + factor * feature, {
        "effective_tau_steps": tau,
        "attenuation_factor": factor,
        "cooldown_support_span": float(lag.cooldown_support_span(pack.curve)),
    }


def predict_support_bracket(pack: iem.CurvePack) -> tuple[np.ndarray, dict[str, float]]:
    tau = lag.response_tau_steps(pack.curve, "support_bracket", None)
    factor = iem.drop_localization_factor(pack.curve)
    feature = lag.lag_feature(pack, tau, "cooldown")
    return pack.baseline + factor * feature, {
        "effective_tau_steps": tau,
        "attenuation_factor": factor,
        "cooldown_support_span": float(lag.cooldown_support_span(pack.curve)),
    }


METHODS = [
    {
        "key": "cooldown_fixed128",
        "label": "Cooldown tau=128",
        "predict": predict_fixed128,
        "color": "#2563eb",
        "linestyle": "--",
    },
    {
        "key": "cooldown_support_bracket",
        "label": "Cooldown support tau",
        "predict": predict_support_bracket,
        "color": "#dc2626",
        "linestyle": "-",
    },
]


def analyze() -> tuple[list[dict[str, object]], dict[tuple[str, str], dict[str, object]]]:
    cache = lag.load_all_packs()
    rows: list[dict[str, object]] = []
    panels: dict[tuple[str, str], dict[str, object]] = {}
    for scale in iem.SCALES:
        for group, curve_name, label in PLOT_TARGETS:
            pack = cache[(scale, curve_name)]
            base_mae = pack.base_mae
            row: dict[str, object] = {
                "scale": scale,
                "group": group,
                "test_curve": curve_name,
                "test_label": label,
                "mpl_mae": base_mae,
            }
            panel: dict[str, object] = {
                "scale": scale,
                "group": group,
                "curve": pack.curve,
                "label": label,
                "mpl_error": pack.curve.loss - pack.baseline,
                "mpl_mae": base_mae,
                "methods": {},
            }
            for method in METHODS:
                pred, info = method["predict"](pack)
                err = pack.curve.loss - pred
                method_mae = mae(pack.curve.loss, pred)
                key = str(method["key"])
                row[f"{key}_mae"] = method_mae
                row[f"{key}_delta_pct"] = 100.0 * (method_mae / base_mae - 1.0)
                row[f"{key}_effective_tau_steps"] = info["effective_tau_steps"]
                row[f"{key}_attenuation_factor"] = info["attenuation_factor"]
                row[f"{key}_cooldown_support_span"] = info["cooldown_support_span"]
                panel["methods"][key] = {
                    "error": err,
                    "mae": method_mae,
                    "delta_pct": row[f"{key}_delta_pct"],
                    "label": method["label"],
                    "color": method["color"],
                    "linestyle": method["linestyle"],
                }
            rows.append(row)
            panels[(scale, curve_name)] = panel
    return rows, panels


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for group in ["core_wsd", "extra_control", "all"]:
        sub = rows if group == "all" else [row for row in rows if row["group"] == group]
        if not sub:
            continue
        entry: dict[str, object] = {"group": group, "rows": len(sub)}
        for method in METHODS:
            key = str(method["key"])
            deltas = np.array([float(row[f"{key}_delta_pct"]) for row in sub], dtype=np.float64)
            entry[f"{key}_mean_delta"] = float(np.mean(deltas))
            entry[f"{key}_worst_delta"] = float(np.max(deltas))
            entry[f"{key}_wins"] = int(np.sum(deltas < 0.0))
            entry[f"{key}_nonharm"] = int(np.sum(deltas <= 1e-12))
        out.append(entry)
    return out


def plot_scale(scale: str, panels: dict[tuple[str, str], dict[str, object]]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16.0, 7.6), constrained_layout=True)
    for idx, (_group, curve_name, _label) in enumerate(PLOT_TARGETS):
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
                color=data["color"],
                lw=1.35,
                linestyle=data["linestyle"],
                label=f"{data['label']} ({float(data['delta_pct']):+.1f}%)",
            )
        ax.set_title(f"{panel['label']} ({scale}M)", fontsize=10)
        ax.set_xlabel("step", fontsize=8)
        ax.set_ylabel("loss error", fontsize=8)
        ax.set_ylim(-1.12 * ylim, 1.12 * ylim)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7, loc="best")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"finite_response_errors_{scale}M.png", dpi=180)
    plt.close(fig)


def write_report(aggregate_rows: list[dict[str, object]]) -> None:
    core = next(row for row in aggregate_rows if row["group"] == "core_wsd")
    ctrl = next(row for row in aggregate_rows if row["group"] == "extra_control")
    lines = [
        "# MPL-LD Finite-Response Error Curves\n\n",
        "These plots compare MPL against the current cooldown finite-response candidates.  "
        "The support-bracket row uses no residual-fitted coefficient; its effective tau is derived from the logging interval and cooldown support span.\n\n",
        "## Aggregate\n\n",
        "| method | WSD mean | WSD worst | WSD wins | controls worst | controls nonharm |\n",
        "|---|---:|---:|---:|---:|---:|\n",
    ]
    for method in METHODS:
        key = str(method["key"])
        lines.append(
            f"| {method['label']} | {float(core[f'{key}_mean_delta']):+.2f}% | "
            f"{float(core[f'{key}_worst_delta']):+.2f}% | "
            f"{int(core[f'{key}_wins'])}/{int(core['rows'])} | "
            f"{float(ctrl[f'{key}_worst_delta']):+.2f}% | "
            f"{int(ctrl[f'{key}_nonharm'])}/{int(ctrl['rows'])} |\n"
        )
    lines += [
        "\n## Figures\n\n",
        "- `figs/finite_response_errors_25M.png`\n",
        "- `figs/finite_response_errors_100M.png`\n",
        "- `figs/finite_response_errors_400M.png`\n",
    ]
    (OUT_DIR / "ERROR_CURVES.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    rows, panels = analyze()
    agg = aggregate(rows)
    write_csv(OUT_DIR / "error_curve_metrics.csv", rows)
    write_csv(OUT_DIR / "error_curve_aggregate.csv", agg)
    for scale in iem.SCALES:
        plot_scale(scale, panels)
    write_report(agg)


if __name__ == "__main__":
    main()
