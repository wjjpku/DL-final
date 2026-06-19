#!/usr/bin/env python3
"""Raw overlay of our residual estimate and MPL's final G-term."""
from __future__ import annotations

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
import interpretable_observation_bracket_audit as oba  # noqa: E402
import interpretable_theory_refinement_audit as tra  # noqa: E402

OUT_DIR = iem.ROOT / "results" / "mpl_g_term_raw_overlay"
FIG_DIR = OUT_DIR / "figs"
FIT_START = 8000

CORE_TARGETS = [
    ("wsd_20000_24000.csv", "WSD sharp"),
    ("wsdld_20000_24000.csv", "WSD linear"),
    ("wsdcon_3.csv", "WSD-con 3e-5"),
    ("wsdcon_9.csv", "WSD-con 9e-5"),
    ("wsdcon_18.csv", "WSD-con 18e-5"),
]


def load_pack(scale: str, curve_name: str, cache: dict[tuple[str, str], iem.CurvePack]) -> iem.CurvePack:
    key = (scale, curve_name)
    if key not in cache:
        cache[key] = tra.load_pack(scale, curve_name, cache)
    return cache[key]


def mpl_final_g_term(curve: iem.Curve) -> np.ndarray:
    params = np.array(iem.MPL_PRECOMPUTED_INIT[curve.scale], dtype=np.float64)
    _l0, _a, _alpha, b_value, c_value, beta, gamma = params
    return b_value * iem.compute_ld(curve, c_value, beta, gamma)


def our_error_estimate(
    source: iem.CurvePack,
    target: iem.CurvePack,
    basis_cache: dict[tuple[str, str, int], np.ndarray],
) -> tuple[np.ndarray, float, float]:
    lam = tra.response_lambda(target.curve, "q2", "halflife")
    coef, _info = oba.fit_coefficient(source, lam, "mpl_ld4", "sample_size_ridge", FIT_START, basis_cache)
    factor = tra.locality_factor(target.curve, "support_projection")
    return factor * coef * iem.causal_drop_response(target.curve, lam), coef, lam


def analyze() -> dict[tuple[str, str], dict[str, object]]:
    cache: dict[tuple[str, str], iem.CurvePack] = {}
    basis_cache: dict[tuple[str, str, int], np.ndarray] = {}
    panels: dict[tuple[str, str], dict[str, object]] = {}
    for scale in iem.SCALES:
        source = load_pack(scale, iem.TRAIN_CURVE, cache)
        for curve_name, label in CORE_TARGETS:
            target = load_pack(scale, curve_name, cache)
            ours, coef, lam = our_error_estimate(source, target, basis_cache)
            l_g = mpl_final_g_term(target.curve)
            panels[(scale, curve_name)] = {
                "scale": scale,
                "curve_name": curve_name,
                "label": label,
                "step": target.curve.step,
                "ours": ours,
                "l_g": l_g,
                "coef": coef,
                "lambda": lam,
            }
    return panels


def y_limits(ours: np.ndarray, l_g: np.ndarray) -> tuple[float, float]:
    ymin = float(min(np.min(ours), np.min(l_g), 0.0))
    ymax = float(max(np.max(ours), np.max(l_g), 0.0))
    pad = 0.08 * max(ymax - ymin, 1e-8)
    return ymin - pad, ymax + pad


def plot_single(panel: dict[str, object], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 4.8), constrained_layout=True)
    step = panel["step"]
    ours = panel["ours"]
    l_g = panel["l_g"]
    ax.axhline(0.0, color="#111827", lw=0.8, alpha=0.7)
    ax.plot(step, ours, color="#2563eb", lw=1.8, label="our estimated residual")
    ax.plot(step, l_g, color="#7c3aed", lw=1.8, label=r"MPL final term $L_G(t)=B D_{LD}(t)$")
    ax.set_xlabel("step")
    ax.set_ylabel("value")
    ax.set_ylim(*y_limits(ours, l_g))
    ax.set_title(f"{panel['label']} ({panel['scale']}M): raw overlay")
    ax.legend(loc="best")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_scale(scale: str, panels: dict[tuple[str, str], dict[str, object]], path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16.0, 7.4), constrained_layout=True)
    for idx, (curve_name, _label) in enumerate(CORE_TARGETS):
        panel = panels[(scale, curve_name)]
        ax = axes.ravel()[idx]
        step = panel["step"]
        ours = panel["ours"]
        l_g = panel["l_g"]
        ax.axhline(0.0, color="#111827", lw=0.8, alpha=0.7)
        ax.plot(step, ours, color="#2563eb", lw=1.35, label="our estimated residual")
        ax.plot(step, l_g, color="#7c3aed", lw=1.35, label=r"MPL final term $L_G(t)$")
        ax.set_ylim(*y_limits(ours, l_g))
        ax.set_title(f"{panel['label']} ({scale}M)", fontsize=10)
        ax.set_xlabel("step", fontsize=8)
        if idx % 3 == 0:
            ax.set_ylabel("value", fontsize=8)
        ax.tick_params(labelsize=8)
        if idx == 0:
            ax.legend(fontsize=7, loc="best")
    axes.ravel()[-1].axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report() -> None:
    lines = [
        "# Raw MPL Final Term Overlay\n\n",
        "这组图只画两条原始曲线，不做平移、不做缩放、不取差分：\n\n",
        "- x-axis: `step`\n",
        "- y-axis: raw value\n",
        "- blue: our estimated residual\n",
        "- purple: MPL final term \\(L_G(t)=B D_{LD}(t)\\)\n\n",
        "## Figures\n\n",
        "- `figs/raw_overlay_single_100M_wsdcon_3.png`\n",
        "- `figs/raw_overlay_core_25M.png`\n",
        "- `figs/raw_overlay_core_100M.png`\n",
        "- `figs/raw_overlay_core_400M.png`\n",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    panels = analyze()
    plot_single(panels[("100", "wsdcon_3.csv")], FIG_DIR / "raw_overlay_single_100M_wsdcon_3.png")
    for scale in iem.SCALES:
        plot_scale(scale, panels, FIG_DIR / f"raw_overlay_core_{scale}M.png")
    write_report()
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
