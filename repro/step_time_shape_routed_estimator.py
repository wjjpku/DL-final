#!/usr/bin/env python3
"""Shape-routed step-time estimator.

The decomposed estimator separates low-frequency nuisance from finite LR-drop
lag.  This script adds a deployment-style generalization head: the target LR
schedule shape chooses the source calibration curve(s), the response time, and
the nuisance projection.  The route uses only LR schedules, never the target
loss residual.
"""
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
ROOT = REPO.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from reproduce_cosine_to_wsd import (  # noqa: E402
    MPL_PRECOMPUTED_INIT,
    PEAK_LR,
    SCALES,
    load_curve,
    metrics,
    mpl_predict,
)
from step_time_decomposed_estimator import (  # noqa: E402
    CURVES,
    NUISANCE_SHRINK,
    StepTimeNuisanceEstimator,
    build_cache as build_decomposed_cache,
    dct_basis,
    key_metrics,
    lowfreq_basis,
    residualize,
    score_long_probe_to_wsd,
    score_single,
    summarize,
)


OUT_DIR = ROOT / "results" / "step_time_shape_routed_estimator"
FIG_DIR = OUT_DIR / "figs"

CORE_CURVES = CURVES
EXTRA_SAFETY_CURVES = [
    ("cosine_24000.csv", "Cosine 24k"),
    ("constant_24000.csv", "Constant 24k"),
    ("constant_72000.csv", "Constant 72k"),
]
EXTENDED_CURVES = CORE_CURVES + EXTRA_SAFETY_CURVES


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_cache(
    curve_defs: list[tuple[str, str]] | tuple[tuple[str, str], ...] = CORE_CURVES,
) -> dict[tuple[str, str], dict[str, object]]:
    cache: dict[tuple[str, str], dict[str, object]] = {}
    for scale in SCALES:
        for curve_name, label in curve_defs:
            curve = load_curve(scale, curve_name)
            base = mpl_predict(MPL_PRECOMPUTED_INIT[scale], curve)
            residual = curve.loss - base
            cache[(scale, curve_name)] = {
                "curve": curve,
                "label": label,
                "base": base,
                "residual": residual,
                "base_mae": metrics(curve.loss, base)["mae"],
            }
    return cache


def positive_drop(curve) -> np.ndarray:
    eta = curve.lrs.astype(np.float64)
    drop = np.zeros_like(eta)
    drop[1:] = np.maximum(eta[:-1] - eta[1:], 0.0) / PEAK_LR
    return drop


def schedule_stats(
    cache: dict[tuple[str, str], dict[str, object]],
    curve_defs: list[tuple[str, str]] | tuple[tuple[str, str], ...] = CORE_CURVES,
) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for curve_name, label in curve_defs:
        curve = cache[(SCALES[0], curve_name)]["curve"]
        drop = positive_drop(curve)
        idx = np.flatnonzero(drop > 1e-14)
        total_drop = float(np.sum(drop))
        span = float(idx[-1] - idx[0] + 1) if len(idx) else 0.0
        entropy = 0.0
        if total_drop > 0.0 and len(idx):
            p = drop[idx] / total_drop
            entropy = float(-np.sum(p * np.log(np.clip(p, 1e-300, None))))
        stats[curve_name] = {
            "label": label,
            "drop_norm": total_drop,
            "decay_span": span,
            "drop_entropy": entropy,
            "final_lr": float(curve.lrs[-1]),
            "schedule_len": float(len(curve.lrs)),
        }
    return stats


def response_feature(curve, tau: float) -> np.ndarray:
    drop = positive_drop(curve)
    out = np.zeros_like(drop)
    acc = 0.0
    decay = math.exp(-1.0 / tau)
    for t in range(len(drop)):
        acc = acc * decay + drop[t]
        out[t] = acc
    return out[curve.step]


def basis_for(curve, kind: str) -> np.ndarray:
    if kind == "none":
        return np.zeros((len(curve.step), 0), dtype=np.float64)
    if kind.startswith("dct"):
        return dct_basis(curve.step, int(kind[3:]))
    if kind.startswith("fourier"):
        return lowfreq_basis(curve.step, int(kind[7:]), normalize=True)
    raise ValueError(kind)


def fit_kappa(
    cache: dict[tuple[str, str], dict[str, object]],
    scale: str,
    train_curves: tuple[str, ...],
    tau: float,
    nuisance: str,
) -> float:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for curve_name in train_curves:
        row = cache[(scale, curve_name)]
        curve = row["curve"]
        phi = response_feature(curve, tau)
        residual = row["residual"]
        basis = basis_for(curve, nuisance)
        if basis.shape[1] > 0:
            phi = residualize(phi, basis)
            residual = residualize(residual, basis)
        xs.append(phi)
        ys.append(residual)
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    denom = float(np.dot(x, x))
    if denom <= 1e-18:
        return 0.0
    return max(0.0, float(np.dot(x, y) / denom))


def strongest_step_probe(stats: dict[str, dict[str, float]], excluded: str) -> tuple[str, ...]:
    step_probes = [
        c
        for c in stats
        if c != excluded and stats[c]["decay_span"] <= 1.0 and stats[c]["drop_norm"] > 0.85
    ]
    return (max(step_probes, key=lambda c: stats[c]["drop_norm"]),)


def wsd_partners(stats: dict[str, dict[str, float]], excluded: str) -> tuple[str, ...]:
    partners = [
        c
        for c in stats
        if c != excluded and 100.0 < stats[c]["decay_span"] <= 16000.0
    ]
    partners.sort(key=lambda c: (abs(stats[c]["drop_norm"] - stats[excluded]["drop_norm"]), c))
    return tuple(partners[:2])


def route_for_target(
    stats: dict[str, dict[str, float]], target_curve: str
) -> dict[str, object]:
    target = stats[target_curve]
    span = target["decay_span"]
    drop = target["drop_norm"]

    if drop <= 0.05:
        return {
            "route": "no_lr_drop",
            "train_curves": tuple(),
            "tau": 0.0,
            "nuisance": "none",
            "rationale": "no positive LR drop, so the LR-drop transient must be zero",
        }

    if span > 16000.0 and target["schedule_len"] <= 30000.0:
        return {
            "route": "short_smooth_no_transfer",
            "train_curves": tuple(),
            "tau": 0.0,
            "nuisance": "none",
            "rationale": "short smooth cosine has no stable nonzero transient transfer across scales",
        }

    if span > 16000.0:
        return {
            "route": "smooth_decay",
            "train_curves": strongest_step_probe(stats, target_curve),
            "tau": 8192.0,
            "nuisance": "dct4",
            "rationale": "very long smooth decay uses the slowest finite response; source is the strongest single-step full-drop probe",
        }

    if span > 100.0:
        return {
            "route": "finite_tail",
            "train_curves": wsd_partners(stats, target_curve) or strongest_step_probe(stats, target_curve),
            "tau": 5120.0,
            "nuisance": "none",
            "rationale": "finite WSD tail uses its paired finite-tail schedule when available",
        }

    if drop >= 0.85:
        return {
            "route": "full_step_drop",
            "train_curves": wsd_partners(stats, target_curve),
            "tau": 1536.0,
            "nuisance": "dct2",
            "rationale": "full single-step drop borrows amplitude from finite-tail WSD curves and removes broad drift",
        }

    if drop >= 0.60:
        train = tuple(
            c
            for c in stats
            if c != target_curve
            and stats[c]["decay_span"] <= 1.0
            and stats[c]["drop_norm"] > 0.05
        )
        return {
            "route": "medium_step_drop",
            "train_curves": train,
            "tau": 768.0,
            "nuisance": "dct2",
            "rationale": "medium single-step drop uses neighboring step probes with a short response",
        }

    stronger_steps = [
        c
        for c in stats
        if c != target_curve
        and stats[c]["decay_span"] <= 1.0
        and stats[c]["drop_norm"] > drop
    ]
    stronger_steps.sort(key=lambda c: (stats[c]["drop_norm"] - drop, c))
    return {
        "route": "weak_step_drop",
        "train_curves": tuple(stronger_steps[:1]),
        "tau": 512.0,
        "nuisance": "none",
        "rationale": "weak single-step drop uses the nearest stronger step probe and the shortest stable response",
    }


def naive_route_without_short_smooth_gate(
    stats: dict[str, dict[str, float]], target_curve: str
) -> dict[str, object]:
    """Ablation route: keep zero-drop safety, but let short smooth cosine transfer."""
    target = stats[target_curve]
    span = target["decay_span"]
    drop = target["drop_norm"]
    if drop <= 0.05:
        return {
            "route": "no_lr_drop",
            "train_curves": tuple(),
            "tau": 0.0,
            "nuisance": "none",
            "rationale": "no positive LR drop, so the LR-drop transient must be zero",
        }
    if span > 16000.0:
        return {
            "route": "smooth_decay_no_short_gate",
            "train_curves": strongest_step_probe(stats, target_curve),
            "tau": 8192.0,
            "nuisance": "dct4",
            "rationale": "ablation: short smooth decay is allowed to transfer like long smooth decay",
        }
    return route_for_target(stats, target_curve)


def score_target(
    cache: dict[tuple[str, str], dict[str, object]],
    scale: str,
    target_curve: str,
    kappa: float,
    tau: float,
) -> dict[str, object]:
    row = cache[(scale, target_curve)]
    curve = row["curve"]
    if tau <= 0.0 or kappa == 0.0:
        pred = row["base"]
    else:
        pred = row["base"] + kappa * response_feature(curve, tau)
    corr_mae = metrics(curve.loss, pred)["mae"]
    base_mae = float(row["base_mae"])
    return {
        "base_mae": base_mae,
        "corr_mae": corr_mae,
        "delta_pct": 100.0 * (corr_mae / base_mae - 1.0),
        "win": int(corr_mae < base_mae),
        "nonharm": int(corr_mae <= base_mae * (1.0 + 1e-12)),
    }


def run_shape_routed(
    curve_defs: list[tuple[str, str]] | tuple[tuple[str, str], ...] = CORE_CURVES,
    mode: str = "shape_routed_target_holdout",
    route_fn=route_for_target,
    tau_multiplier: float = 1.0,
    force_tau: float | None = None,
    force_nuisance: str | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    cache = build_cache(curve_defs)
    stats = schedule_stats(cache, curve_defs)
    routes = []
    details = []
    for target_curve, target_label in curve_defs:
        route = route_fn(stats, target_curve)
        train_curves = tuple(route["train_curves"])
        raw_tau = float(route["tau"])
        tau = raw_tau
        if raw_tau > 0.0:
            tau = float(force_tau) if force_tau is not None else raw_tau * tau_multiplier
        nuisance = str(force_nuisance) if force_nuisance is not None and train_curves and tau > 0.0 else str(route["nuisance"])
        routes.append(
            {
                "target_curve": target_curve,
                "target_label": target_label,
                "target_drop_norm": stats[target_curve]["drop_norm"],
                "target_decay_span": stats[target_curve]["decay_span"],
                "target_drop_entropy": stats[target_curve]["drop_entropy"],
                "target_schedule_len": stats[target_curve]["schedule_len"],
                "route": route["route"],
                "train_curves": "+".join(c.replace(".csv", "") for c in train_curves) if train_curves else "none",
                "tau": tau,
                "base_tau": raw_tau,
                "nuisance": nuisance,
                "base_nuisance": route["nuisance"],
                "rationale": route["rationale"],
            }
        )
        for scale in SCALES:
            if train_curves and tau > 0.0:
                kappa = fit_kappa(cache, scale, train_curves, tau, nuisance)
            else:
                kappa = 0.0
            scored = score_target(cache, scale, target_curve, kappa, tau)
            details.append(
                {
                    "mode": mode,
                    "scale": scale,
                    "target_curve": target_curve,
                    "target_label": target_label,
                    "route": route["route"],
                    "train_curves": "+".join(c.replace(".csv", "") for c in train_curves) if train_curves else "none",
                    "tau": tau,
                    "base_tau": raw_tau,
                    "nuisance": nuisance,
                    "base_nuisance": route["nuisance"],
                    "kappa": kappa,
                    **scored,
                }
            )
    summary = []
    for target_curve, target_label in curve_defs:
        sub = [r for r in details if r["target_curve"] == target_curve]
        summary.append(
            {
                "target_curve": target_curve,
                "target_label": target_label,
                **summarize(sub),
            }
        )
    return routes, details, summary


def run_ablation_audits() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    configs = [
        {
            "audit": "final_core",
            "curve_defs": CORE_CURVES,
            "route_fn": route_for_target,
            "tau_multiplier": 1.0,
            "force_tau": None,
            "force_nuisance": None,
            "description": "final routed rule on core target-holdout curves",
        },
        {
            "audit": "final_extended",
            "curve_defs": EXTENDED_CURVES,
            "route_fn": route_for_target,
            "tau_multiplier": 1.0,
            "force_tau": None,
            "force_nuisance": None,
            "description": "final routed rule plus cosine/constant safety controls",
        },
        {
            "audit": "no_short_smooth_gate",
            "curve_defs": EXTENDED_CURVES,
            "route_fn": naive_route_without_short_smooth_gate,
            "tau_multiplier": 1.0,
            "force_tau": None,
            "force_nuisance": None,
            "description": "ablation that lets cosine_24000 transfer as a long smooth decay",
        },
        {
            "audit": "no_nuisance_projection",
            "curve_defs": CORE_CURVES,
            "route_fn": route_for_target,
            "tau_multiplier": 1.0,
            "force_tau": None,
            "force_nuisance": "none",
            "description": "core routes with nuisance projection disabled",
        },
        {
            "audit": "fixed_tau_1024",
            "curve_defs": CORE_CURVES,
            "route_fn": route_for_target,
            "tau_multiplier": 1.0,
            "force_tau": 1024.0,
            "force_nuisance": None,
            "description": "core routes with one universal tau=1024",
        },
    ]
    for mult in [0.5, 0.75, 1.25, 1.5, 2.0]:
        configs.append(
            {
                "audit": f"tau_x{mult:g}",
                "curve_defs": CORE_CURVES,
                "route_fn": route_for_target,
                "tau_multiplier": mult,
                "force_tau": None,
                "force_nuisance": None,
                "description": f"core routes with all nonzero taus multiplied by {mult:g}",
            }
        )

    all_details: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for config in configs:
        _, details, _ = run_shape_routed(
            config["curve_defs"],
            mode=str(config["audit"]),
            route_fn=config["route_fn"],
            tau_multiplier=float(config["tau_multiplier"]),
            force_tau=config["force_tau"],
            force_nuisance=config["force_nuisance"],
        )
        for row in details:
            row["audit"] = config["audit"]
            row["description"] = config["description"]
        all_details.extend(details)
        summary_rows.append(
            {
                "audit": config["audit"],
                "description": config["description"],
                **summarize(details),
            }
        )
    return summary_rows, all_details


def comparison_rows(shape_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    estimator = StepTimeNuisanceEstimator()
    decomposed_cache = build_decomposed_cache()
    decomposed_rows = score_single(decomposed_cache, estimator, NUISANCE_SHRINK)
    long_rows = score_long_probe_to_wsd(decomposed_cache)
    km = key_metrics(decomposed_rows)
    shape = summarize(shape_rows)
    long = summarize(long_rows)
    return [
        {
            "metric": "decomposed_self_fit",
            "mean_delta": km["self_mean_delta"],
            "worst_delta": km["self_worst_delta"],
            "nonharm": f"{km['self_nonharm']}/{km['self_rows']}",
        },
        {
            "metric": "decomposed_full_offdiag",
            "mean_delta": km["offdiag_mean_delta"],
            "worst_delta": km["offdiag_worst_delta"],
            "nonharm": f"{km['offdiag_nonharm']}/{km['offdiag_rows']}",
        },
        {
            "metric": "long_probe_to_wsd",
            "mean_delta": long["mean_delta"],
            "worst_delta": long["worst_delta"],
            "nonharm": f"{long['nonharm']}/{long['rows']}",
        },
        {
            "metric": "shape_routed_target_holdout",
            "mean_delta": shape["mean_delta"],
            "worst_delta": shape["worst_delta"],
            "nonharm": f"{shape['nonharm']}/{shape['rows']}",
        },
    ]


def plot_target_holdout(path: Path, details: list[dict[str, object]], summary: list[dict[str, object]]) -> None:
    labels = [
        str(r["target_label"])
        .replace("WSD sharp", "WSD\nsharp")
        .replace("WSD linear", "WSD\nlinear")
        .replace("WSD-con ", "con\n")
        for r in summary
    ]
    means = np.array([float(r["mean_delta"]) for r in summary])
    worst = np.array([float(r["worst_delta"]) for r in summary])
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10.8, 5.2))
    ax.axhline(0.0, color="#111111", lw=0.9)
    bars = ax.bar(x, means, color="#3b82f6", width=0.66, label="mean over scales")
    ax.scatter(x, worst, color="#dc2626", zorder=3, label="worst scale")
    ax.set_xticks(x, labels)
    ax.set_ylabel("MAE change vs MPL (%)")
    ax.set_title("Shape-routed target-holdout generalization")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    for bar in bars:
        bar.set_linewidth(0)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.18)
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def write_report(
    routes: list[dict[str, object]],
    details: list[dict[str, object]],
    summary: list[dict[str, object]],
    comparison: list[dict[str, object]],
    extended_routes: list[dict[str, object]],
    extended_details: list[dict[str, object]],
    extended_summary: list[dict[str, object]],
    ablation_summary: list[dict[str, object]],
) -> None:
    shape = summarize(details)
    extended = summarize(extended_details)
    extra_details = [
        r
        for r in extended_details
        if r["target_curve"] in {curve_name for curve_name, _ in EXTRA_SAFETY_CURVES}
    ]
    extra = summarize(extra_details)
    lines = [
        "# Shape-Routed Step-Time Estimator\n\n",
        "This is the stronger generalization head suggested by the residual figures.  The rule does not ask every calibration curve to predict every target.  Instead, the target LR schedule shape chooses a finite response time and a source calibration set.\n\n",
        "## Route Rule\n\n",
        "- Very long smooth decay: use a slow finite response (`tau=8192`) calibrated by the strongest full-drop step probe, with `dct4` nuisance projection.\n",
        "- Finite WSD tail: use the paired finite-tail WSD curve when available and `tau=5120`.\n",
        "- Full single-step drop: use finite-tail WSD curves, `tau=1536`, and `dct2` nuisance projection.\n",
        "- Medium single-step drop: use neighboring step probes, `tau=768`, and `dct2` nuisance projection.\n",
        "- Weak single-step drop: use the nearest stronger step probe and `tau=512`.\n\n",
        "- No positive LR drop or short smooth cosine: use no transient correction.  These are safety gates, not performance claims.\n\n",
        "The route uses only the LR schedule: total positive LR drop, positive-drop span, and whether the decay is smooth, finite-tail, or single-step.  Target loss residuals are not used by routing or kappa fitting.\n\n",
        "## Main Result\n\n",
        f"- Shape-routed target-holdout: mean `{float(shape['mean_delta']):+.1f}%`, worst `{float(shape['worst_delta']):+.1f}%`, non-harm `{int(shape['nonharm'])}/{int(shape['rows'])}`.\n",
        "- This is a target-holdout deployment audit: each target curve is predicted from other curve(s) selected by schedule shape.\n\n",
        "![target holdout](figs/shape_routed_target_holdout.png)\n\n",
        "## Route Table\n\n",
        "| target | drop | span | route | source | tau | nuisance |\n",
        "|---|---:|---:|---|---|---:|---|\n",
    ]
    for row in routes:
        lines.append(
            f"| {row['target_label']} | {float(row['target_drop_norm']):.3f} | "
            f"{float(row['target_decay_span']):.0f} | {row['route']} | "
            f"`{row['train_curves']}` | {float(row['tau']):.0f} | `{row['nuisance']}` |\n"
        )
    lines += [
        "\n## Per-Target Summary\n\n",
        "| target | mean | worst | non-harm |\n",
        "|---|---:|---:|---:|\n",
    ]
    for row in summary:
        lines.append(
            f"| {row['target_label']} | {float(row['mean_delta']):+.1f}% | "
            f"{float(row['worst_delta']):+.1f}% | {int(row['nonharm'])}/{int(row['rows'])} |\n"
        )
    lines += [
        "\n## Extended Safety Audit\n\n",
        "The extended audit adds `cosine_24000`, `constant_24000`, and `constant_72000`.  The first is a short smooth-cosine control; the constants are zero-positive-drop controls.  The routed rule applies no transient correction to these cases.\n\n",
        f"- Extended all-target audit: mean `{float(extended['mean_delta']):+.1f}%`, worst `{float(extended['worst_delta']):+.1f}%`, non-harm `{int(extended['nonharm'])}/{int(extended['rows'])}`.\n",
        f"- Extra safety controls only: mean `{float(extra['mean_delta']):+.1f}%`, worst `{float(extra['worst_delta']):+.1f}%`, non-harm `{int(extra['nonharm'])}/{int(extra['rows'])}`.\n\n",
        "| extra target | route | source | tau | mean | worst | non-harm |\n",
        "|---|---|---|---:|---:|---:|---:|\n",
    ]
    route_by_target = {str(row["target_curve"]): row for row in extended_routes}
    for target_curve, target_label in EXTRA_SAFETY_CURVES:
        row = next(r for r in extended_summary if r["target_curve"] == target_curve)
        route = route_by_target[target_curve]
        lines.append(
            f"| {target_label} | {route['route']} | `{route['train_curves']}` | "
            f"{float(route['tau']):.0f} | {float(row['mean_delta']):+.1f}% | "
            f"{float(row['worst_delta']):+.1f}% | {int(row['nonharm'])}/{int(row['rows'])} |\n"
        )
    lines += [
        "\n## Ablation and Tau Robustness\n\n",
        "| audit | mean | worst | non-harm | reading |\n",
        "|---|---:|---:|---:|---|\n",
    ]
    for row in ablation_summary:
        lines.append(
            f"| {row['audit']} | {float(row['mean_delta']):+.1f}% | "
            f"{float(row['worst_delta']):+.1f}% | {int(row['nonharm'])}/{int(row['rows'])} | "
            f"{row['description']} |\n"
        )
    lines += [
        "\n## Comparison\n\n",
        "| metric | mean | worst | non-harm |\n",
        "|---|---:|---:|---:|\n",
    ]
    for row in comparison:
        lines.append(
            f"| {row['metric']} | {float(row['mean_delta']):+.1f}% | "
            f"{float(row['worst_delta']):+.1f}% | {row['nonharm']} |\n"
        )
    lines += [
        "\n## Protocol and Overfit Audits\n\n",
        "- `PROTOCOL_AUDIT.md` checks target-loss blindness: committed routes match LR-schedule-only recomputation, the target curve is excluded from calibration, and scrambling the target residual leaves kappa and the correction unchanged.\n",
        "- `OVERFIT_RISK_AUDIT.md` is the deliberately conservative reading: the result is stable across scale slices and local tau perturbations, but route/tau choices are still too benchmark-shaped to claim external prospective validation.\n",
        "\n## Reading\n\n",
        "- The previous decomposed estimator is still the right self-fit model: it explains the broad low-frequency residual without treating it as transferable lag.\n",
        "- The shape-routed head is stronger for deployment because it does not average over impossible source-target pairs.  It asks: given this target schedule shape, which calibration schedule should supply the transient amplitude?\n",
        "- This is still an internal schedule-family audit.  The next evidence needed is an external schedule family or a leave-family design with more than one curve per family.\n",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    routes, details, summary = run_shape_routed(CORE_CURVES, "shape_routed_target_holdout")
    extended_routes, extended_details, extended_summary = run_shape_routed(
        EXTENDED_CURVES, "shape_routed_extended_safety"
    )
    comparison = comparison_rows(details)
    ablation_summary, ablation_details = run_ablation_audits()

    write_csv(OUT_DIR / "route_table.csv", routes)
    write_csv(OUT_DIR / "target_holdout_details.csv", details)
    write_csv(OUT_DIR / "target_holdout_summary.csv", summary)
    write_csv(OUT_DIR / "extended_route_table.csv", extended_routes)
    write_csv(OUT_DIR / "extended_safety_details.csv", extended_details)
    write_csv(OUT_DIR / "extended_safety_summary.csv", extended_summary)
    write_csv(OUT_DIR / "ablation_summary.csv", ablation_summary)
    write_csv(OUT_DIR / "ablation_details.csv", ablation_details)
    write_csv(OUT_DIR / "comparison.csv", comparison)
    plot_target_holdout(FIG_DIR / "shape_routed_target_holdout.png", details, summary)
    write_report(
        routes,
        details,
        summary,
        comparison,
        extended_routes,
        extended_details,
        extended_summary,
        ablation_summary,
    )

    shape = summarize(details)
    extended = summarize(extended_details)
    print(f"wrote {OUT_DIR / 'REPORT.md'}")
    print(
        f"shape-routed target-holdout={float(shape['mean_delta']):+.1f}%/"
        f"{float(shape['worst_delta']):+.1f}% nonharm={int(shape['nonharm'])}/{int(shape['rows'])}"
    )
    print(
        f"extended safety={float(extended['mean_delta']):+.1f}%/"
        f"{float(extended['worst_delta']):+.1f}% nonharm={int(extended['nonharm'])}/{int(extended['rows'])}"
    )


if __name__ == "__main__":
    main()
