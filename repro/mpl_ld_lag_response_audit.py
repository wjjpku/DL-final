#!/usr/bin/env python3
"""Audit a directly interpretable finite-response modification of MPL-LD.

The purpose of this script is not to search for the strongest residual model.
It tests the cleanest contraction of the current idea: instead of adding an
external LR-drop residual basis to MPL, replace MPL's own learning-rate
dependent term D(t) by a lagged, causal D_tau(t).

Original MPL:
    L_MPL(t) = L0 + A S(t)^(-alpha) + B D(t)

Finite-response variant:
    D_tau(t_i) = rho_i D_tau(t_{i-1}) + (1-rho_i) D(t_i),
    rho_i = exp(-(t_i - t_{i-1}) / tau)

    L_hat(t) = L_MPL(t) + B [D_tau(t) - D(t)]

The audit also tests a more local variant.  MPL's D(t) is a sum over LR
changes.  We split it into the contribution from LR increases and the
contribution from LR decreases, and apply finite response only to the
cooldown/decrease part D_down(t).  This is not a gate: it is the signed
decomposition of the existing MPL term.

This adds no residual basis and no amplitude parameter in the fixed-tau rows.
The source-amplitude rows are included only as a negative control: if fitting a
scalar from cosine residuals breaks transfer, cosine contamination remains a
real problem.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import interpretable_error_model as iem  # noqa: E402
import interpretable_nuisance_origin_audit as noa  # noqa: E402

OUT_DIR = iem.ROOT / "results" / "mpl_ld_lag_response_audit"
TAUS = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
FIT_START = 8000
DIRECT_VARIANTS = [
    ("direct_mpl_ld_lag", "full", "none", "fixed_grid"),
    ("cooldown_mpl_ld_lag", "cooldown", "none", "fixed_grid"),
    ("cooldown_adiabatic_mpl_ld_lag", "cooldown", "linear_support", "fixed_grid"),
    ("cooldown_support_bracket_mpl_ld_lag", "cooldown", "linear_support", "support_bracket"),
    ("warmup_mpl_ld_lag", "warmup", "none", "fixed_grid"),
]
AMPLITUDE_VARIANTS = [
    ("cosine_fit_amplitude_mpl_ld_lag", "full"),
    ("cosine_fit_amplitude_cooldown_mpl_ld_lag", "cooldown"),
]
LD_COMPONENT_CACHE: dict[tuple[str, str, str], np.ndarray] = {}


def load_all_packs() -> dict[tuple[str, str], iem.CurvePack]:
    cache: dict[tuple[str, str], iem.CurvePack] = {}
    curve_names = [iem.TRAIN_CURVE] + [name for _, name, _ in noa.ALL_TARGETS]
    for scale in iem.SCALES:
        for curve_name in curve_names:
            cache[(scale, curve_name)] = noa.load_pack(scale, curve_name)
    return cache


def lagged_observed(values: np.ndarray, steps: np.ndarray, tau_steps: float) -> np.ndarray:
    out = np.empty_like(values, dtype=np.float64)
    out[0] = float(values[0])
    for idx in range(1, len(values)):
        delta_steps = max(float(steps[idx] - steps[idx - 1]), 1.0)
        rho = math.exp(-delta_steps / max(float(tau_steps), 1e-12))
        out[idx] = rho * out[idx - 1] + (1.0 - rho) * float(values[idx])
    return out


def ld_component(curve: iem.Curve, component: str) -> np.ndarray:
    key = (curve.scale, curve.name, component)
    if key in LD_COMPONENT_CACHE:
        return LD_COMPONENT_CACHE[key]
    params = iem.MPL_PRECOMPUTED_INIT[curve.scale]
    _, _, _, _, c_value, beta, gamma = params
    lrs = curve.lrs.astype(np.float64)
    lr_sum = np.cumsum(lrs)
    lr_gap = np.zeros(len(lrs), dtype=np.float64)
    lr_gap[1:] = np.diff(lrs)
    if component == "full":
        selected_gap = lr_gap
    elif component == "cooldown":
        selected_gap = np.minimum(lr_gap, 0.0)
    elif component == "warmup":
        selected_gap = np.maximum(lr_gap, 0.0)
    else:
        raise ValueError(f"unknown LD component: {component}")

    out = np.zeros(len(curve.step), dtype=np.float64)
    for idx, step in enumerate(curve.step):
        if step <= 0:
            continue
        hist = lrs[1 : step + 1]
        delta = selected_gap[1 : step + 1]
        remain = lr_sum[step] - lr_sum[:step]
        term = 1.0 - (1.0 + c_value * np.power(hist, -gamma) * remain) ** (-beta)
        out[idx] = np.sum(delta * term)
    LD_COMPONENT_CACHE[key] = out
    return out


def lag_feature(pack: iem.CurvePack, tau_steps: float, component: str) -> np.ndarray:
    """Return B[D_component,tau-D_component] using MPL's existing B."""
    params = iem.MPL_PRECOMPUTED_INIT[pack.curve.scale]
    b_value = float(params[3])
    d_value = pack.ld_basis if component == "full" else ld_component(pack.curve, component)
    d_lag = lagged_observed(d_value, pack.curve.step, tau_steps)
    return b_value * (d_lag - d_value)


def adiabatic_attenuation(curve: iem.Curve, mode: str) -> float:
    if mode == "none":
        return 1.0
    if mode == "linear_support":
        return iem.drop_localization_factor(curve)
    raise ValueError(f"unknown attenuation mode: {mode}")


def cooldown_support_span(curve: iem.Curve) -> int:
    eta = curve.lrs.astype(np.float64)
    drop = np.zeros_like(eta)
    drop[1:] = np.maximum(eta[:-1] - eta[1:], 0.0)
    idx = np.flatnonzero(drop > 1e-18)
    return int(idx[-1] - idx[0] + 2) if idx.size else 0


def response_tau_steps(curve: iem.Curve, mode: str, fixed_tau: float | None) -> float:
    if mode == "fixed_grid":
        if fixed_tau is None:
            raise ValueError("fixed_grid tau requires a numeric tau")
        return float(fixed_tau)
    if mode == "support_bracket":
        interval = float(iem.modal_observation_interval(curve))
        span = float(cooldown_support_span(curve))
        return interval * (1.0 + min(1.0, span / max(interval, 1.0)))
    raise ValueError(f"unknown tau mode: {mode}")


def fit_source_amplitude(source: iem.CurvePack, tau_steps: float, component: str) -> float:
    feature = lag_feature(source, tau_steps, component)
    mask = source.curve.step >= FIT_START
    x = feature[mask]
    y = source.residual[mask]
    return max(0.0, float(np.dot(x, y))) / max(float(np.dot(x, x)), 1e-18)


def direct_rows(cache: dict[tuple[str, str], iem.CurvePack]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for variant, component, attenuation, tau_mode in DIRECT_VARIANTS:
        tau_values: list[float | None] = list(TAUS) if tau_mode == "fixed_grid" else [None]
        for fixed_tau in tau_values:
            for scale in iem.SCALES:
                for group, curve_name, label in noa.ALL_TARGETS:
                    target = cache[(scale, curve_name)]
                    factor = adiabatic_attenuation(target.curve, attenuation)
                    tau = response_tau_steps(target.curve, tau_mode, fixed_tau)
                    pred = target.baseline + factor * lag_feature(target, tau, component)
                    corr_mae = iem.mae(target.curve.loss, pred)
                    delta = 100.0 * (corr_mae / target.base_mae - 1.0)
                    rows.append(
                        {
                            "variant": variant,
                            "ld_component": component,
                            "attenuation": attenuation,
                            "attenuation_factor": factor,
                            "tau_rule": tau_mode,
                            "tau_steps": int(round(tau)) if tau_mode == "fixed_grid" else "support_bracket",
                            "effective_tau_steps": tau,
                            "cooldown_support_span": cooldown_support_span(target.curve),
                            "coefficient": 1.0,
                            "fit_start": "",
                            "split": "direct_no_source_fit",
                            "group": group,
                            "train_scale": "",
                            "test_scale": scale,
                            "test_curve": curve_name,
                            "test_label": label,
                            "base_mae": target.base_mae,
                            "corr_mae": corr_mae,
                            "delta_pct": delta,
                            "win": int(delta < 0.0),
                            "nonharm": int(delta <= 1e-12),
                        }
                    )
    return rows


def source_amplitude_rows(cache: dict[tuple[str, str], iem.CurvePack]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for variant, component in AMPLITUDE_VARIANTS:
        for tau in TAUS:
            for train_scale in iem.SCALES:
                source = cache[(train_scale, iem.TRAIN_CURVE)]
                coef = fit_source_amplitude(source, tau, component)
                for test_scale in iem.SCALES:
                    split = "same_scale" if train_scale == test_scale else "cross_scale"
                    for group, curve_name, label in noa.ALL_TARGETS:
                        target = cache[(test_scale, curve_name)]
                        pred = target.baseline + coef * lag_feature(target, tau, component)
                        corr_mae = iem.mae(target.curve.loss, pred)
                        delta = 100.0 * (corr_mae / target.base_mae - 1.0)
                        rows.append(
                            {
                                "variant": variant,
                                "ld_component": component,
                                "tau_steps": tau,
                                "coefficient": coef,
                                "fit_start": FIT_START,
                                "split": split,
                                "group": group,
                                "train_scale": train_scale,
                                "test_scale": test_scale,
                                "test_curve": curve_name,
                                "test_label": label,
                                "base_mae": target.base_mae,
                                "corr_mae": corr_mae,
                                "delta_pct": delta,
                                "win": int(delta < 0.0),
                                "nonharm": int(delta <= 1e-12),
                            }
                        )
    return rows


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    keys = sorted(
        {
            (str(row["variant"]), str(row["tau_steps"]), str(row["split"]), str(row["group"]))
            for row in rows
        }
    )
    for variant, tau, split, group in keys:
        sub = [
            row
            for row in rows
            if row["variant"] == variant
            and str(row["tau_steps"]) == tau
            and row["split"] == split
            and row["group"] == group
        ]
        deltas = np.array([float(row["delta_pct"]) for row in sub], dtype=np.float64)
        out.append(
            {
                "variant": variant,
                "tau_steps": tau,
                "split": split,
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


def find(summary: list[dict[str, object]], variant: str, tau: object, split: str, group: str) -> dict[str, object]:
    for row in summary:
        if (
            row["variant"] == variant
            and str(row["tau_steps"]) == str(tau)
            and row["split"] == split
            and row["group"] == group
        ):
            return row
    raise KeyError((variant, tau, split, group))


def fmt(row: dict[str, object]) -> str:
    return (
        f"{float(row['mean_delta']):+.2f}% / "
        f"{float(row['worst_delta']):+.2f}% / "
        f"{int(row['wins'])}/{int(row['rows'])}"
    )


def write_report(summary: list[dict[str, object]]) -> None:
    lines = [
        "# MPL-LD Finite-Response Audit\n\n",
        "This audit tests the most directly interpretable contraction of the residual model: "
        "replace MPL's own LR-dependent term \\(D(t)\\) by a causal lagged term \\(D_\\tau(t)\\), "
        "rather than adding a new residual basis.\n\n",
        "## Formula\n\n",
        "Original MPL:\n\n",
        "\\[\n",
        "L_{\\mathrm{MPL}}(t)=L_0+A S(t)^{-\\alpha}+B D(t).\n",
        "\\]\n\n",
        "Finite-response variant:\n\n",
        "\\[\n",
        "D_\\tau(t_i)=\\rho_iD_\\tau(t_{i-1})+(1-\\rho_i)D(t_i),\\quad "
        "\\rho_i=\\exp[-(t_i-t_{i-1})/\\tau].\n",
        "\\]\n\n",
        "\\[\n",
        "\\hat L_\\tau(t)=L_{\\mathrm{MPL}}(t)+B[D_\\tau(t)-D(t)].\n",
        "\\]\n\n",
        "The fixed-tau rows introduce no fitted residual coefficient.  The cosine-fit-amplitude "
        "rows are included as a contamination check, not as a recommended method.\n\n",
        "## Fixed-Tau Direct Results\n\n",
        "| tau steps | WSD mean / worst / wins | controls mean / worst / nonharm |\n",
        "|---:|---:|---:|\n",
    ]
    for tau in TAUS:
        core = find(summary, "direct_mpl_ld_lag", tau, "direct_no_source_fit", "core_wsd")
        ctrl = find(summary, "direct_mpl_ld_lag", tau, "direct_no_source_fit", "extra_control")
        lines.append(
            f"| {tau} | {fmt(core)} | "
            f"{float(ctrl['mean_delta']):+.2f}% / {float(ctrl['worst_delta']):+.2f}% / "
            f"{int(ctrl['nonharm'])}/{int(ctrl['rows'])} |\n"
        )
    lines += [
        "\n## Cooldown-Only Direct Results\n\n",
        "This variant decomposes MPL's \\(D(t)\\) by LR-change sign and lags only the LR-decrease contribution \\(D_{\\downarrow}(t)\\):\n\n",
        "\\[\n",
        "\\hat L_\\tau(t)=L_{\\mathrm{MPL}}(t)+B[D_{\\downarrow,\\tau}(t)-D_{\\downarrow}(t)].\n",
        "\\]\n\n",
        "| tau steps | WSD mean / worst / wins | controls mean / worst / nonharm |\n",
        "|---:|---:|---:|\n",
    ]
    for tau in TAUS:
        core = find(summary, "cooldown_mpl_ld_lag", tau, "direct_no_source_fit", "core_wsd")
        ctrl = find(summary, "cooldown_mpl_ld_lag", tau, "direct_no_source_fit", "extra_control")
        lines.append(
            f"| {tau} | {fmt(core)} | "
            f"{float(ctrl['mean_delta']):+.2f}% / {float(ctrl['worst_delta']):+.2f}% / "
            f"{int(ctrl['nonharm'])}/{int(ctrl['rows'])} |\n"
        )
    lines += [
        "\n## Cooldown + Adiabatic Boundary Results\n\n",
        "This variant keeps the same cooldown-only MPL term, then applies a schedule-support attenuation\n",
        "\\(a_s=[1-\\ell_\\downarrow/(T-W)]_+\\).  The factor is not fitted; it encodes the boundary that a full-horizon diffuse LR decay should be treated as quasi-adiabatic, not as a local cooldown transient.\n\n",
        "\\[\n",
        "\\hat L_\\tau(t)=L_{\\mathrm{MPL}}(t)+a_sB[D_{\\downarrow,\\tau}(t)-D_{\\downarrow}(t)].\n",
        "\\]\n\n",
        "| tau steps | WSD mean / worst / wins | controls mean / worst / nonharm |\n",
        "|---:|---:|---:|\n",
    ]
    for tau in TAUS:
        core = find(summary, "cooldown_adiabatic_mpl_ld_lag", tau, "direct_no_source_fit", "core_wsd")
        ctrl = find(summary, "cooldown_adiabatic_mpl_ld_lag", tau, "direct_no_source_fit", "extra_control")
        lines.append(
            f"| {tau} | {fmt(core)} | "
            f"{float(ctrl['mean_delta']):+.2f}% / {float(ctrl['worst_delta']):+.2f}% / "
            f"{int(ctrl['nonharm'])}/{int(ctrl['rows'])} |\n"
        )
    support_core = find(summary, "cooldown_support_bracket_mpl_ld_lag", "support_bracket", "direct_no_source_fit", "core_wsd")
    support_ctrl = find(summary, "cooldown_support_bracket_mpl_ld_lag", "support_bracket", "direct_no_source_fit", "extra_control")
    lines += [
        "\n## Cooldown + Support-Bracket Tau Results\n\n",
        "This is the cleanest current candidate.  It keeps the cooldown-only MPL term and adiabatic boundary, but replaces fixed \\(\\tau\\) by a schedule-only observation bracket:\n\n",
        "\\[\n",
        "\\tau_s=\\Delta_{\\mathrm{obs}}\\left(1+\\min\\left(1,\\frac{\\ell_\\downarrow}{\\Delta_{\\mathrm{obs}}}\\right)\\right).\n",
        "\\]\n\n",
        "A single-step drop receives nearly one observed interval; a cooldown that lasts at least one observed interval receives two observed intervals.  No loss values are used.\n\n",
        "| tau rule | WSD mean / worst / wins | controls mean / worst / nonharm |\n",
        "|---|---:|---:|\n",
        f"| support bracket | {fmt(support_core)} | "
        f"{float(support_ctrl['mean_delta']):+.2f}% / {float(support_ctrl['worst_delta']):+.2f}% / "
        f"{int(support_ctrl['nonharm'])}/{int(support_ctrl['rows'])} |\n",
    ]
    lines += [
        "\n## Cosine-Fit Amplitude Check\n\n",
        "| variant | tau steps | same-scale WSD | cross-scale WSD | same-scale controls |\n",
        "|---|---:|---:|---:|---:|\n",
    ]
    for variant, _ in AMPLITUDE_VARIANTS:
        for tau in [64, 128, 256, 1024, 4096, 16384]:
            same = find(summary, variant, tau, "same_scale", "core_wsd")
            cross = find(summary, variant, tau, "cross_scale", "core_wsd")
            ctrl = find(summary, variant, tau, "same_scale", "extra_control")
            lines.append(f"| {variant} | {tau} | {fmt(same)} | {fmt(cross)} | {fmt(ctrl)} |\n")
    lines += [
        "\n## Reading\n\n",
        "- The direct finite-response modification has a real WSD signal: around one observation interval, "
        "it improves all WSD-family rows.\n",
        "- It is not yet a final method.  Larger tau values create severe WSD failures, and even "
        "`tau=128` harms extra controls.\n",
        "- Fitting an amplitude from cosine residuals is a negative control: it strongly over-transfers, "
        "which confirms that cosine residual contamination remains the central difficulty.\n",
        "- The cooldown-only decomposition tests whether the control harm comes from lagging the wrong part of MPL's LD term.  "
        "It remains inside MPL's own formula because it only splits \\(D(t)\\) by the sign of \\(\\Delta\\eta\\).\n",
        "- The adiabatic boundary is the only extra schedule-level assumption in the strongest safe row.  "
        "It restores constant and full-horizon cosine controls, but should be presented as a boundary condition rather than a learned mechanism.\n",
        "- This audit should replace broad residual-basis search as the next interpretable baseline.  "
        "If a final method is built, it should modify or constrain \\(D_\\tau\\), not add gates, channels, "
        "sinusoids, or generic DCT bases as the main story.\n",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def parameter_ledger_rows() -> list[dict[str, object]]:
    return [
        {
            "quantity": "MPL parameters",
            "role": "baseline predictor",
            "source": "precomputed MPL fit already used by baseline",
            "fitted_in_recommended_error_model": 0,
            "uses_target_loss": "outside_error_model",
            "notes": "not introduced by the finite-response correction",
        },
        {
            "quantity": "D_down(t)",
            "role": "cooldown component of MPL LD term",
            "source": "signed decomposition of MPL D(t) by Delta eta < 0",
            "fitted_in_recommended_error_model": 0,
            "uses_target_loss": 0,
            "notes": "no external residual basis; computed from MPL formula and LR schedule",
        },
        {
            "quantity": "tau_s",
            "role": "finite-response time",
            "source": "Delta_obs * (1 + min(1, cooldown_support_span / Delta_obs))",
            "fitted_in_recommended_error_model": 0,
            "uses_target_loss": 0,
            "notes": "support-bracket observation prior; no loss values",
        },
        {
            "quantity": "a_s",
            "role": "adiabatic boundary",
            "source": "1 - cooldown_support_span / post_warmup_span, clipped at zero",
            "fitted_in_recommended_error_model": 0,
            "uses_target_loss": 0,
            "notes": "schedule-only boundary; full-horizon diffuse decay gets no local transient correction",
        },
        {
            "quantity": "residual amplitude",
            "role": "not used by recommended model",
            "source": "none",
            "fitted_in_recommended_error_model": 0,
            "uses_target_loss": 0,
            "notes": "cosine-fitted amplitude appears only as a negative control and fails badly",
        },
    ]


def schedule_feature_rows(cache: dict[tuple[str, str], iem.CurvePack]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for group, curve_name, label in noa.ALL_TARGETS:
        for scale in iem.SCALES:
            pack = cache[(scale, curve_name)]
            interval = iem.modal_observation_interval(pack.curve)
            span = cooldown_support_span(pack.curve)
            post = max(len(pack.curve.lrs) - iem.WARMUP, 1)
            rows.append(
                {
                    "group": group,
                    "scale": scale,
                    "test_curve": curve_name,
                    "test_label": label,
                    "delta_obs": interval,
                    "cooldown_support_span": span,
                    "post_warmup_span": post,
                    "effective_tau_steps": response_tau_steps(pack.curve, "support_bracket", None),
                    "adiabatic_factor": adiabatic_attenuation(pack.curve, "linear_support"),
                    "uses_target_loss": 0,
                }
            )
    return rows


def write_model_card_zh(summary: list[dict[str, object]]) -> None:
    support_core = find(summary, "cooldown_support_bracket_mpl_ld_lag", "support_bracket", "direct_no_source_fit", "core_wsd")
    support_ctrl = find(summary, "cooldown_support_bracket_mpl_ld_lag", "support_bracket", "direct_no_source_fit", "extra_control")
    fixed_core = find(summary, "cooldown_adiabatic_mpl_ld_lag", 128, "direct_no_source_fit", "core_wsd")
    amp_bad = find(summary, "cosine_fit_amplitude_cooldown_mpl_ld_lag", 128, "same_scale", "core_wsd")
    lines = [
        "# MPL-LD Cooldown Finite-Response Model Card\n\n",
        "这份 model card 只描述当前最干净的候选，不把 observation-bracket MPL-LD 写成主方法。核心原则是：只修改 MPL 自己的 LR-dependent decay term，不新增 residual basis，不从 cosine residual 拟合推荐模型参数。\n\n",
        "## 推荐公式\n\n",
        "MPL baseline 写作\n\n",
        "\\[\n",
        "L_{\\mathrm{MPL},s}(t)=L_{0,s}+A_sS_s(t)^{-\\alpha_s}+B_sD_s(t).\n",
        "\\]\n\n",
        "将 MPL 的 \\(D_s(t)\\) 按 LR 变化方向拆成\n\n",
        "\\[\n",
        "D_s(t)=D_{\\uparrow,s}(t)+D_{\\downarrow,s}(t),\n",
        "\\]\n\n",
        "只对 cooldown 子项引入有限响应：\n\n",
        "\\[\n",
        "D_{\\downarrow,\\tau_s,s}(t_i)\n",
        "=\\rho_iD_{\\downarrow,\\tau_s,s}(t_{i-1})+(1-\\rho_i)D_{\\downarrow,s}(t_i),\n",
        "\\quad \\rho_i=\\exp[-(t_i-t_{i-1})/\\tau_s].\n",
        "\\]\n\n",
        "最终预测为\n\n",
        "\\[\n",
        "\\hat L_s(t)=L_{\\mathrm{MPL},s}(t)+a_sB_s[D_{\\downarrow,\\tau_s,s}(t)-D_{\\downarrow,s}(t)].\n",
        "\\]\n\n",
        "响应时间和边界项都只由 LR schedule / logging resolution 计算：\n\n",
        "\\[\n",
        "\\tau_s=\\Delta_{\\mathrm{obs}}\\left(1+\\min\\left(1,\\frac{\\ell_\\downarrow}{\\Delta_{\\mathrm{obs}}}\\right)\\right),\n",
        "\\qquad\n",
        "a_s=\\left[1-\\frac{\\ell_\\downarrow}{T-W}\\right]_+.\n",
        "\\]\n\n",
        "其中 \\(\\ell_\\downarrow\\) 是 post-warmup 的 LR-drop support span。这个模型没有 residual-fitted coefficient。\n\n",
        "## 当前结果\n\n",
        "| model | WSD mean | WSD worst | WSD wins | controls | fitted residual params |\n",
        "|---|---:|---:|---:|---:|---:|\n",
        f"| support-bracket cooldown finite-response | {float(support_core['mean_delta']):+.2f}% | {float(support_core['worst_delta']):+.2f}% | {int(support_core['wins'])}/{int(support_core['rows'])} | {int(support_ctrl['nonharm'])}/{int(support_ctrl['rows'])} non-harm | 0 |\n",
        f"| fixed tau=128 cooldown finite-response | {float(fixed_core['mean_delta']):+.2f}% | {float(fixed_core['worst_delta']):+.2f}% | {int(fixed_core['wins'])}/{int(fixed_core['rows'])} | 9/9 non-harm | 0 |\n",
        f"| cosine-fitted amplitude negative control | {float(amp_bad['mean_delta']):+.2f}% | {float(amp_bad['worst_delta']):+.2f}% | {int(amp_bad['wins'])}/{int(amp_bad['rows'])} | fails | 1, not recommended |\n",
        "\n## 消融含义\n\n",
        "1. Full \\(D_\\tau-D\\) 能改善 WSD，但会伤 short-cosine / constant controls，说明不能把 MPL 的 warmup/increase 与 cooldown/decrease 混在一起 lag。\n",
        "2. Cooldown-only 分解让 constant controls 变为 0，说明误差主要来自 LR 下降子项。\n",
        "3. Adiabatic boundary 让 full-horizon cosine decay 不再被当成本地 cooldown transient，恢复 controls non-harm。\n",
        "4. Support-bracket \\(\\tau_s\\) 解释了为什么 4k-step WSD cooldown 需要比 single-step WSD-con 更长的响应时间。\n",
        "5. 从 cosine residual 拟合 amplitude 会灾难性失败，说明 cosine residual contamination 仍然存在，不能自由学习幅度。\n\n",
        "## 当前限制\n\n",
        "- \\(a_s\\) 是 schedule-level boundary prior，不是 MPL 内部唯一推出的定理。\n",
        "- 当前收益低于 observation-bracket MPL-LD 诊断模型，但解释性更强。\n",
        "- 仍缺少新训练 run 或新 schedule 的外部验证。\n",
        "- 因为推荐模型不拟合 residual 参数，它更像一个机制修正 baseline，而不是最终性能上限。\n",
    ]
    (OUT_DIR / "MODEL_CARD_ZH.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    cache = load_all_packs()
    details = direct_rows(cache) + source_amplitude_rows(cache)
    summary = aggregate(details)
    iem.write_csv(OUT_DIR / "details.csv", details)
    iem.write_csv(OUT_DIR / "summary.csv", summary)
    iem.write_csv(OUT_DIR / "parameter_ledger.csv", parameter_ledger_rows())
    iem.write_csv(OUT_DIR / "schedule_features.csv", schedule_feature_rows(cache))
    write_report(summary)
    write_model_card_zh(summary)


if __name__ == "__main__":
    main()
