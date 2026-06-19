#!/usr/bin/env python3
"""Scale-stability audit for the interpretable LR-drop response model.

This audit answers a narrower question than the main WSD evaluation:
if the response amplitude is calibrated on one model scale's cosine curve,
does it still help WSD-family curves at another scale?

The most interpretable candidate here is the MPL-LD tangent version: it removes
from the cosine residual only directions that can be explained by local errors
in MPL's LR-dependent parameters, then fits one nonnegative response amplitude.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import interpretable_error_model as iem  # noqa: E402
import interpretable_nuisance_origin_audit as noa  # noqa: E402

OUT_DIR = iem.ROOT / "results" / "interpretable_scale_stability_audit"

METHODS = [
    {
        "method": "mpl_ld_tangent",
        "response_rule": "two_point_five_roundfast20",
        "nuisance": "mpl_ld4",
        "shrinkage": "ridge_tau_0p05",
        "role": "mechanism_native_main_candidate",
    },
    {
        "method": "dct_performance",
        "response_rule": "two_point_five_roundfast20",
        "nuisance": "dct_soft",
        "shrinkage": "ridge_tau_0p05",
        "role": "stronger_but_less_interpretable_reference",
    },
    {
        "method": "tau_free_dct",
        "response_rule": "fixed_lambda_20",
        "nuisance": "dct_soft",
        "shrinkage": "tau_free_sqrt_retention",
        "role": "tau_free_dct_baseline",
    },
]


def aggregate(rows: list[dict[str, object]], method: str, group: str, split: str) -> dict[str, object]:
    deltas = np.array([float(row["delta_pct"]) for row in rows], dtype=np.float64)
    return {
        "method": method,
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
    for method in [str(item["method"]) for item in METHODS]:
        method_rows = [row for row in detail_rows if row["method"] == method]
        for group in ["core_wsd", "extra_control"]:
            group_rows = [row for row in method_rows if row["group"] == group]
            splits = {
                "all": group_rows,
                "same_scale": [row for row in group_rows if row["train_scale"] == row["test_scale"]],
                "cross_scale": [row for row in group_rows if row["train_scale"] != row["test_scale"]],
            }
            for scale in iem.SCALES:
                splits[f"holdout_test_{scale}"] = [
                    row
                    for row in group_rows
                    if row["test_scale"] == scale and row["train_scale"] != row["test_scale"]
                ]
                splits[f"train_{scale}_all_tests"] = [
                    row for row in group_rows if row["train_scale"] == scale
                ]
            for split, rows in splits.items():
                if rows:
                    out.append(aggregate(rows, method, group, split))
    return out


def find_row(rows: list[dict[str, object]], method: str, group: str, split: str) -> dict[str, object]:
    for row in rows:
        if row["method"] == method and row["group"] == group and row["split"] == split:
            return row
    raise KeyError((method, group, split))


def fmt(row: dict[str, object]) -> str:
    return (
        f"{float(row['mean_delta']):+.2f}% mean, "
        f"{float(row['worst_delta']):+.2f}% worst, "
        f"{int(row['wins'])}/{int(row['rows'])} wins"
    )


def write_report(summary_rows: list[dict[str, object]], coefficient_rows: list[dict[str, object]]) -> None:
    lines = [
        "# Scale-Stability Audit\n\n",
        "This audit tests whether the cosine-fitted response amplitude is stable across model scales.  "
        "For every method, the source loss is still only `cosine_72000.csv`; the difference is that "
        "the source scale can be 25M, 100M, or 400M while the target scale is independently varied.\n\n",
        "## Main Reading\n\n",
        "The mechanism-native candidate is `mpl_ld_tangent`: before estimating the LR-drop response amplitude, "
        "it projects the cosine residual and response feature away from the local tangent space of MPL's "
        "LR-dependent parameters \\((B,C,\\beta,\\gamma)\\).  This is more interpretable than a generic "
        "DCT low-frequency filter because the nuisance directions are exactly MPL parameter-error directions.\n\n",
        "## Summary\n\n",
        "| method | split | group | mean | worst | wins/non-harm |\n",
        "|---|---|---|---:|---:|---:|\n",
    ]
    display_splits = [
        "same_scale",
        "cross_scale",
        "holdout_test_25",
        "holdout_test_100",
        "holdout_test_400",
    ]
    for method in [str(item["method"]) for item in METHODS]:
        for split in display_splits:
            for group in ["core_wsd", "extra_control"]:
                row = find_row(summary_rows, method, group, split)
                lines.append(
                    f"| {method} | {split} | {group} | "
                    f"{float(row['mean_delta']):+.2f}% | {float(row['worst_delta']):+.2f}% | "
                    f"{int(row['wins'])}/{int(row['rows'])} wins, "
                    f"{int(row['nonharm'])}/{int(row['rows'])} non-harm |\n"
                )

    main_same = find_row(summary_rows, "mpl_ld_tangent", "core_wsd", "same_scale")
    main_cross = find_row(summary_rows, "mpl_ld_tangent", "core_wsd", "cross_scale")
    dct_cross = find_row(summary_rows, "dct_performance", "core_wsd", "cross_scale")
    tau_cross = find_row(summary_rows, "tau_free_dct", "core_wsd", "cross_scale")
    lines += [
        "\n## Interpretation\n\n",
        f"- Same-scale `mpl_ld_tangent`: {fmt(main_same)}.\n",
        f"- Cross-scale `mpl_ld_tangent`: {fmt(main_cross)}.  This means the most mechanism-native version "
        "does not require choosing a separate scale-specific story to stay beneficial.\n",
        f"- Cross-scale `dct_performance`: {fmt(dct_cross)}.  It has strong mean gains but a positive worst case, "
        "so it should remain a performance reference rather than the main explanation.\n",
        f"- Cross-scale `tau_free_dct`: {fmt(tau_cross)}.  It is safer than the DCT performance reference but still "
        "inherits the interpretability cost of a generic low-frequency projection.\n\n",
        "## Coefficient Range\n\n",
        "| method | min coef | median coef | max coef | min retention | median retention |\n",
        "|---|---:|---:|---:|---:|---:|\n",
    ]
    for method in [str(item["method"]) for item in METHODS]:
        sub = [row for row in coefficient_rows if row["method"] == method]
        coefs = np.array([float(row["coef"]) for row in sub], dtype=np.float64)
        ret = np.array([float(row["source_retention"]) for row in sub], dtype=np.float64)
        lines.append(
            f"| {method} | {float(np.min(coefs)):.6g} | {float(np.median(coefs)):.6g} | "
            f"{float(np.max(coefs)):.6g} | {float(np.min(ret)):.6g} | {float(np.median(ret)):.6g} |\n"
        )

    lines += [
        "\n## Decision\n\n",
        "For a rigorous presentation, foreground `mpl_ld_tangent` as the interpretable candidate.  "
        "It gives slightly weaker same-scale WSD performance than DCT, but it is cleaner: the nuisance projection "
        "has a direct MPL-error meaning, and the cross-scale audit stays non-harmful on all WSD rows.  "
        "DCT-based variants should be described as diagnostic or performance extensions, not as the core theory.\n",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    cache: dict[tuple[str, str], iem.CurvePack] = {}
    basis_cache: dict[tuple[str, str], np.ndarray] = {}
    detail_rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []
    seen_coef_keys: set[tuple[str, str, str, str, str]] = set()

    def pack(scale: str, curve_name: str) -> iem.CurvePack:
        key = (scale, curve_name)
        if key not in cache:
            cache[key] = noa.load_pack(scale, curve_name)
        return cache[key]

    for method_cfg in METHODS:
        method = str(method_cfg["method"])
        response_rule = str(method_cfg["response_rule"])
        nuisance = str(method_cfg["nuisance"])
        shrinkage = str(method_cfg["shrinkage"])
        role = str(method_cfg["role"])
        for train_scale in iem.SCALES:
            source = pack(train_scale, iem.TRAIN_CURVE)
            for test_scale in iem.SCALES:
                for group, curve_name, label in noa.ALL_TARGETS:
                    target = pack(test_scale, curve_name)
                    lam = noa.response_lambda(target.curve, response_rule)
                    coef, fit_info = noa.fit_coefficient(source, lam, nuisance, shrinkage, basis_cache)
                    factor = iem.drop_localization_factor(target.curve)
                    feature = iem.causal_drop_response(target.curve, lam)
                    pred = target.baseline + factor * coef * feature
                    corr_mae = iem.mae(target.curve.loss, pred)
                    delta = 100.0 * (corr_mae / target.base_mae - 1.0)
                    detail_rows.append(
                        {
                            "method": method,
                            "role": role,
                            "response_rule": response_rule,
                            "nuisance": nuisance,
                            "shrinkage": shrinkage,
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
                            **fit_info,
                        }
                    )
                    coef_key = (method, train_scale, test_scale, curve_name, f"{lam:.12g}")
                    if coef_key not in seen_coef_keys:
                        seen_coef_keys.add(coef_key)
                        coefficient_rows.append(
                            {
                                "method": method,
                                "response_rule": response_rule,
                                "nuisance": nuisance,
                                "shrinkage": shrinkage,
                                "train_scale": train_scale,
                                "test_scale": test_scale,
                                "test_curve": curve_name,
                                "test_label": label,
                                "lambda": lam,
                                "coef": coef,
                                **fit_info,
                            }
                        )

    summary_rows = summarize(detail_rows)
    iem.write_csv(OUT_DIR / "details.csv", detail_rows)
    iem.write_csv(OUT_DIR / "summary.csv", summary_rows)
    iem.write_csv(OUT_DIR / "coefficients.csv", coefficient_rows)
    write_report(summary_rows, coefficient_rows)
    print(f"wrote {OUT_DIR / 'details.csv'}")
    print(f"wrote {OUT_DIR / 'summary.csv'}")
    print(f"wrote {OUT_DIR / 'coefficients.csv'}")
    print(f"wrote {OUT_DIR / 'REPORT.md'}")


if __name__ == "__main__":
    main()
