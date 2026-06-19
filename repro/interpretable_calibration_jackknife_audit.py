#!/usr/bin/env python3
"""Calibration jackknife audit for observation-bracket MPL-LD.

The current residual model fits only one new scalar, kappa_hat_s, from the
source cosine residual.  This audit checks whether that scalar is stable under
source-suffix resampling.  Each subset is used only to estimate kappa; target
losses remain evaluation-only.
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
import interpretable_observation_bracket_audit as oba  # noqa: E402

OUT_DIR = iem.ROOT / "results" / "interpretable_calibration_jackknife_audit"

SUBSETS = [
    "full",
    "even_index",
    "odd_index",
    "first_half",
    "second_half",
    "leave_block_0",
    "leave_block_1",
    "leave_block_2",
    "leave_block_3",
    "leave_block_4",
]


def subset_mask(n: int, subset: str) -> np.ndarray:
    idx = np.arange(n)
    if subset == "full":
        return np.ones(n, dtype=bool)
    if subset == "even_index":
        return idx % 2 == 0
    if subset == "odd_index":
        return idx % 2 == 1
    if subset == "first_half":
        return idx < n // 2
    if subset == "second_half":
        return idx >= n // 2
    if subset.startswith("leave_block_"):
        block = int(subset.rsplit("_", 1)[1])
        blocks = np.floor(idx * 5 / max(n, 1)).astype(int)
        return blocks != block
    raise ValueError(f"unknown subset: {subset}")


def fit_subset_coefficient(
    source: iem.CurvePack,
    lam: float,
    fit_start: int,
    subset: str,
    basis_cache: dict[tuple[str, str, int], np.ndarray],
) -> tuple[float, dict[str, float]]:
    phi, phi_o, residual_o, basis_dim = oba.residualized_pair(
        source,
        lam,
        "mpl_ld4",
        fit_start,
        basis_cache,
    )
    mask = subset_mask(len(phi_o), subset)
    phi_sub = phi[mask]
    x_sub = phi_o[mask]
    y_sub = residual_o[mask]
    n_cal = int(np.sum(mask))
    dot = max(0.0, float(np.dot(x_sub, y_sub)))
    full_norm = float(np.linalg.norm(phi_sub))
    perp_norm = float(np.linalg.norm(x_sub))
    perp_energy = perp_norm * perp_norm
    ridge = 1.0 / max(n_cal, 1)
    denom = perp_energy + ridge
    return dot / max(denom, 1e-18), {
        "basis_dim": basis_dim,
        "n_cal": n_cal,
        "ridge": ridge,
        "source_dot": dot,
        "source_full_norm": full_norm,
        "source_perp_norm": perp_norm,
        "source_retention": float(perp_energy / max(full_norm * full_norm, 1e-18)),
        "denominator": float(denom),
    }


def load_pack_cached(
    cache: dict[tuple[str, str], iem.CurvePack],
    scale: str,
    curve_name: str,
) -> iem.CurvePack:
    key = (scale, curve_name)
    if key not in cache:
        cache[key] = oba.load_pack(scale, curve_name)
    return cache[key]


def run_audit() -> tuple[list[dict[str, object]], list[dict[str, object]], int]:
    cache: dict[tuple[str, str], iem.CurvePack] = {}
    basis_cache: dict[tuple[str, str, int], np.ndarray] = {}
    rule_rows = oba.fit_start_rule_rows(cache, basis_cache)
    selected_fit_start = oba.select_fit_start(rule_rows)

    detail_rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []
    coef_cache: dict[tuple[str, str, str, float], tuple[float, dict[str, float]]] = {}

    for subset in SUBSETS:
        for train_scale in iem.SCALES:
            source = load_pack_cached(cache, train_scale, iem.TRAIN_CURVE)
            for test_scale in iem.SCALES:
                for group, curve_name, label in noa.ALL_TARGETS:
                    target = load_pack_cached(cache, test_scale, curve_name)
                    lam = oba.response_lambda(target.curve, "observation_bracket")
                    coef_key = (subset, train_scale, curve_name, float(lam))
                    if coef_key not in coef_cache:
                        coef_cache[coef_key] = fit_subset_coefficient(
                            source,
                            lam,
                            selected_fit_start,
                            subset,
                            basis_cache,
                        )
                    coef, info = coef_cache[coef_key]
                    factor = oba.locality_factor(target.curve, "linear")
                    pred = target.baseline + factor * coef * iem.causal_drop_response(target.curve, lam)
                    corr_mae = iem.mae(target.curve.loss, pred)
                    delta = 100.0 * (corr_mae / target.base_mae - 1.0)
                    detail_rows.append(
                        {
                            "subset": subset,
                            "fit_start": selected_fit_start,
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
                            **info,
                        }
                    )

    for (subset, train_scale, curve_name, lam), (coef, info) in coef_cache.items():
        coefficient_rows.append(
            {
                "subset": subset,
                "train_scale": train_scale,
                "test_curve": curve_name,
                "lambda": lam,
                "coef": coef,
                **info,
            }
        )
    return detail_rows, coefficient_rows, selected_fit_start


def aggregate(rows: list[dict[str, object]], subset: str, group: str, split: str) -> dict[str, object]:
    deltas = np.array([float(row["delta_pct"]) for row in rows], dtype=np.float64)
    return {
        "subset": subset,
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
    summary_rows: list[dict[str, object]] = []
    for subset in SUBSETS:
        rows_s = [row for row in detail_rows if row["subset"] == subset]
        for group in ["core_wsd", "extra_control"]:
            rows_g = [row for row in rows_s if row["group"] == group]
            splits = {
                "same_scale": [row for row in rows_g if row["train_scale"] == row["test_scale"]],
                "cross_scale": [row for row in rows_g if row["train_scale"] != row["test_scale"]],
                "all": rows_g,
            }
            for split, rows in splits.items():
                if rows:
                    summary_rows.append(aggregate(rows, subset, group, split))
    return summary_rows


def coefficient_summary(coefficient_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    keys = sorted(
        {
            (str(row["train_scale"]), str(row["test_curve"]), float(row["lambda"]))
            for row in coefficient_rows
        }
    )
    for train_scale, curve_name, lam in keys:
        sub = [
            row
            for row in coefficient_rows
            if row["train_scale"] == train_scale
            and row["test_curve"] == curve_name
            and abs(float(row["lambda"]) - lam) < 1e-12
        ]
        coefs = np.array([float(row["coef"]) for row in sub], dtype=np.float64)
        full = next(row for row in sub if row["subset"] == "full")
        out.append(
            {
                "train_scale": train_scale,
                "test_curve": curve_name,
                "lambda": lam,
                "rows": len(sub),
                "full_coef": float(full["coef"]),
                "mean_coef": float(np.mean(coefs)),
                "std_coef": float(np.std(coefs)),
                "min_coef": float(np.min(coefs)),
                "max_coef": float(np.max(coefs)),
                "relative_range": float((np.max(coefs) - np.min(coefs)) / max(abs(float(full["coef"])), 1e-18)),
            }
        )
    return out


def find(summary_rows: list[dict[str, object]], subset: str, group: str, split: str) -> dict[str, object]:
    for row in summary_rows:
        if row["subset"] == subset and row["group"] == group and row["split"] == split:
            return row
    raise KeyError((subset, group, split))


def fmt(row: dict[str, object]) -> str:
    return f"{float(row['mean_delta']):+.2f}% / {float(row['worst_delta']):+.2f}% / {int(row['wins'])}/{int(row['rows'])}"


def write_report(
    summary_rows: list[dict[str, object]],
    coeff_rows: list[dict[str, object]],
    selected_fit_start: int,
) -> None:
    lines = [
        "# Calibration Jackknife Audit\n\n",
        "This audit checks whether the only residual-fitted scalar, `kappa_hat_s`, depends on a fragile subset of the cosine calibration suffix.  Each subset refits kappa from source `cosine_72000.csv`; WSD and control losses remain evaluation-only.\n\n",
        f"Selected fit start from the source-only rule: `{selected_fit_start}`.\n\n",
        "## Summary\n\n",
        "| subset | split | group | mean / worst / wins | non-harm |\n",
        "|---|---|---|---:|---:|\n",
    ]
    for subset in SUBSETS:
        for split in ["same_scale", "cross_scale"]:
            for group in ["core_wsd", "extra_control"]:
                row = find(summary_rows, subset, group, split)
                lines.append(
                    f"| {subset} | {split} | {group} | {fmt(row)} | "
                    f"{int(row['nonharm'])}/{int(row['rows'])} |\n"
                )

    coeff_rel = np.array([float(row["relative_range"]) for row in coeff_rows], dtype=np.float64)
    lines += [
        "\n## Coefficient Stability\n\n",
        f"- Median relative coefficient range across subset fits: `{float(np.median(coeff_rel)):.3f}`.\n",
        f"- Worst relative coefficient range across subset fits: `{float(np.max(coeff_rel)):.3f}`.\n\n",
        "## Reading\n\n",
        "- Odd/even and leave-one-block fits should remain close to the full-suffix result if kappa is not driven by isolated points.\n",
        "- First-half and second-half fits are a stricter stress test because they change the calibration time region.  Failures here would indicate residual nonstationarity inside the suffix.\n",
        "- The desired result is not identical coefficients; it is all-win WSD transfer and non-harm controls under major source-suffix resampling.\n",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    detail_rows, coefficient_rows, selected_fit_start = run_audit()
    summary_rows = summarize(detail_rows)
    coeff_summary_rows = coefficient_summary(coefficient_rows)
    iem.write_csv(OUT_DIR / "details.csv", detail_rows)
    iem.write_csv(OUT_DIR / "summary.csv", summary_rows)
    iem.write_csv(OUT_DIR / "coefficients.csv", coefficient_rows)
    iem.write_csv(OUT_DIR / "coefficient_summary.csv", coeff_summary_rows)
    write_report(summary_rows, coeff_summary_rows, selected_fit_start)
    print(f"wrote {OUT_DIR / 'details.csv'}")
    print(f"wrote {OUT_DIR / 'summary.csv'}")
    print(f"wrote {OUT_DIR / 'coefficients.csv'}")
    print(f"wrote {OUT_DIR / 'coefficient_summary.csv'}")
    print(f"wrote {OUT_DIR / 'REPORT.md'}")


if __name__ == "__main__":
    main()
