#!/usr/bin/env python3
"""Calibration search for the current S-time DropRelaxS law.

This script deliberately does not introduce new residual features.  It keeps the
current law

    L_pred(t) = L_MPL(t) + kappa * DropRelaxS_lambda(t)

and searches only how to calibrate lambda and kappa from non-target curves.  The
main final check is still cosine-fit MPL evaluated on the sharp WSD curves.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
ROOT = REPO.parent
sys.path.insert(0, str(REPO))
from reproduce_cosine_to_wsd import (  # noqa: E402
    SCALES,
    PEAK_LR,
    load_curve,
    metrics,
    mpl_predict,
    MPL_PRECOMPUTED_INIT,
)
from deep_stime import stime_feature  # noqa: E402
from nonadiabatic_theory import estimate_dLeq_deta, fit_origin  # noqa: E402


TARGET = ["wsd_20000_24000.csv", "wsdld_20000_24000.csv"]
PROBES = ["wsdcon_3.csv", "wsdcon_9.csv", "wsdcon_18.csv"]
ALL_NONCOS = TARGET + PROBES
LAM_GRID = [2, 3, 5, 7, 10, 14, 20, 30, 50]


def residual(scale: str, curve_name: str) -> tuple[np.ndarray, np.ndarray]:
    c = load_curve(scale, curve_name)
    p = MPL_PRECOMPUTED_INIT[scale]
    return c.loss - mpl_predict(p, c), c.step


def fit_kappa(scale: str, curve_names: list[str], lam: float) -> float:
    xs, ys = [], []
    p = MPL_PRECOMPUTED_INIT[scale]
    for name in curve_names:
        c = load_curve(scale, name)
        xs.append(stime_feature(c, lam))
        ys.append(c.loss - mpl_predict(p, c))
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    if float(np.dot(x, x)) <= 1e-18:
        return 0.0
    return max(0.0, fit_origin(x, y)[0])


def score(scale: str, curve_names: list[str], lam: float, kappa: float) -> tuple[float, int, int]:
    maes_base, maes_corr = [], []
    p = MPL_PRECOMPUTED_INIT[scale]
    for name in curve_names:
        c = load_curve(scale, name)
        base = mpl_predict(p, c)
        pred = base + kappa * stime_feature(c, lam)
        maes_base.append(metrics(c.loss, base)["mae"])
        maes_corr.append(metrics(c.loss, pred)["mae"])
    mb = float(np.mean(maes_base))
    mc = float(np.mean(maes_corr))
    return mc, int(np.sum(np.array(maes_corr) < np.array(maes_base))), len(curve_names)


def calibrate_from_probe_residuals(train_names: list[str]) -> list[dict[str, object]]:
    rows = []
    for lam in LAM_GRID:
        base_all, corr_all, wins, tests = [], [], 0, 0
        kappas = {}
        for scale in SCALES:
            kappa = fit_kappa(scale, train_names, lam)
            kappas[scale] = kappa
            p = MPL_PRECOMPUTED_INIT[scale]
            for name in TARGET:
                c = load_curve(scale, name)
                base = mpl_predict(p, c)
                pred = base + kappa * stime_feature(c, lam)
                base_mae = metrics(c.loss, base)["mae"]
                corr_mae = metrics(c.loss, pred)["mae"]
                base_all.append(base_mae)
                corr_all.append(corr_mae)
                wins += int(corr_mae < base_mae)
                tests += 1
        rows.append(
            {
                "protocol": "fit_kappa_on_" + "+".join(x.replace(".csv", "") for x in train_names),
                "lambda": lam,
                "base_mae": float(np.mean(base_all)),
                "corr_mae": float(np.mean(corr_all)),
                "delta_pct": 100.0 * (float(np.mean(corr_all)) / float(np.mean(base_all)) - 1.0),
                "wins": wins,
                "tests": tests,
                "kappa_25": kappas["25"],
                "kappa_100": kappas["100"],
                "kappa_400": kappas["400"],
            }
        )
    return rows


def calibrate_cross_scale_c() -> list[dict[str, object]]:
    rows = []
    for lam in LAM_GRID:
        ratio = {}
        kfit = {}
        for scale in SCALES:
            kfit[scale] = fit_kappa(scale, TARGET, lam)
            kpred = estimate_dLeq_deta(scale)[0] * PEAK_LR
            ratio[scale] = kfit[scale] / kpred

        base_all, corr_all, wins, tests = [], [], 0, 0
        kappas = {}
        for tgt in SCALES:
            c_loo = float(np.mean([ratio[s] for s in SCALES if s != tgt]))
            kappa = c_loo * estimate_dLeq_deta(tgt)[0] * PEAK_LR
            kappas[tgt] = kappa
            p = MPL_PRECOMPUTED_INIT[tgt]
            for name in TARGET:
                c = load_curve(tgt, name)
                base = mpl_predict(p, c)
                pred = base + kappa * stime_feature(c, lam)
                base_mae = metrics(c.loss, base)["mae"]
                corr_mae = metrics(c.loss, pred)["mae"]
                base_all.append(base_mae)
                corr_all.append(corr_mae)
                wins += int(corr_mae < base_mae)
                tests += 1
        rows.append(
            {
                "protocol": "cross_scale_c_from_target_decays_LOO",
                "lambda": lam,
                "base_mae": float(np.mean(base_all)),
                "corr_mae": float(np.mean(corr_all)),
                "delta_pct": 100.0 * (float(np.mean(corr_all)) / float(np.mean(base_all)) - 1.0),
                "wins": wins,
                "tests": tests,
                "kappa_25": kappas["25"],
                "kappa_100": kappas["100"],
                "kappa_400": kappas["400"],
            }
        )
    return rows


def leave_one_noncos() -> list[dict[str, object]]:
    rows = []
    for hold in ALL_NONCOS:
        train = [x for x in ALL_NONCOS if x != hold]
        for lam in [5, 10, 20]:
            base_all, corr_all, wins, tests = [], [], 0, 0
            for scale in SCALES:
                kappa = fit_kappa(scale, train, lam)
                p = MPL_PRECOMPUTED_INIT[scale]
                c = load_curve(scale, hold)
                base = mpl_predict(p, c)
                pred = base + kappa * stime_feature(c, lam)
                base_mae = metrics(c.loss, base)["mae"]
                corr_mae = metrics(c.loss, pred)["mae"]
                base_all.append(base_mae)
                corr_all.append(corr_mae)
                wins += int(corr_mae < base_mae)
                tests += 1
            rows.append(
                {
                    "protocol": "leave_one_noncos_hold_" + hold.replace(".csv", ""),
                    "lambda": lam,
                    "base_mae": float(np.mean(base_all)),
                    "corr_mae": float(np.mean(corr_all)),
                    "delta_pct": 100.0 * (float(np.mean(corr_all)) / float(np.mean(base_all)) - 1.0),
                    "wins": wins,
                    "tests": tests,
                    "kappa_25": np.nan,
                    "kappa_100": np.nan,
                    "kappa_400": np.nan,
                }
            )
    return rows


def main() -> None:
    out_dir = ROOT / "results" / "current_law_calibration"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    rows += calibrate_from_probe_residuals(["wsdcon_9.csv"])
    rows += calibrate_from_probe_residuals(PROBES)
    rows += calibrate_cross_scale_c()
    rows += leave_one_noncos()

    csv_path = out_dir / "summary.csv"
    fieldnames = [
        "protocol",
        "lambda",
        "base_mae",
        "corr_mae",
        "delta_pct",
        "wins",
        "tests",
        "kappa_25",
        "kappa_100",
        "kappa_400",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    final_rows = [
        r for r in rows
        if r["protocol"].startswith("fit_kappa") or r["protocol"].startswith("cross_scale")
    ]
    final_rows.sort(key=lambda r: (r["corr_mae"], abs(float(r["lambda"]) - 10.0)))

    report_path = out_dir / "REPORT.md"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# Current-Law Calibration Search\n\n")
        f.write("Law is fixed: `MPL + kappa * DropRelaxS_lambda`. Only calibration changes.\n\n")
        f.write("Final target is still cosine-fit MPL evaluated on `wsd` and `wsdld`.\n\n")
        f.write("## Best final-target protocols\n\n")
        f.write("| rank | protocol | lambda | MAE | vs MPL | wins | kappa 25/100/400 |\n")
        f.write("|---:|---|---:|---:|---:|---:|---|\n")
        for i, r in enumerate(final_rows[:12], 1):
            f.write(
                f"| {i} | `{r['protocol']}` | {r['lambda']} | {r['corr_mae']:.5f} | "
                f"{r['delta_pct']:+.1f}% | {r['wins']}/{r['tests']} | "
                f"{r['kappa_25']:.4g}, {r['kappa_100']:.4g}, {r['kappa_400']:.4g} |\n"
            )
        f.write("\n## Leave-one-noncos sanity check\n\n")
        f.write("These rows ask whether the same law predicts arbitrary held-out non-cosine curves.\n\n")
        f.write("| held-out | lambda | MAE | vs MPL | wins |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        loo = [r for r in rows if r["protocol"].startswith("leave_one_noncos")]
        for r in loo:
            hold = r["protocol"].replace("leave_one_noncos_hold_", "")
            f.write(
                f"| `{hold}` | {r['lambda']} | {r['corr_mae']:.5f} | "
                f"{r['delta_pct']:+.1f}% | {r['wins']}/{r['tests']} |\n"
            )

    print(f"wrote {csv_path}")
    print(f"wrote {report_path}")
    print("\nBest final-target protocols:")
    for r in final_rows[:8]:
        print(
            f"  {r['protocol']:44s} lam={r['lambda']:>4} "
            f"MAE={r['corr_mae']:.5f} delta={r['delta_pct']:+.1f}% "
            f"wins={r['wins']}/{r['tests']}"
        )


if __name__ == "__main__":
    main()
