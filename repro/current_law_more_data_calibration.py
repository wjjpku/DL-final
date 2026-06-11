#!/usr/bin/env python3
"""Use more calibration data while keeping the current DropRelaxS law fixed.

The previous audit showed that probe-only calibration helps the final
cosine->WSD target, but only modestly.  This script asks whether using more
non-target calibration data stabilizes the same law:

  * all wsdcon probes;
  * all wsdcon probes plus the other sharp-decay curve;
  * global lambda selected from all scales, with per-scale kappa.

No new residual feature is introduced.
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
    load_curve,
    metrics,
    mpl_predict,
    MPL_PRECOMPUTED_INIT,
)
from deep_stime import stime_feature  # noqa: E402
from nonadiabatic_theory import fit_origin  # noqa: E402


TARGETS = ["wsd_20000_24000.csv", "wsdld_20000_24000.csv"]
PROBES = ["wsdcon_3.csv", "wsdcon_9.csv", "wsdcon_18.csv"]
LAM_GRID = [2, 3, 5, 7, 10, 14, 20, 30, 50, 80]


def feature_cache():
    cache = {}
    for scale in SCALES:
        for name in TARGETS + PROBES:
            curve = load_curve(scale, name)
            for lam in LAM_GRID:
                cache[(scale, name, float(lam))] = stime_feature(curve, float(lam))
    return cache


def fit_kappa(scale: str, train: list[str], lam: float, feats) -> float:
    p = MPL_PRECOMPUTED_INIT[scale]
    xs, ys = [], []
    for name in train:
        curve = load_curve(scale, name)
        xs.append(feats[(scale, name, float(lam))])
        ys.append(curve.loss - mpl_predict(p, curve))
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    if float(np.dot(x, x)) <= 1e-18:
        return 0.0
    return max(0.0, fit_origin(x, y)[0])


def score_curve(scale: str, target: str, lam: float, kappa: float, feats) -> tuple[float, float, int]:
    p = MPL_PRECOMPUTED_INIT[scale]
    curve = load_curve(scale, target)
    base = mpl_predict(p, curve)
    pred = base + kappa * feats[(scale, target, float(lam))]
    mb = metrics(curve.loss, base)["mae"]
    mc = metrics(curve.loss, pred)["mae"]
    return mb, mc, int(mc < mb)


def train_mae_pooled(train_by_target: dict[str, list[str]], lam: float, feats) -> float:
    """Pooled calibration objective.  Kappa is per-scale; lambda is shared."""
    maes = []
    for target, train in train_by_target.items():
        del target
        for scale in SCALES:
            kappa = fit_kappa(scale, train, lam, feats)
            p = MPL_PRECOMPUTED_INIT[scale]
            for name in train:
                curve = load_curve(scale, name)
                base = mpl_predict(p, curve)
                pred = base + kappa * feats[(scale, name, float(lam))]
                maes.append(metrics(curve.loss, pred)["mae"])
    return float(np.mean(maes))


def evaluate_protocol(name: str, train_for_target: dict[str, list[str]], lam_mode: str, feats) -> dict[str, object]:
    if lam_mode == "fixed10":
        lam = 10.0
    elif lam_mode == "global_train":
        lam = min(LAM_GRID, key=lambda x: train_mae_pooled(train_for_target, float(x), feats))
        lam = float(lam)
    else:
        raise ValueError(lam_mode)

    base_all, corr_all, wins, rows = [], [], 0, []
    for target, train in train_for_target.items():
        for scale in SCALES:
            kappa = fit_kappa(scale, train, lam, feats)
            mb, mc, win = score_curve(scale, target, lam, kappa, feats)
            base_all.append(mb)
            corr_all.append(mc)
            wins += win
            rows.append((scale, target, train, kappa, mb, mc, win))

    base = float(np.mean(base_all))
    corr = float(np.mean(corr_all))
    return {
        "protocol": name,
        "lambda_mode": lam_mode,
        "lambda": lam,
        "base_mae": base,
        "corr_mae": corr,
        "delta_pct": 100.0 * (corr / base - 1.0),
        "wins": wins,
        "tests": len(corr_all),
        "details": rows,
    }


def main() -> None:
    feats = feature_cache()
    protocols: list[tuple[str, dict[str, list[str]]]] = []

    protocols.append((
        "all_wsdcon_probes_to_both_sharp",
        {target: list(PROBES) for target in TARGETS},
    ))

    protocols.append((
        "all_wsdcon_plus_other_sharp_leave_one_sharp",
        {
            "wsd_20000_24000.csv": PROBES + ["wsdld_20000_24000.csv"],
            "wsdld_20000_24000.csv": PROBES + ["wsd_20000_24000.csv"],
        },
    ))

    protocols.append((
        "other_sharp_only_leave_one_sharp",
        {
            "wsd_20000_24000.csv": ["wsdld_20000_24000.csv"],
            "wsdld_20000_24000.csv": ["wsd_20000_24000.csv"],
        },
    ))

    rows = []
    for name, train_for_target in protocols:
        for lam_mode in ["fixed10", "global_train"]:
            rows.append(evaluate_protocol(name, train_for_target, lam_mode, feats))

    out_dir = ROOT / "results" / "current_law_more_data_calibration"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["protocol", "lambda_mode", "lambda", "base_mae", "corr_mae", "delta_pct", "wins", "tests"])
        for r in rows:
            writer.writerow([
                r["protocol"],
                r["lambda_mode"],
                r["lambda"],
                r["base_mae"],
                r["corr_mae"],
                r["delta_pct"],
                r["wins"],
                r["tests"],
            ])

    detail_path = out_dir / "details.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["protocol", "lambda_mode", "lambda", "scale", "target", "train", "kappa", "base_mae", "corr_mae", "win"])
        for r in rows:
            for scale, target, train, kappa, mb, mc, win in r["details"]:
                writer.writerow([
                    r["protocol"],
                    r["lambda_mode"],
                    r["lambda"],
                    scale,
                    target,
                    "+".join(x.replace(".csv", "") for x in train),
                    kappa,
                    mb,
                    mc,
                    win,
                ])

    report_path = out_dir / "REPORT.md"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# Current-Law More-Data Calibration\n\n")
        f.write("Fixed law: `MPL + kappa * DropRelaxS_lambda`. More calibration data, no new feature.\n\n")
        f.write("| protocol | lambda mode | lambda | MAE | vs MPL | wins |\n")
        f.write("|---|---|---:|---:|---:|---:|\n")
        for r in sorted(rows, key=lambda x: float(x["corr_mae"])):
            f.write(
                f"| `{r['protocol']}` | `{r['lambda_mode']}` | {r['lambda']:.0f} | "
                f"{r['corr_mae']:.5f} | {r['delta_pct']:+.1f}% | {r['wins']}/{r['tests']} |\n"
            )
        f.write("\n## Interpretation\n\n")
        f.write("- More data is not monotone. The strongest held-out sharp-decay prediction comes from "
                "the opposite sharp-decay shape alone; adding heterogeneous `wsdcon` probes dilutes the amplitude.\n")
        f.write("- Probe-only calibration is leakage-safe for the final `cosine -> wsd/wsdld` story, "
                "but its gain is smaller.\n")
        f.write("- Selecting lambda from the larger calibration set often prefers a smaller lambda than "
                "pure probe fitting; fixed lambda=10 remains a conservative theory-first setting.\n")

    print(f"wrote {summary_path}")
    print(f"wrote {detail_path}")
    print(f"wrote {report_path}")
    for r in sorted(rows, key=lambda x: float(x["corr_mae"])):
        print(
            f"{r['protocol']:45s} {r['lambda_mode']:12s} "
            f"lam={r['lambda']:>4.0f} MAE={r['corr_mae']:.5f} "
            f"delta={r['delta_pct']:+.1f}% wins={r['wins']}/{r['tests']}"
        )


if __name__ == "__main__":
    main()
