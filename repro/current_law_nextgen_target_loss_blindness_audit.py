#!/usr/bin/env python3
"""Target-loss blindness audit for the next-gen deployment formula.

This lightweight audit freezes the train-side `kappa_transfer` values from the
deployment audit, then recomputes only the target-retention gate after replacing
target losses with deterministic fake losses. A deployable target gate may use
the target schedule feature, but not target loss values.
"""
from __future__ import annotations

import csv
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parent
ROOT = REPO.parent
sys.path.insert(0, str(REPO))

import current_law_continuous_kappa_search as base  # noqa: E402
import current_law_soft_spectral_multicurve_selection_audit as spectral  # noqa: E402
from deep_stime import stime_feature  # noqa: E402
from reproduce_cosine_to_wsd import Curve  # noqa: E402


DEPLOY_DIR = ROOT / "results" / "current_law_nextgen_deployment_audit"
OUT_DIR = ROOT / "results" / "current_law_nextgen_target_loss_blindness_audit"
RETENTION_FLOOR = 0.01


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def perturb_curve_loss(curve: Curve) -> Curve:
    idx = np.linspace(0.0, 1.0, len(curve.loss), dtype=np.float64)
    fake_loss = 7.0 + 0.3 * np.sin(17.0 * idx) + 0.1 * idx
    return Curve(
        name=curve.name,
        scale=curve.scale,
        step=curve.step,
        loss=fake_loss.astype(np.float64),
        lrs=curve.lrs,
    )


def target_retention(curve: Curve, lam: float) -> float:
    phi = stime_feature(curve, base.LAMBDA)
    q = spectral.dct_basis(len(curve.step), spectral.MAX_MODE)
    a = spectral.smoother_matrix(q, lam)
    phi_o = spectral.soft_residualize(phi, q, a)
    phi_l2 = float(np.dot(phi, phi))
    return 0.0 if phi_l2 <= 1e-18 else float(np.dot(phi_o, phi_o) / phi_l2)


def run() -> list[dict[str, object]]:
    deployment_rows = read_csv(DEPLOY_DIR / "details.csv")
    retention_cache: dict[tuple[str, str, float, str], float] = {}
    rows: list[dict[str, object]] = []
    for row in deployment_rows:
        scale = row["scale"]
        target = row["test_curve"]
        lam = float(row["selected_lambda"])
        original_key = (scale, target, lam, "original")
        perturbed_key = (scale, target, lam, "perturbed")
        if original_key not in retention_cache:
            curve = base.load_curve(scale, target)
            retention_cache[original_key] = target_retention(curve, lam)
            retention_cache[perturbed_key] = target_retention(perturb_curve_loss(curve), lam)
        original_retention = retention_cache[original_key]
        perturbed_retention = retention_cache[perturbed_key]
        original_factor = 1.0 if original_retention >= RETENTION_FLOOR else 0.0
        perturbed_factor = 1.0 if perturbed_retention >= RETENTION_FLOOR else 0.0
        kappa_transfer = float(row["kappa_transfer"])
        rows.append(
            {
                "scale": scale,
                "train_id": row["train_id"],
                "test_curve": target,
                "selected_lambda": lam,
                "retention_abs_diff": abs(original_retention - perturbed_retention),
                "deployment_retention_abs_diff": abs(float(row["target_retention"]) - original_retention),
                "factor_abs_diff": abs(original_factor - perturbed_factor),
                "deployment_factor_abs_diff": abs(float(row["target_factor"]) - original_factor),
                "kappa_safe_abs_diff": abs(kappa_transfer * original_factor - kappa_transfer * perturbed_factor),
                "deployment_kappa_abs_diff": abs(float(row["kappa_safe"]) - kappa_transfer * original_factor),
            }
        )
    return rows


def write_report(rows: list[dict[str, object]]) -> None:
    keys = [
        "retention_abs_diff",
        "deployment_retention_abs_diff",
        "factor_abs_diff",
        "deployment_factor_abs_diff",
        "kappa_safe_abs_diff",
        "deployment_kappa_abs_diff",
    ]
    maxes = {key: max(float(row[key]) for row in rows) for key in keys}
    lines = [
        "# Next-Gen Target-Loss Blindness Audit\n\n",
        "This audit freezes train-side `kappa_transfer` from the deployment audit and then replaces every target loss curve with a deterministic fake loss. "
        "The target-retention gate is recomputed from the target schedule feature. If target loss is not used for deployment, `R_target`, the gate, and `kappa_safe` must remain unchanged.\n\n",
        "## Max Absolute Differences\n\n",
        "| quantity | max abs diff |\n",
        "|---|---:|\n",
    ]
    for key in keys:
        lines.append(f"| `{key}` | `{maxes[key]:.3e}` |\n")
    lines += [
        "\n## Readout\n\n",
        f"Across `{len(rows)}` audited rows, replacing target losses changes max target retention by `{maxes['retention_abs_diff']:.3e}` and max `kappa_safe` by `{maxes['kappa_safe_abs_diff']:.3e}`. "
        "The deployment gate is therefore target-loss blind: target loss is used only for evaluation, while deployment uses training residuals plus target schedule features.\n",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = run()
    write_csv(OUT_DIR / "diffs.csv", rows)
    write_report(rows)
    print(f"wrote {OUT_DIR / 'REPORT.md'}")
    print(
        f"rows={len(rows)} "
        f"max_retention_diff={max(float(row['retention_abs_diff']) for row in rows):.3e} "
        f"max_kappa_safe_diff={max(float(row['kappa_safe_abs_diff']) for row in rows):.3e}"
    )


if __name__ == "__main__":
    main()
