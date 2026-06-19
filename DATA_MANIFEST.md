# Data Manifest

This repository is organized around the public-curve reproduction path used by
the current slides and reports.  The complete data required for that path is
included in Git.

## Main Included Data

Path:

```text
external/MultiPowerLaw/loss_curve_repo/
```

Scale subdirectories:

```text
csv_25/
csv_100/
csv_400/
```

Included curve files per scale:

| File | Role |
|---|---|
| `cosine_72000.csv` | Source curve for projected residual calibration. |
| `wsd_20000_24000.csv` | WSD sharp cooldown target. |
| `wsdld_20000_24000.csv` | WSD linear decay target. |
| `wsdcon_3.csv` | WSD-con target with final LR `3e-5`. |
| `wsdcon_9.csv` | WSD-con target with final LR `9e-5`. |
| `wsdcon_18.csv` | WSD-con target with final LR `18e-5`. |
| `cosine_24000.csv` | Auxiliary public cosine curve used by baseline scripts. |
| `constant_24000.csv` | Auxiliary public constant-LR curve. |
| `constant_72000.csv` | Auxiliary public constant-LR curve. |

The full public-curve directory is small enough for normal GitHub use.

## Generated Result Data

The current main generated result directory is:

```text
results/schedule_response_robustness/
```

It contains:

- CSV tables for lambda sensitivity, kernel ablation, cross-scale transfer,
  source-window selection, and WSD-con failure analysis.
- PNG figures used by the slides.
- `REPORT.md` and `LEAKAGE_AUDIT.md`.

These files can be regenerated with:

```bash
python3 repro/schedule_response_robustness_audit.py
```

## Ignored Local Or Historical Data

The release package intentionally excludes:

- raw transformer-training bytes;
- historical search outputs under `results/current_law_*`, `results/step_time_*`,
  and `results/cosine_to_wsd_response_search/`;
- paper-draft figures and independent reproduction branches;
- local scratch plotting workspaces.

These files are not needed to reproduce the slides' core experiments.  The
release verifier fails if non-release result files are still tracked.
