# Reproduction Scripts

This directory contains only the scripts needed for the public-facing
reproduction path.  Historical exploration scripts are intentionally excluded
from the GitHub release.

## Current Entry Points

| Script | Purpose |
|---|---|
| `schedule_response_robustness_audit.py` | Main projected-kappa audit. Regenerates robustness reports, CSVs, and slide figures. |
| `reproduce_cosine_to_wsd.py` | CPU-friendly MPL/Tissue baseline reproduction on public curves. |
| `interpretable_error_model.py` | Shared curve loading and frozen MPL residual utilities used by the main audit. |
| `interpretable_nuisance_origin_audit.py` | Shared target/source definitions and curve-pack loading. |
| `interpretable_observation_bracket_audit.py` | MPL-LD tangent basis and projection utilities. |
| `verify_release.py` | Lightweight release gate for required files, data, PDFs, headline numbers, and large-file risks. |

## Main Command

```bash
python3 repro/schedule_response_robustness_audit.py
```

Committed summary outputs and regenerated slide figures:

```text
results/schedule_response_robustness/REPORT.md
results/schedule_response_robustness/LEAKAGE_AUDIT.md
results/schedule_response_robustness/*_summary.csv
results/schedule_response_robustness/window_rule.csv
results/schedule_response_robustness/wsdcon_failure_slice.csv
slides/figs/fig_mpl_residual_anomaly_100M.png
slides/figs/fig_projection_decomposition_cosine_100M.png
slides/figs/fig_projection_ablation_time_errors_100M.png
slides/figs/fig_schedule_response_mae_heatmap.png
slides/figs/fig_schedule_response_time_errors_100M.png
slides/figs/fig_kappa_clean_scatter.png
```

## Baseline Command

```bash
python3 repro/reproduce_cosine_to_wsd.py --scales 25 100 400
```

This verifies the public-curve baseline fitting path.  It is not the main
project claim; the main claim is the residual-identification correction on top
of frozen MPL.

Expected 15-target WSD-family test aggregates:

| Baseline | MAE | RMSE | R2 |
|---|---:|---:|---:|
| Tissue/Momentum | `0.007493` | `0.010415` | `0.995223` |
| MPL | `0.006292` | `0.009848` | `0.995760` |

## Release Boundary

Only the scripts listed above should be committed in `repro/`.  The release
verifier fails if old exploration scripts remain tracked.
