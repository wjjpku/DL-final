# Reproduction Scripts

This directory contains both current entry points and historical exploration
scripts.  For the public-facing project, use the current entry points below.

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

Generated outputs:

```text
results/schedule_response_robustness/
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

## Historical Scripts

Files with names such as `current_law_*`, `step_time_*`,
`cosine_to_wsd_*`, and `mpl_ld_lag_*` are retained as development history.
They are useful for provenance and ablation archaeology, but they should not be
treated as the current deployable model unless referenced by
`schedule_response_robustness_audit.py` or the current slides.
