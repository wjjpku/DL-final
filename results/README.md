# Results Directory

This directory contains committed result artifacts.  The current main result is
under:

```text
results/schedule_response_robustness/
```

## Current Main Result

| File | Content |
|---|---|
| `schedule_response_robustness/REPORT.md` | Main robustness tables and interpretation. |
| `schedule_response_robustness/LEAKAGE_AUDIT.md` | Target-loss usage audit. |
| `schedule_response_robustness/lambda_sensitivity_*.csv` | q2 half-life and lambda sensitivity. |
| `schedule_response_robustness/kernel_ablation_*.csv` | Same-capacity feature controls. |
| `schedule_response_robustness/cross_scale_*.csv` | Cross-scale transfer audits. |
| `schedule_response_robustness/projection_ablation_*.csv` | Direct no-projection negative control. |
| `schedule_response_robustness/window_*.csv` | Source-only calibration-window rule and sensitivity. |
| `schedule_response_robustness/wsdcon_failure_slice.csv` | WSD-con final-LR failure-mode slice. |
| `schedule_response_robustness/figs/` | Figures copied into `slides/figs/`. |

## Headline Numbers

| Metric | Value |
|---|---:|
| Same-scale WSD-family mean MAE change vs MPL | `-30.88%` |
| Worst WSD-family row | `-4.67%` |
| WSD-family wins | `15/15` |
| Projected `kappa_hat` vs target oracle `kappa_star` Pearson | `+0.910` |
| No-projection negative control | `+625.92%`, `0/15` wins |
| Leave-one-scale-out mean-kappa transfer | `-25.62%`, `15/15` wins |

## Baseline Reproduction Outputs

The baseline reproduction command is:

```bash
python3 repro/reproduce_cosine_to_wsd.py --scales 25 100 400
```

It writes:

| File or directory | Content |
|---|---|
| `tables/cosine_to_wsd_metrics.csv` | Tissue/Momentum and MPL metrics on source cosine and WSD-family targets. |
| `tables/fitted_params.json` | Fitted baseline parameters by scale. |
| `figures/avg_test_mae.png` | Average WSD-family MAE comparison. |
| `figures/avg_test_rmse.png` | Average WSD-family RMSE comparison. |
| `predictions/` | Per-curve loss and prediction CSV files. |

Expected aggregate values over the 15 WSD-family test targets:

| Baseline | Cosine train MAE | WSD test MAE | WSD test RMSE | WSD test R2 |
|---|---:|---:|---:|---:|
| Tissue/Momentum | `0.002242` | `0.007493` | `0.010415` | `0.995223` |
| MPL | `0.003517` | `0.006292` | `0.009848` | `0.995760` |

## Historical Result Directories

Many other directories are retained from earlier development:

- `current_law_*`
- `step_time_*`
- `cosine_to_wsd_response_search/`
- `interpretable_*`
- `mpl_ld_*`

They are not deleted because they document the modeling path and negative
controls.  For grading or public reproduction, start with
`schedule_response_robustness/` and `../REPRODUCIBILITY.md`.
