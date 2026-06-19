# Geometry Tau Residual Error Comparison

This compares two target-holdout one-kappa corrections: the previous discrete route-tau table and the schedule-geometry tau formula.  Neither curve uses target residuals.

## Aggregate MAE Change vs MPL

| group | table mean | table worst | table non-harm | geometry mean | geometry worst | geometry non-harm | geometry beats table |
|---|---:|---:|---:|---:|---:|---:|---:|
| core | -32.0% | -0.4% | 18/18 | -32.3% | -1.5% | 18/18 | 7/18 |
| extended | -21.4% | +0.0% | 27/27 | -21.5% | +0.0% | 27/27 | 7/27 |
| safety_controls | +0.0% | +0.0% | 9/9 | +0.0% | +0.0% | 9/9 | 0/9 |

## Figures

- `25M` core: `core_residuals_25M.png`
- `25M` safety controls: `safety_residuals_25M.png`
- `100M` core: `core_residuals_100M.png`
- `100M` safety controls: `safety_residuals_100M.png`
- `400M` core: `core_residuals_400M.png`
- `400M` safety controls: `safety_residuals_400M.png`
- Core MAE bar summary: `mae_bar_summary.png`

## Reading

- Core target-holdout changes from table mean `-32.0%` / worst `-0.4%` to geometry mean `-32.3%` / worst `-1.5%`.
- Extended safety remains non-harming under geometry tau: `27/27` rows, with safety controls `9/9`.
- The plots show that geometry tau leaves most residual shapes unchanged, but tightens the weak and medium single-step corrections enough to improve the no-nuisance worst case.
