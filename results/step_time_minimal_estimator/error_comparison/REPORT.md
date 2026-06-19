# MPL vs Old vs Minimal Error Comparison

This reruns the residual-style plots with three error curves: `MPL`, `MPL+old`, and `MPL+minimal`.

- `MPL+old` uses the previous cumulative-LR / S-time response feature and fits its amplitude on the target residual in each panel.  It is a same-curve shape diagnostic, not a deployment protocol.
- `MPL+minimal` is the current one-kappa target-holdout rule: source and tau come from the LR schedule, and target residuals are not used.

## Aggregate MAE Change vs MPL

| group | old same-fit mean | old worst | old non-harm | minimal mean | minimal worst | minimal non-harm |
|---|---:|---:|---:|---:|---:|---:|
| core | -32.3% | -5.6% | 18/18 | -32.0% | -0.4% | 18/18 |
| extended | -21.5% | +0.0% | 27/27 | -21.4% | +0.0% | 27/27 |
| safety_controls | +0.0% | +0.0% | 9/9 | +0.0% | +0.0% | 9/9 |

## Figures

- `25M` core: `core_residuals_25M.png`
- `25M` safety controls: `safety_residuals_25M.png`
- `25M` compact overview: `error_comparison_25M.png`
- `100M` core: `core_residuals_100M.png`
- `100M` safety controls: `safety_residuals_100M.png`
- `100M` compact overview: `error_comparison_100M.png`
- `400M` core: `core_residuals_400M.png`
- `400M` safety controls: `safety_residuals_400M.png`
- `400M` compact overview: `error_comparison_400M.png`
- Core MAE bar summary: `mae_bar_summary.png`

## Reading

- On core targets, the old same-curve feature gives mean `-32.3%` and worst `-5.6%`; the minimal holdout rule gives mean `-32.0%` and worst `-0.4%`.
- On extended controls, minimal remains non-harming (`27/27`), while the old same-fit curve is only a target-residual diagnostic.
- Safety controls are intentionally unchanged by minimal (`9/9` non-harm), which is the desired behavior for short-smooth and zero-drop schedules.
