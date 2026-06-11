# Next-Gen Target-Loss Blindness Audit

This audit freezes train-side `kappa_transfer` from the deployment audit and then replaces every target loss curve with a deterministic fake loss. The target-retention gate is recomputed from the target schedule feature. If target loss is not used for deployment, `R_target`, the gate, and `kappa_safe` must remain unchanged.

## Max Absolute Differences

| quantity | max abs diff |
|---|---:|
| `retention_abs_diff` | `0.000e+00` |
| `deployment_retention_abs_diff` | `0.000e+00` |
| `factor_abs_diff` | `0.000e+00` |
| `deployment_factor_abs_diff` | `0.000e+00` |
| `kappa_safe_abs_diff` | `0.000e+00` |
| `deployment_kappa_abs_diff` | `0.000e+00` |

## Readout

Across `1116` audited rows, replacing target losses changes max target retention by `0.000e+00` and max `kappa_safe` by `0.000e+00`. The deployment gate is therefore target-loss blind: target loss is used only for evaluation, while deployment uses training residuals plus target schedule features.
