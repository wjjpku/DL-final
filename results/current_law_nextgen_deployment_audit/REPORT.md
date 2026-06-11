# Next-Gen Deployment Estimator Audit

This audit runs the reusable `NextGenKappaEstimator` end to end on the same all-train-size main-plus-extra matrix. It verifies that the deployed formula reproduces the established rho-margin audit while not relying on report-specific glue code.

## Summary

| group | mean delta | worst delta | non-harm cells | wins | target factor | mean kappa_safe |
|---|---:|---:|---:|---:|---:|---:|
| main_matrix | -11.8% | -1.0% | 558/558 | 558/558 | 1.000 | 0.01664 |
| extra_holdout | +0.0% | +0.0% | 558/558 | 0/558 | 0.000 | 0.00000 |
| all | -5.9% | +0.0% | 1116/1116 | 558/1116 | 0.500 | 0.00832 |

## Reference Agreement

| quantity | max absolute difference |
|---|---:|
| delta pct | `0.000e+00` |
| kappa safe | `0.000e+00` |
| target retention | `0.000e+00` |

## Readout

The deployment estimator reproduces the safe audit with `1116/1116` non-harming cells, mean `-5.9%`, and worst `+0.0%`. The maximum absolute delta difference from the rho-margin reference is `0.000e+00`, and the maximum kappa difference is `0.000e+00`. This makes the formula implementation auditable as a reusable estimator rather than a collection of one-off analysis scripts.
