# Cosine-to-WSD Focus Report

This report isolates the assignment target: learn the correction from `cosine_72000.csv` and apply it to WSD-family targets.  It intentionally excludes WSD/probe-source routes.

## Formula Direction

The useful cosine-calibrated estimator is not the raw cosine residual fit.  It decomposes the cosine residual before estimating the transferable amplitude:

```text
r = L_true - L_MPL = kappa * phi + g_lowfreq + eps
M_lambda y = y - Q (Q^T Q + lambda D)^(-1) Q^T y
kappa = [n/(n+0.5)] * sqrt(||M_lambda phi||^2 / ||phi||^2)
        * max(0, <M_lambda phi, M_lambda r> / (||M_lambda phi||^2 + tau_EB^2))
```

Here `Q` is a DCT low-frequency nuisance basis, `lambda` is selected inside the train-only identifiable band `[0.01, 0.03]`, and the WSD target is used only through its LR-derived response feature and target retention gate.

## Main Result

- Conservative `final_no_cap`: mean `-5.8%`, worst `+0.0%`, wins `10/15`.
- Focused `nextgen_safe`: mean `-17.2%`, worst `-2.2%`, wins `15/15`.

## Per-Target Summary

| method | target | mean delta | worst scale | wins |
|---|---|---:|---:|---:|
| final_no_cap | WSD linear | -3.5% | +0.0% | 2/3 |
| final_no_cap | WSD sharp | -4.3% | +0.0% | 2/3 |
| final_no_cap | WSD-con 18e-5 | -4.5% | +0.0% | 2/3 |
| final_no_cap | WSD-con 3e-5 | -11.6% | +0.0% | 2/3 |
| final_no_cap | WSD-con 9e-5 | -5.3% | +0.0% | 2/3 |
| nextgen_safe_rho0p5_Rtarget0p01 | WSD linear | -17.3% | -9.8% | 3/3 |
| nextgen_safe_rho0p5_Rtarget0p01 | WSD sharp | -20.5% | -12.5% | 3/3 |
| nextgen_safe_rho0p5_Rtarget0p01 | WSD-con 18e-5 | -8.3% | -3.6% | 3/3 |
| nextgen_safe_rho0p5_Rtarget0p01 | WSD-con 3e-5 | -30.2% | -19.9% | 3/3 |
| nextgen_safe_rho0p5_Rtarget0p01 | WSD-con 9e-5 | -9.8% | -2.2% | 3/3 |

## Raw Cosine-Kappa Failure

| raw transfer group | MAE change | wins |
|---|---:|---:|
| WSD sharp | +240.6% | 0/3 |
| WSD-con step | +894.8% | 0/9 |

## Interpretation

- Raw cosine residual fitting fails because cosine residual is dominated by smooth MPL backbone drift, not only WSD-like decay lag.
- The actual solution is a residual decomposition problem: remove low-frequency drift, estimate only the identifiable response component, shrink the amplitude because one cosine curve is limited evidence, then apply it to WSD targets whose response direction is identifiable.
- This keeps the assignment goal intact: the calibration source remains cosine; WSD loss curves are used only for evaluation.
