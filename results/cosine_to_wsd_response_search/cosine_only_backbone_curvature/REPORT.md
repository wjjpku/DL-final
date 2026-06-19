# Strict Cosine-Only Backbone LR-Curvature Audit

This audit uses the MPL backbone refit from `cosine_24000.csv` and `cosine_72000.csv`, then fits the residual correction from `cosine_72000.csv` only. WSD-family losses are used for development ranking and evaluation, not for coefficient fitting.

## Backbone Check

- Cosine-only MPL vs frozen official MPL on WSD: mean MAE change `+55.0%`, worst `+106.8%`.
- This confirms that the strict cosine-only MPL backbone is much weaker on WSD than the frozen MPL backbone.

## Best Strict-Backbone Curvature Candidate

- Mean / worst vs strict cosine-only MPL: `-28.30%` / `-9.89%`.
- Wins/non-harm: `15/15` and `15/15`.
- Config: `curvature_lambda=7`, `mode=diff_drop`, `tau2=0.001`, `shrink_curvature=1`, `signed_curvature_coef=0`.
- Mean step coefficients: primary `0.03914`, curvature `0.04487`.

## Best Worst-Case Strict Candidate

- Mean / worst: `-28.03%` / `-11.07%`.
- Config: `curvature_lambda=20`, `mode=diff_drop`, `tau2=0.001`, `shrink_curvature=0`, `signed_curvature_coef=0`.

## Comparison

- Strict decoupled-channel: mean `-33.35%`, worst `-13.16%`.
- Strict LR-curvature: mean `-28.30%`, worst `-9.89%`.
- Frozen-backbone LR-curvature main result: mean `-37.47%`, worst `-9.43%`.

## Per-Target Strict Result

| target | mean delta | worst scale | wins |
|---|---:|---:|---:|
| WSD sharp | -38.8% | -29.5% | 3/3 |
| WSD linear | -26.8% | -16.9% | 3/3 |
| WSD-con 3e-5 | -42.5% | -34.8% | 3/3 |
| WSD-con 9e-5 | -19.8% | -16.3% | 3/3 |
| WSD-con 18e-5 | -13.6% | -9.9% | 3/3 |

## Top-Safe Holdout Check

| split | selected config | dev mean | dev worst | test mean | test worst | test wins |
|---|---|---:|---:|---:|---:|---:|
| dev_sharp_linear__test_wsdcon | `lambda2=4, mode=signed_d2_lr, tau2=0.01, shrink=1, signed=1` | -32.8% | -16.9% | -23.1% | -8.3% | 9/9 |
| dev_wsdcon__test_sharp_linear | `lambda2=7, mode=diff_drop, tau2=0.001, shrink=1, signed=0` | -25.3% | -9.9% | -32.8% | -16.9% | 6/6 |
| leave_target__wsd_20000_24000.csv | `lambda2=7, mode=diff_drop, tau2=0.001, shrink=1, signed=0` | -25.7% | -9.9% | -38.8% | -29.5% | 3/3 |
| leave_target__wsdcon_18.csv | `lambda2=7, mode=diff_drop, tau2=0.001, shrink=1, signed=0` | -32.0% | -16.3% | -13.6% | -9.9% | 3/3 |
| leave_target__wsdcon_3.csv | `lambda2=7, mode=diff_drop, tau2=0.001, shrink=1, signed=0` | -24.8% | -9.9% | -42.5% | -34.8% | 3/3 |
| leave_target__wsdcon_9.csv | `lambda2=7, mode=diff_drop, tau2=0.001, shrink=1, signed=0` | -30.4% | -9.9% | -19.8% | -16.3% | 3/3 |
| leave_target__wsdld_20000_24000.csv | `lambda2=7, mode=diff_drop, tau2=0.001, shrink=1, signed=0` | -28.7% | -9.9% | -26.8% | -16.9% | 3/3 |
| leave_scale__25M | `lambda2=7, mode=signed_d2_lr, tau2=0.003, shrink=0, signed=1` | -30.3% | -10.8% | -20.6% | -4.8% | 5/5 |
| leave_scale__100M | `lambda2=30, mode=diff_drop, tau2=0.001, shrink=0, signed=0` | -25.9% | -10.4% | -31.6% | -15.8% | 5/5 |
| leave_scale__400M | `lambda2=10, mode=diff_drop, tau2=0.001, shrink=0, signed=0` | -29.4% | -12.9% | -25.7% | -8.5% | 5/5 |

## Reading

- The correction still improves every WSD-family row when the MPL backbone itself is fit from cosine-only evidence.
- The strict-backbone percentages are measured against a weaker baseline, so they should be reported as a robustness audit rather than replacing the frozen-backbone main result.
- If this protocol becomes the final story, the next step is to tune the smooth/step channel calibration under the strict backbone as well, instead of only reusing the frozen-backbone channel settings.
