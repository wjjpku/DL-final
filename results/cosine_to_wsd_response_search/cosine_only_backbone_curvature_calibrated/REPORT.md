# Strict-Calibrated LR-Curvature Audit

This audit keeps the strict cosine-only MPL backbone, but uses the best strict-backbone decoupled-channel calibration before adding the LR-curvature term. Coefficients are still fit from `cosine_72000.csv` residuals only.

## Channel Calibration

- Smooth channel: `start=12000`, `lambda=4`, `mu=0.05`, `modes=8`, `p=0.25`, `rho=0.2`.
- Step channel: `start=12000`, `lambda=20`, `mu=0.02`, `modes=12`, `p=0`, `rho=0`.

## Backbone Check

- Cosine-only MPL vs frozen official MPL on WSD: mean MAE change `+55.0%`, worst `+106.8%`.
- Strict percentages below are therefore robustness numbers against a weaker backbone, not the main frozen-backbone result.

## Best Strict-Calibrated Curvature Candidate

- Strict decoupled-channel baseline: mean `-33.35%`, worst `-13.16%`.
- Strict-calibrated curvature: mean `-33.68%`, worst `-14.27%`.
- Wins/non-harm: `15/15` and `15/15`.
- Config: `curvature_lambda=30`, `mode=diff_drop`, `tau2=0.001`, `shrink_curvature=1`, `signed_curvature_coef=0`.
- Mean step coefficients: primary `0.06915`, curvature `0.01551`.

## Best Worst-Case Strict-Calibrated Candidate

- Mean / worst: `-33.57%` / `-14.30%`.
- Config: `curvature_lambda=20`, `mode=diff_drop`, `tau2=0.001`, `shrink_curvature=1`, `signed_curvature_coef=0`.

## Frozen-Backbone Reference

- Frozen-backbone LR-curvature main result: mean `-37.47%`, worst `-9.43%`.

## Per-Target Strict-Calibrated Result

| target | mean delta | worst scale | wins |
|---|---:|---:|---:|
| WSD sharp | -38.8% | -29.5% | 3/3 |
| WSD linear | -26.8% | -16.9% | 3/3 |
| WSD-con 3e-5 | -56.6% | -52.9% | 3/3 |
| WSD-con 9e-5 | -27.9% | -27.4% | 3/3 |
| WSD-con 18e-5 | -18.3% | -14.3% | 3/3 |

## Top-Safe Holdout Check

| split | selected config | dev mean | dev worst | test mean | test worst | test wins |
|---|---|---:|---:|---:|---:|---:|
| dev_sharp_linear__test_wsdcon | `lambda2=4, mode=signed_d2_lr, tau2=0.003, shrink=1, signed=1` | -32.8% | -16.9% | -34.1% | -13.8% | 9/9 |
| dev_wsdcon__test_sharp_linear | `lambda2=30, mode=diff_drop, tau2=0.001, shrink=1, signed=0` | -34.3% | -14.3% | -32.8% | -16.9% | 6/6 |
| leave_target__wsd_20000_24000.csv | `lambda2=30, mode=diff_drop, tau2=0.001, shrink=1, signed=0` | -32.4% | -14.3% | -38.8% | -29.5% | 3/3 |
| leave_target__wsdcon_18.csv | `lambda2=4, mode=abs_diff_drop, tau2=0.01, shrink=1, signed=1` | -37.9% | -16.9% | -16.8% | -11.9% | 3/3 |
| leave_target__wsdcon_3.csv | `lambda2=30, mode=diff_drop, tau2=0.001, shrink=1, signed=0` | -27.9% | -14.3% | -56.6% | -52.9% | 3/3 |
| leave_target__wsdcon_9.csv | `lambda2=4, mode=abs_diff_drop, tau2=0.01, shrink=1, signed=1` | -35.2% | -11.9% | -27.6% | -26.5% | 3/3 |
| leave_target__wsdld_20000_24000.csv | `lambda2=30, mode=diff_drop, tau2=0.001, shrink=1, signed=0` | -35.4% | -14.3% | -26.8% | -16.9% | 3/3 |
| leave_scale__25M | `lambda2=4, mode=abs_diff_drop, tau2=0.01, shrink=1, signed=1` | -36.4% | -14.7% | -28.1% | -11.9% | 5/5 |
| leave_scale__100M | `lambda2=7, mode=diff_drop, tau2=0.03, shrink=1, signed=0` | -31.8% | -13.2% | -36.6% | -23.6% | 5/5 |
| leave_scale__400M | `lambda2=4, mode=abs_diff_drop, tau2=0.01, shrink=1, signed=1` | -33.4% | -11.9% | -34.2% | -14.7% | 5/5 |

## Reading

- With channel calibration aligned to the strict backbone, the LR-curvature term improves both mean and worst-case error over the strict decoupled-channel baseline.
- The selected curvature mode changes from signed second LR difference to `diff_drop`, which suggests strict MPL residuals encode the local step transition through the change in positive LR drop rather than raw LR curvature.
- This strengthens the robustness story, but the main result should still use the frozen-backbone protocol unless the assignment requires refitting MPL from cosine only.
