# Schedule-Response Robustness Audit

This audit adds evidence around the projected cosine-kappa formula without changing the deployable model.

## Data Boundary

The repository contains five WSD-family target schedules for each of three scales.  It does not contain additional unseen WSD-family training runs.  Therefore a truly new held-out schedule experiment must be run externally; this audit focuses on robustness checks possible with the current data.

## Lambda Sensitivity

| lambda rule | mean / worst / wins / Pearson |
|---|---:|
| q2 half-life | -30.88% / -4.67% / 15/15, Pearson +0.910 |
| fixed 1 observation | -17.52% / -4.67% / 15/15, Pearson +0.259 |
| fixed 2 observations | +20.63% / +185.42% / 7/15, Pearson +0.255 |
| wrong fast lambda=20 | -14.76% / -3.49% / 15/15, Pearson +0.254 |
| oracle grid best (target-loss diagnostic) | -31.55% / -4.67% / 15/15, Pearson +0.921 |

The q2 rule is not presented as a target-tuned optimum.  The oracle grid is marked as diagnostic because it uses target loss.  The deployable q2 rule remains stable and close to the best fixed observation-scale rules without target loss.

## Projection Negative Control

| estimator | mean / worst / wins / Pearson |
|---|---:|
| projected q2 half-life kappa | -30.88% / -4.67% / 15/15, Pearson +0.910 |
| direct cosine kappa without MPL-LD projection | +625.92% / +2366.71% / 0/15, Pearson -0.849 |

The no-projection row uses the same LR-time response feature and the same source cosine residual, but estimates kappa before removing the MPL-LD tangent nuisance.  It fails catastrophically, which is the main evidence that this is an identification problem rather than arbitrary residual fitting.

## Same-Capacity Kernel Alternatives

| kernel | mean / worst / wins / Pearson |
|---|---:|
| lr_time_exp | -30.88% / -4.67% / 15/15, Pearson +0.910 |
| eta_level | +0.00% / +0.00% / 0/15, Pearson +nan |
| drop_cumsum | +27.15% / +221.99% / 7/15, Pearson +0.121 |
| drop_impulse | -0.00% / -0.00% / 15/15, Pearson +0.145 |
| step_time_exp | -3.05% / -0.42% / 15/15, Pearson +0.870 |
| power_law_lr_time | +119.07% / +350.44% / 1/15, Pearson +0.461 |
| signed_lr_time_exp | -30.24% / -5.67% / 15/15, Pearson +0.891 |

The LR-time exponential drop response is the main one-scalar feature.  Same-capacity alternatives are useful controls; the important comparison is that arbitrary level/drop features do not explain the response as cleanly as a causal drop relaxation kernel.

## Cross-Scale Transfer

| protocol | mean / worst / wins / Pearson |
|---|---:|
| same scale | -30.88% / -4.67% / 15/15, Pearson +0.910 |
| leave-one-scale-out pooled | +62.57% / +231.45% / 1/15, Pearson +0.641 |
| leave-one-scale-out mean kappa | -25.62% / -3.96% / 15/15, Pearson +0.671 |
| all-scale pooled | +193.89% / +521.21% / 0/15, Pearson +0.808 |
| all-scale mean kappa | -29.15% / -4.20% / 15/15, Pearson +0.808 |
| single_source_100M | -24.83% / -4.69% / 10/10, Pearson +0.731 |
| single_source_400M | -27.06% / -7.80% / 10/10, Pearson +0.869 |
| single_source_25M | -21.90% / -3.15% / 10/10, Pearson +0.921 |

Cross-scale transfer is a stricter test because kappa amplitudes are not guaranteed to be scale invariant.  These rows define the boundary of the current method more clearly than same-scale evaluation alone.

## Source-Only Calibration Window Rule

| fit start | max retention | finite-sample floor | passes source rule |
|---:|---:|---:|---:|
| 5000 | 0.013748 | 0.001912 | 0 |
| 6500 | 0.004225 | 0.001953 | 0 |
| 8000 | 0.001478 | 0.002000 | 1 |
| 10000 | 0.000489 | 0.002066 | 1 |
| 12000 | 0.000353 | 0.002132 | 1 |

The selected source-only fit start is `8000`.  Its target evaluation is -30.88% / -4.67% / 15/15, Pearson +0.910.  Target losses are not used in the selection rule.

## WSD-con Failure-Mode Slice

| scale | target | kappa_hat | kappa_star | ratio | MAE change | terminal MPL err | terminal corrected err |
|---:|---|---:|---:|---:|---:|---:|---:|
| 25 | WSD-con 3e-5 | 0.0186 | 0.0243 | 0.766 | -44.65% | +3.6153e-03 | +4.2162e-03 |
| 25 | WSD-con 9e-5 | 0.0186 | 0.0206 | 0.904 | -21.19% | +4.9339e-03 | +4.9345e-03 |
| 25 | WSD-con 18e-5 | 0.0186 | 0.0338 | 0.551 | -9.19% | -1.3363e-03 | -1.3363e-03 |
| 100 | WSD-con 3e-5 | 0.0327 | 0.0317 | 1.032 | -42.76% | +9.4968e-03 | +9.8974e-03 |
| 100 | WSD-con 9e-5 | 0.0327 | 0.0227 | 1.444 | -9.84% | +1.0203e-02 | +1.0203e-02 |
| 100 | WSD-con 18e-5 | 0.0327 | 0.0217 | 1.506 | -12.80% | -1.8822e-03 | -1.8822e-03 |
| 400 | WSD-con 3e-5 | 0.0318 | 0.0504 | 0.631 | -35.97% | +8.8612e-03 | +9.2503e-03 |
| 400 | WSD-con 9e-5 | 0.0318 | 0.0376 | 0.845 | -9.17% | +1.3847e-02 | +1.3847e-02 |
| 400 | WSD-con 18e-5 | 0.0318 | 0.0277 | 1.149 | -4.67% | +4.4782e-05 | +4.4782e-05 |

WSD-con remains the main fine-grained limitation: aggregate MAE improves, but the ordering across final LR values is weaker than the WSD sharp/linear split.

## Figures

- `figs/mpl_residual_anomaly_100M.png`
- `figs/projection_decomposition_cosine_100M.png`
- `figs/projection_ablation_time_errors_100M.png`
- `figs/representative_time_errors_100M.png`
- `figs/mae_change_heatmap.png`
- `LEAKAGE_AUDIT.md`
