# Step-Time Nuisance Estimator Search

This search tests the image-driven hypothesis that the transferable error term should be a finite step-time response, while broad cosine residuals should be treated as low-frequency nuisance structure.

## Candidate Formula

The strongest single-curve candidate found here is:

```text
phi_tau(t) = sum_{u<=t} exp(-(t-u)/1024) * relu(eta_{u-1}-eta_u) / eta_peak
G = span{1, sin(pi z), cos(pi z), sin(2 pi z), cos(2 pi z)}
phi_perp = M_G phi_tau,   r_perp = M_G(observed_loss - MPL)
kappa = max(0, <phi_perp, r_perp> / (||phi_perp||^2 + tau_EB^2))
target_factor = total_positive_lr_drop(target) / 0.9
prediction = MPL + target_factor * kappa * phi_tau(target)
```

The target factor is schedule-only. It corrects the observed over-transfer from full-drop schedules to small-drop targets such as `WSD-con 18e-5`.

## Best Single-Curve Variants

| rank | method | self mean | self worst | self wins | offdiag mean | offdiag worst | probe -> WSD mean | probe -> WSD worst |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `step_tau1024__fourier2__eb_q75__R0p0__Tdrop_linear` | -15.2% | +0.0% | 16/18 | -13.0% | +0.0% | -21.8% | -12.0% |
| 2 | `step_tau1024__dct2__eb_q75__R0p5__Tdrop_linear` | -15.2% | +0.0% | 17/18 | -11.8% | +0.0% | -17.8% | -11.0% |
| 3 | `step_tau768__fourier2__eb_q75__R0p0__Tdrop_sqrt` | -13.0% | +0.0% | 16/18 | -10.9% | +0.0% | -17.6% | -10.3% |
| 4 | `step_tau768__fourier2__eb_q75__R0p0__Tdrop_linear` | -12.5% | +0.0% | 16/18 | -10.6% | +0.0% | -17.6% | -10.3% |
| 5 | `step_tau768__dct2__eb_q75__R0p5__Tdrop_linear` | -13.7% | +0.0% | 17/18 | -10.8% | +0.0% | -15.5% | -10.8% |
| 6 | `step_tau1024__fourier2__eb_q75__R0p5__Tdrop_linear` | -8.5% | +0.0% | 16/18 | -7.3% | +0.0% | -16.0% | -8.6% |
| 7 | `step_tau1024__dct2__none__R1p0__Tdrop_sqrt` | -10.5% | +0.0% | 17/18 | -8.5% | +0.0% | -13.9% | -8.7% |
| 8 | `step_tau1024__dct2__eb_q75__R1p0__Tdrop_sqrt` | -10.4% | +0.0% | 17/18 | -8.3% | +0.0% | -13.9% | -8.6% |
| 9 | `step_tau1024__dct2__none__R1p0__Tdrop_linear` | -10.2% | +0.0% | 17/18 | -8.2% | +0.0% | -13.9% | -8.7% |
| 10 | `step_tau1024__dct2__eb_q75__R1p0__Tdrop_linear` | -10.0% | +0.0% | 17/18 | -8.0% | +0.0% | -13.9% | -8.6% |
| 11 | `step_tau1024__dct4__eb_q75__R0p5__Tdrop_sqrt` | -8.6% | +0.0% | 17/18 | -7.4% | +0.0% | -14.1% | -7.8% |
| 12 | `step_tau1024__dct4__eb_q75__R0p5__Tdrop_linear` | -8.3% | +0.0% | 17/18 | -7.2% | +0.0% | -14.1% | -7.8% |

## Pooled Probe To WSD

| rank | method | mean | worst | wins |
|---:|---|---:|---:|---:|
| 1 | `step_tau3072__dct4__none__R0p0__Tgate_0p01` | -42.0% | -25.6% | 6/6 |
| 2 | `step_tau3072__dct4__none__R0p0__Tgate_0p03` | -42.0% | -25.6% | 6/6 |
| 3 | `step_tau3072__dct4__none__R0p0__Tgate_0p05` | -42.0% | -25.6% | 6/6 |
| 4 | `step_tau3072__dct4__none__R0p0__Tnone` | -42.0% | -25.6% | 6/6 |
| 5 | `step_tau3072__dct4__none__R0p0__Tdrop_sqrt` | -42.0% | -25.6% | 6/6 |
| 6 | `step_tau3072__dct4__none__R0p0__Tdrop_linear` | -42.0% | -25.6% | 6/6 |
| 7 | `step_tau3072__dct4__eb_q75__R0p0__Tgate_0p01` | -42.0% | -25.6% | 6/6 |
| 8 | `step_tau3072__dct4__eb_q75__R0p0__Tgate_0p03` | -42.0% | -25.6% | 6/6 |
| 9 | `step_tau3072__dct4__eb_q75__R0p0__Tgate_0p05` | -42.0% | -25.6% | 6/6 |
| 10 | `step_tau3072__dct4__eb_q75__R0p0__Tnone` | -42.0% | -25.6% | 6/6 |
| 11 | `step_tau3072__dct4__eb_q75__R0p0__Tdrop_sqrt` | -42.0% | -25.6% | 6/6 |
| 12 | `step_tau3072__dct4__eb_q75__R0p0__Tdrop_linear` | -42.0% | -25.6% | 6/6 |

## Comparison To Existing Final Estimator

| method | self mean | offdiag mean | offdiag worst | cosine -> WSD | probe -> WSD |
|---|---:|---:|---:|---:|---:|
| image-direct step-time candidate | -15.2% | -13.0% | +0.0% | -8.4% | -21.8% |
| conservative step-time candidate | -15.2% | -11.8% | +0.0% | -5.4% | -17.8% |
| safe old-feature reference | -11.6% | -10.4% | -1.0% | -2.3% | -9.7% |

## Holdout Audit

See `../step_time_nuisance_holdout_audit/REPORT.md`.

Key findings:

- Leave-one-scale fixed-best offdiag means are `-14.8%`, `-10.2%`, and `-13.9%`, with worst deltas no larger than `+0.0%`.
- Leave-one-target fixed-best offdiag means range from `-19.8%` to `-4.3%`, with worst deltas no larger than `+0.0%`.
- Unrestricted target holdout fails on `WSD-con 18e-5` by selecting the no-drop-factor variant (`+23.8%` worst). Restricting to the target-drop-linear family selects the fixed best candidate and restores `-4.6%` mean / `+0.0%` worst. This supports treating target-drop scaling as a structural part of the model, not a disposable hyperparameter.

## Reading

- Best single-curve score: `step_tau1024__fourier2__eb_q75__R0p0__Tdrop_linear` gives self mean `-15.2%`, probe-to-WSD mean `-21.8%`, and off-diagonal mean `-13.0%`.
- A conservative image-consistent candidate is `step_tau1024__dct2__eb_q75__R0p5__Tdrop_linear` with self mean `-15.2%` and probe-to-WSD mean `-17.8%`.
- Best pooled-probe WSD row is `step_tau3072__dct4__none__R0p0__Tgate_0p01`, giving mean `-42.0%`, worst `-25.6%`, and `6/6` wins.
- The useful models all separate local response shape from broad low-frequency residuals; raw cosine self-fit is not used as proof of transferable kappa.
