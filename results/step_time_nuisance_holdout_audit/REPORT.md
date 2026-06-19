# Step-Time Nuisance Holdout Audit

This audit tests whether the image-driven step-time estimator is robust beyond the full-grid selection set.

## Component Ablation

| component | self mean | offdiag mean | offdiag worst | probe -> WSD | cosine -> WSD |
|---|---:|---:|---:|---:|---:|
| raw_step_tau1024 | -36.8% | +77.4% | +1205.5% | -24.7% | +349.7% |
| +fourier_nuisance | +164.2% | +1585.1% | +41341.7% | -21.8% | +6694.0% |
| +EB | -15.5% | -12.1% | +23.8% | -21.8% | -8.1% |
| +target_drop_linear | -15.2% | -13.0% | +0.0% | -21.8% | -8.1% |
| conservative_dct2_retention | -15.2% | -11.8% | +0.0% | -17.8% | -5.5% |
| safe_old_feature_ref | -11.6% | -10.4% | -1.0% | -9.7% | -2.1% |

## Leave-One-Scale

| heldout scale | unrestricted selected | drop-linear-family selected | fixed best |
|---|---:|---:|---:|
| 25M | `step_tau1024__fourier2__eb_q75__R0p0__Tdrop_linear`: -14.8% / -1.5% | `step_tau1024__fourier2__eb_q75__R0p0__Tdrop_linear`: -14.8% / -1.5% | -14.8% / -1.5% |
| 100M | `step_tau1024__fourier2__eb_q75__R0p0__Tdrop_linear`: -10.2% / +0.0% | `step_tau1024__fourier2__eb_q75__R0p0__Tdrop_linear`: -10.2% / +0.0% | -10.2% / +0.0% |
| 400M | `step_tau1024__fourier2__eb_q75__R0p0__Tdrop_linear`: -13.9% / +0.0% | `step_tau1024__fourier2__eb_q75__R0p0__Tdrop_linear`: -13.9% / +0.0% | -13.9% / +0.0% |

## Leave-One-Target Schedule

| heldout target | unrestricted selected | drop-linear-family selected | fixed best |
|---|---:|---:|---:|
| Cosine | `step_tau1024__fourier2__eb_q75__R0p0__Tdrop_linear`: -4.3% / -1.5% | `step_tau1024__fourier2__eb_q75__R0p0__Tdrop_linear`: -4.3% / -1.5% | -4.3% / -1.5% |
| WSD sharp | `step_tau1024__fourier2__eb_q75__R0p0__Tdrop_linear`: -19.1% / +0.0% | `step_tau1024__fourier2__eb_q75__R0p0__Tdrop_linear`: -19.1% / +0.0% | -19.1% / +0.0% |
| WSD linear | `step_tau1024__fourier2__eb_q75__R0p0__Tdrop_linear`: -19.3% / +0.0% | `step_tau1024__fourier2__eb_q75__R0p0__Tdrop_linear`: -19.3% / +0.0% | -19.3% / +0.0% |
| WSD-con 3e-5 | `step_tau1024__fourier2__eb_q75__R0p0__Tdrop_linear`: -19.8% / +0.0% | `step_tau1024__fourier2__eb_q75__R0p0__Tdrop_linear`: -19.8% / +0.0% | -19.8% / +0.0% |
| WSD-con 9e-5 | `step_tau1024__fourier2__eb_q75__R0p0__Tdrop_linear`: -10.6% / +0.0% | `step_tau1024__fourier2__eb_q75__R0p0__Tdrop_linear`: -10.6% / +0.0% | -10.6% / +0.0% |
| WSD-con 18e-5 | `step_tau1024__fourier2__eb_q75__R0p0__Tnone`: +0.1% / +23.8% | `step_tau1024__fourier2__eb_q75__R0p0__Tdrop_linear`: -4.6% / +0.0% | -4.6% / +0.0% |

## Reading

- Fixed image-driven candidate `step_tau1024__fourier2__eb_q75__R0p0__Tdrop_linear` has full-grid self mean `-15.2%`, offdiag mean `-13.0%`, offdiag worst `+0.0%`, and probe-to-WSD mean `-21.8%`.
- Leave-one-scale fixed-best offdiag means range from `-14.8%` to `-10.2%`; worst deltas stay at or below `+0.0%`.
- Leave-one-target fixed-best offdiag means range from `-19.8%` to `-4.3%`; worst deltas stay at or below `+0.0%`.
- Unrestricted target holdout exposes the same issue seen in the plots: without a structural target-drop factor, selection can prefer a full-drop correction and then over-transfer to the smallest-drop target. Restricting to the drop-linear model family or using the fixed image-driven candidate removes this failure.
- The ablation shows why the target drop factor matters: it preserves WSD-target improvements while removing over-transfer to small-drop WSD-con targets.
