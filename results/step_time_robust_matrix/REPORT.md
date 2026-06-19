# Step-Time Robust Matrix

This promotes the residual-shape finding into a matrix-style method comparison. The finite step-time response prevents the low-LR tail from producing the broad delayed cosine correction seen in the error plots.

## Self-Fit

| feature | mean delta | worst delta | wins | mean kappa |
|---|---:|---:|---:|---:|
| S10_current | -32.3% | -5.6% | 18/18 | 0.1125 |
| step_tau1024 | -36.8% | -1.7% | 18/18 | 0.1132 |
| step_tau1536 | -38.9% | +2.4% | 16/18 | 0.0797 |
| step_tau2304 | -38.2% | +4.9% | 14/18 | 0.0562 |

## Pooled Probe Calibration To WSD

| feature | probe -> WSD mean | worst | wins | probe -> cosine sanity |
|---|---:|---:|---:|---:|
| S10_current | -17.1% | -12.7% | 6/6 | -3.5% |
| step_tau1024 | -30.2% | -25.3% | 6/6 | -5.8% |
| step_tau1536 | -33.2% | -27.7% | 6/6 | -7.3% |
| step_tau2304 | -31.7% | -26.2% | 6/6 | -8.4% |

## Endpoint-Matched WSD Rule

For WSD targets whose final LR is `3e-5`, use the `wsdcon_3` probe to estimate kappa. This rule uses the target schedule endpoint, not target losses.

| rule | WSD mean | worst | wins | mean kappa |
|---|---:|---:|---:|---:|
| endpoint_tau1024 | -37.4% | -29.9% | 6/6 | 0.0436 |
| endpoint_tau1536 | -42.9% | -33.6% | 6/6 | 0.0385 |
| endpoint_tau2304 | -44.9% | -34.1% | 6/6 | 0.0316 |

## Response-Shape Diagnostic

The image-driven audit is in `response_shapes/REPORT.md`.

Key averaged shape metrics:

| schedule | S10 lag | tau1024 lag | S10 width | tau1024 width | raw low-freq R2 |
|---|---:|---:|---:|---:|---:|
| cosine 72k | +19029 | -2688 | 71.7% | 71.4% | 0.98 |
| WSD exp cooldown | +597 | -1963 | 15.3% | 14.7% | 0.88 |
| WSD linear cooldown | +256 | +256 | 15.3% | 14.7% | 0.91 |
| step to 3e-5 | -384 | -384 | 46.2% | 22.5% | 0.72 |

The main visual takeaway is that cosine residuals form a broad low-frequency wave. A finite step-time kernel fixes the extreme late peak of the old cumulative-LR response, but continuous cosine decay still spreads the response over most of the curve. Therefore same-curve cosine self-fit can look strong while still estimating a nuisance-dominated, weakly transferable kappa.

## Follow-Up Estimator Search

The follow-up search is in `../step_time_nuisance_estimator/REPORT.md`. It combines the finite step-time response with a low-frequency Fourier nuisance projection and a target-side total-drop factor. The best single-curve candidate found there is:

```text
step_tau1024 + Fourier2 nuisance + EB q75 + target drop_linear
```

It gives self mean `-15.2%`, off-diagonal mean `-13.0%`, off-diagonal worst `+0.0%`, cosine -> WSD `-8.4%`, and probe -> WSD `-21.8%`. This is the current best image-driven candidate for balancing self-fit and single-curve generalization. Pooled probe calibration remains much stronger for WSD targets, with the best row reaching WSD mean `-42.0%` and worst `-25.6%`.

## Reading

- Conservative replacement `step_tau1024` improves self-fit from `-32.3%` to `-36.8%` while keeping `18/18` self-fit wins.
- Pooled-probe WSD generalization improves from `-17.1%` to `-30.2%` with `step_tau1024`, preserving `6/6` WSD wins.
- Endpoint-matched aggressive variant `endpoint_tau2304` reaches `-44.9%` WSD mean MAE change with worst `-34.1%`.
- The remaining failure mode is diffuse cosine calibration. Step-time reduces but does not eliminate raw cosine over-transfer; cosine should still be treated as a low-frequency nuisance diagnostic rather than a primary kappa source.
