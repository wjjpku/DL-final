# MPL-LD Finite-Response Amplitude Sensitivity

The recommended model uses `amplitude_scale=1`, i.e. MPL's own `B` coefficient.  Other rows are fixed-scale probes; no amplitude is fitted from residuals or target losses.

| amplitude scale | WSD mean / worst / wins | controls mean / worst / nonharm |
|---:|---:|---:|
| 0.00 | +0.00% / +0.00% / 0/15 | +0.00% / +0.00% / 9/9 |
| 0.25 | -3.93% / -2.09% / 15/15 | +0.00% / +0.00% / 9/9 |
| 0.50 | -7.86% / -4.18% / 15/15 | +0.00% / +0.00% / 9/9 |
| 0.75 | -11.25% / -6.27% / 15/15 | +0.00% / +0.00% / 9/9 |
| 1.00 | -13.77% / -6.29% / 15/15 | +0.00% / +0.00% / 9/9 |
| 1.25 | -15.17% / -5.90% / 15/15 | +0.00% / +0.00% / 9/9 |
| 1.50 | -15.95% / -3.43% / 15/15 | +0.00% / +0.00% / 9/9 |
| 2.00 | -15.61% / +3.76% / 14/15 | +0.00% / +0.00% / 9/9 |

## Reading

- All-win WSD scale interval among tested fixed probes: `0.25` to `1.50`.
- The recommended `1.00` scale is inside a broad non-harm region, so the result is not an isolated exact-B accident.
- `0.00` is the MPL baseline and has no wins; improvement requires the finite-response term.
- Large scales are stronger on average but can over-correct individual targets; they are not recommended because they would require amplitude selection.
