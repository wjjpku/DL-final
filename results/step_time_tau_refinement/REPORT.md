# Step-Time Tau Refinement

Candidate response feature:

```text
Phi_tau(t) = sum_{k<=t} exp(-(t-k)/tau) * (eta_{k-1}-eta_k)_+ / eta_peak
prediction = MPL + kappa * Phi_tau
```

This is motivated by the residual gallery: smooth cosine residuals look like broad low-frequency MPL mismatch, while sharp/probe residuals catch up over a finite number of steps. A step-time kernel prevents the low-LR tail from becoming arbitrarily slow.

## Pareto Table

| tau | self mean | self worst | self wins | probes->WSD mean | probes->WSD worst | wsdcon3->WSD mean | wsdcon3->WSD worst |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | -30.5% | -6.6% | 18/18 | -19.3% | -16.7% | -23.2% | -18.7% |
| 768 | -34.4% | -4.5% | 18/18 | -26.1% | -22.0% | -31.9% | -25.6% |
| 1024 | -36.8% | -1.7% | 18/18 | -30.2% | -25.3% | -37.4% | -29.9% |
| 1280 | -38.1% | +0.4% | 16/18 | -32.4% | -27.0% | -40.8% | -32.3% |
| 1536 | -38.9% | +2.4% | 16/18 | -33.2% | -27.7% | -42.9% | -33.6% |
| 1792 | -39.0% | +3.8% | 16/18 | -33.2% | -27.9% | -44.1% | -34.1% |
| 2048 | -38.7% | +4.6% | 16/18 | -32.6% | -27.5% | -44.7% | -34.2% |
| 2304 | -38.2% | +4.9% | 14/18 | -31.7% | -26.2% | -44.9% | -34.1% |
| 2560 | -37.7% | +5.2% | 14/18 | -30.7% | -24.8% | -44.9% | -33.7% |
| 3072 | -36.7% | +5.5% | 14/18 | -28.4% | -22.0% | -43.8% | -32.5% |
| 3584 | -35.8% | +5.3% | 14/18 | -26.3% | -19.5% | -42.0% | -31.2% |
| 4096 | -35.1% | +5.0% | 14/18 | -24.4% | -17.3% | -40.0% | -29.9% |
| 5120 | -33.9% | +4.4% | 14/18 | -21.2% | -14.0% | -36.6% | -27.8% |
| 6144 | -32.9% | +3.8% | 13/18 | -18.9% | -11.5% | -33.8% | -26.2% |

## Current Best Reading

- Conservative deployment candidate: `tau=1024`. It keeps self-fit non-harming on `18/18` curves and gives pooled-probe WSD improvement `-30.2%` with worst `-25.3%`.
- Strong target-matched candidate: `tau=2304` with `wsdcon_3` calibration. It gives WSD improvement `-44.9%` with worst `-34.1%`, but it no longer keeps every probe self-fit non-harming.
- Best pooled-probe WSD mean occurs at `tau=1536`: `-33.2%` mean, worst `-27.7%`.
- The practical modeling update is to treat `S10_current` as too diffuse for cosine-like residuals, and to add a finite step-time catch-up channel for localized LR-drop prediction.
