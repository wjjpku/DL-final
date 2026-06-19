# All-Ratio WSD-con Route Audit

This audit uses the same Gaussian LR-level gate family for all WSD-con constant-tail schedules, routing by `final_lr / peak_lr`.  Coefficients are fitted from the cosine residual only; WSD-family losses are used for development ranking.

## Best Mean Route

- Mean / worst: `-37.88%` / `-11.80%`.
- Wins/non-harm: `15/15` and `15/15`.
- Ratio 0.1 config: `790` (center=0.5, width=0.1, time_power=0, tau=0.01, sign=signed, shrink=1).
- Ratio 0.3 config: `23` (center=0.2, width=0.05, time_power=1, tau=0.001, sign=signed, shrink=0).
- Ratio 0.6 config: `202` (center=0.2, width=0.25, time_power=1, tau=0.001, sign=signed, shrink=1).

## Comparison

- Joint-channel LR-curvature: mean `-37.53%`, worst `-10.80%`.
- Low-tail route: mean `-37.67%`, worst `-10.80%`.
- Mid/high-ratio route: mean `-37.85%`, worst `-11.80%`.
- All-ratio route: mean `-37.88%`, worst `-11.80%`.

## Per-Target Result

| target | mean delta | worst scale | wins |
|---|---:|---:|---:|
| WSD sharp | -54.3% | -40.5% | 3/3 |
| WSD linear | -46.3% | -33.1% | 3/3 |
| WSD-con 3e-5 | -57.9% | -48.9% | 3/3 |
| WSD-con 9e-5 | -17.6% | -11.8% | 3/3 |
| WSD-con 18e-5 | -13.3% | -12.3% | 3/3 |

## Routed Rows

| target | scale | route | delta | corr_mae | base_mae |
|---|---|---|---:|---:|---:|
| WSD-con 3e-5 | 25M | ratio_0.1 | -59.5% | 0.00181859 | 0.00448719 |
| WSD-con 9e-5 | 25M | ratio_0.3 | -22.2% | 0.00227633 | 0.00292648 |
| WSD-con 18e-5 | 25M | ratio_0.6 | -12.3% | 0.00219763 | 0.00250482 |
| WSD-con 3e-5 | 100M | ratio_0.1 | -48.9% | 0.00289642 | 0.0056694 |
| WSD-con 9e-5 | 100M | ratio_0.3 | -11.8% | 0.00435231 | 0.00493462 |
| WSD-con 18e-5 | 100M | ratio_0.6 | -14.1% | 0.0011874 | 0.00138164 |
| WSD-con 3e-5 | 400M | ratio_0.1 | -65.4% | 0.00282641 | 0.00817361 |
| WSD-con 9e-5 | 400M | ratio_0.3 | -18.7% | 0.00543904 | 0.00668731 |
| WSD-con 18e-5 | 400M | ratio_0.6 | -13.7% | 0.00220797 | 0.00255795 |

## Reading

- This is the best development number so far, but it uses one selected gate configuration per WSD-con final-LR ratio.
- The ratio branches are schedule-only at prediction time, but the branch hyperparameters were selected on the available WSD-family development set.
