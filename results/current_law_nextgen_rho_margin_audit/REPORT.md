# Next-Gen Rho Margin Audit

This audit rescans the posterior-predictive transfer shrinkage `c_n = n/(n+rho)` while keeping the target-identifiability gate fixed at `R_target >= 0.01`. It asks whether `rho=0.5` is a knife-edge or part of a stable safe range.

## Rho Sweep

| rho | all mean | all worst | all non-harm | main wins | mean shrink |
|---:|---:|---:|---:|---:|---:|
| `0.00` | -6.3% | +13.2% | 1095/1116 | 537/558 | 1.000 |
| `0.10` | -6.3% | +8.0% | 1101/1116 | 543/558 | 0.958 |
| `0.20` | -6.2% | +4.3% | 1109/1116 | 551/558 | 0.921 |
| `0.25` | -6.2% | +2.8% | 1111/1116 | 553/558 | 0.903 |
| `0.30` | -6.1% | +1.4% | 1112/1116 | 554/558 | 0.886 |
| `0.35` | -6.1% | +0.2% | 1114/1116 | 556/558 | 0.870 |
| `0.40` | -6.0% | +0.0% | 1116/1116 | 558/558 | 0.855 |
| `0.45` | -6.0% | +0.0% | 1116/1116 | 558/558 | 0.840 |
| `0.50` | -5.9% | +0.0% | 1116/1116 | 558/558 | 0.826 |
| `0.60` | -5.8% | +0.0% | 1116/1116 | 558/558 | 0.800 |
| `0.75` | -5.6% | +0.0% | 1116/1116 | 558/558 | 0.763 |
| `1.00` | -5.3% | +0.0% | 1116/1116 | 558/558 | 0.710 |
| `1.25` | -5.0% | +0.0% | 1116/1116 | 558/558 | 0.664 |
| `1.50` | -4.8% | +0.0% | 1116/1116 | 558/558 | 0.624 |
| `2.00` | -4.3% | +0.0% | 1116/1116 | 558/558 | 0.558 |

## Train-Size Breakdown For Selected Rho

| train curves | mean delta | worst delta | non-harm cells | wins | mean shrink |
|---:|---:|---:|---:|---:|---:|
| 1 | -7.5% | +0.0% | 144/144 | 90/144 | 0.667 |
| 2 | -6.4% | +0.0% | 315/315 | 180/315 | 0.800 |
| 3 | -5.9% | +0.0% | 360/360 | 180/360 | 0.857 |
| 4 | -5.0% | +0.0% | 225/225 | 90/225 | 0.889 |
| 5 | -3.1% | +0.0% | 72/72 | 18/72 | 0.909 |

## Readout

With the target gate fixed, the first fully non-harming grid value is `rho=0.40`. The full-useful safe plateau, defined as fully non-harming overall while preserving all `558/558` main-matrix wins, spans `rho=0.40` through `rho=2.00` on this grid. The selected `rho=0.50` lies inside this plateau, with mean `-5.9%`, worst `+0.0%`, and `1116/1116` non-harming cells. This means `rho=0.5` is not a knife-edge. Smaller values are more aggressive and can preserve slightly stronger mean improvement, but the current formula uses `0.5` because it is the weakest simple posterior-predictive half-degree prior that remains inside the stable non-harming range with margin on this audit.
