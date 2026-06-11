# Next-Gen Component Ablation Audit

This audit isolates the two stabilizers in the next-generation formula: posterior-predictive shrinkage and the target-identifiability gate.

## Summary

| mode | group | mean delta | worst delta | non-harm cells | wins | target factor |
|---|---|---:|---:|---:|---:|---:|
| `no_predictive_shrinkage` | main_matrix | -12.6% | +13.2% | 537/558 | 537/558 | 1.000 |
| `no_predictive_shrinkage` | extra_holdout | +2.7% | +32.6% | 372/558 | 0/558 | 1.000 |
| `no_predictive_shrinkage` | all | -4.9% | +32.6% | 909/1116 | 537/1116 | 1.000 |
| `rho0p5_shrinkage` | main_matrix | -11.8% | -1.0% | 558/558 | 558/558 | 1.000 |
| `rho0p5_shrinkage` | extra_holdout | +2.2% | +22.5% | 372/558 | 0/558 | 1.000 |
| `rho0p5_shrinkage` | all | -4.8% | +22.5% | 930/1116 | 558/1116 | 1.000 |
| `rho0p5_plus_Rtarget_gate` | main_matrix | -11.8% | -1.0% | 558/558 | 558/558 | 1.000 |
| `rho0p5_plus_Rtarget_gate` | extra_holdout | +0.0% | +0.0% | 558/558 | 0/558 | 0.000 |
| `rho0p5_plus_Rtarget_gate` | all | -5.9% | +0.0% | 1116/1116 | 558/1116 | 0.500 |

## Train-Size Breakdown

| mode | train curves | mean delta | worst delta | non-harm cells | wins | target factor |
|---|---:|---:|---:|---:|---:|---:|
| `no_predictive_shrinkage` | 1 | -6.4% | +32.6% | 116/144 | 80/144 | 1.000 |
| `no_predictive_shrinkage` | 2 | -5.8% | +27.7% | 262/315 | 172/315 | 1.000 |
| `no_predictive_shrinkage` | 3 | -5.1% | +26.3% | 297/360 | 177/360 | 1.000 |
| `no_predictive_shrinkage` | 4 | -3.8% | +21.0% | 180/225 | 90/225 | 1.000 |
| `no_predictive_shrinkage` | 5 | -1.4% | +16.5% | 54/72 | 18/72 | 1.000 |
| `rho0p5_shrinkage` | 1 | -6.6% | +21.8% | 126/144 | 90/144 | 1.000 |
| `rho0p5_shrinkage` | 2 | -5.5% | +22.2% | 270/315 | 180/315 | 1.000 |
| `rho0p5_shrinkage` | 3 | -4.8% | +22.5% | 300/360 | 180/360 | 1.000 |
| `rho0p5_shrinkage` | 4 | -3.6% | +18.7% | 180/225 | 90/225 | 1.000 |
| `rho0p5_shrinkage` | 5 | -1.4% | +15.0% | 54/72 | 18/72 | 1.000 |
| `rho0p5_plus_Rtarget_gate` | 1 | -7.5% | +0.0% | 144/144 | 90/144 | 0.625 |
| `rho0p5_plus_Rtarget_gate` | 2 | -6.4% | +0.0% | 315/315 | 180/315 | 0.571 |
| `rho0p5_plus_Rtarget_gate` | 3 | -5.9% | +0.0% | 360/360 | 180/360 | 0.500 |
| `rho0p5_plus_Rtarget_gate` | 4 | -5.0% | +0.0% | 225/225 | 90/225 | 0.400 |
| `rho0p5_plus_Rtarget_gate` | 5 | -3.1% | +0.0% | 72/72 | 18/72 | 0.250 |

## Readout

Without posterior-predictive shrinkage, the next-gen direction has mean `-4.9%` but worst `+32.6%`. Adding `rho=0.5` shrinkage improves the worst case to `+22.5%` on the combined audit but still leaves diffuse-target failures. Adding the `R_target >= 0.01` gate gives `1116/1116` non-harming cells with worst `+0.0%`. Thus shrinkage controls finite-calibration amplitude over-transfer, while the target gate controls non-identifiable target directions.
