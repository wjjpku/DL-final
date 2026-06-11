# Target-Retention Margin Audit

This audit checks whether the target-identifiability threshold `R_target(lambda) >= 0.01` is a knife-edge. It reuses the raw next-generation predictions from the target-identifiability audit and rescans binary retention gates without refitting kappa.

## Margin

| quantity | value |
|---|---:|
| max raw-harmful target retention | `0.005721` |
| chosen threshold | `0.010000` |
| min main-matrix target retention | `0.014797` |
| geometric midpoint | `0.009201` |
| chosen / harmful max | `1.75x` |
| main min / chosen | `1.48x` |

For a binary gate that admits targets with `R_target >= threshold`, every threshold strictly above the harmful maximum and no larger than the main-matrix minimum blocks the observed harmful diffuse targets while retaining the full main-matrix target set. The chosen `0.01` lies inside this interval, not at either boundary.

## Threshold Sweep

| threshold | all mean | all worst | all non-harm | main wins | admitted target factor |
|---:|---:|---:|---:|---:|---:|
| `0.0000` | -4.8% | +22.5% | 930/1116 | 558/558 | 1.000 |
| `0.0025` | -4.8% | +22.5% | 930/1116 | 558/558 | 0.667 |
| `0.0050` | -5.0% | +22.5% | 981/1116 | 558/558 | 0.621 |
| `0.0075` | -5.9% | +0.0% | 1116/1116 | 558/558 | 0.500 |
| `0.0100` | -5.9% | +0.0% | 1116/1116 | 558/558 | 0.500 |
| `0.0150` | -5.9% | +0.0% | 1116/1116 | 540/558 | 0.484 |
| `0.0200` | -5.8% | +0.0% | 1116/1116 | 513/558 | 0.460 |
| `0.0500` | -5.7% | +0.0% | 1116/1116 | 465/558 | 0.417 |
| `0.1000` | -5.7% | +0.0% | 1116/1116 | 465/558 | 0.417 |
| `0.2000` | -4.8% | +0.0% | 1116/1116 | 383/558 | 0.343 |

## Curve Retention Summary

| curve | group | min R | max R | raw worst | raw harm cells |
|---|---|---:|---:|---:|---:|
| `Constant 24k` | extra_holdout | 0.000000 | 0.000000 | +0.0% | 0/186 |
| `Constant 72k` | extra_holdout | 0.000000 | 0.000000 | +0.0% | 0/186 |
| `Cosine 24k` | extra_holdout | 0.004180 | 0.005721 | +22.5% | 186/186 |
| `Cosine` | main_matrix | 0.014797 | 0.020411 | -1.0% | 0/93 |
| `WSD sharp` | main_matrix | 0.127387 | 0.211331 | -6.1% | 0/93 |
| `WSD-con 18e-5` | main_matrix | 0.654885 | 0.750869 | -2.8% | 0/93 |
| `WSD-con 3e-5` | main_matrix | 0.192332 | 0.272263 | -16.3% | 0/93 |
| `WSD-con 9e-5` | main_matrix | 0.453438 | 0.571272 | -1.4% | 0/93 |
| `WSD linear` | main_matrix | 0.159492 | 0.264453 | -4.8% | 0/93 |

## Readout

The full-transfer plateau with `1116/1116` non-harming cells and all `558/558` main-matrix wins spans approximately `0.005721` to `0.014797` in this event-grid scan. At the chosen threshold, the combined audit has mean `-5.9%`, worst `+0.0%`, and `1116/1116` non-harming cells. Lowering the threshold to `0.005` admits the diffuse cosine target and restores the `+22.5%` failure; increasing it beyond the main cosine retention remains safe but starts dropping useful main-matrix transfers. Thus `0.01` is best read as a margin-based identifiability floor, not a tuned loss-optimal threshold.
