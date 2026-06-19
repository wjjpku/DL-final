# Next-Gen Scale-Holdout Constant Audit

This audit holds out one model scale at a time. It asks whether the two fixed constants, `R_target >= 0.01` and `rho=0.5`, remain safe when their supporting margin is inspected only on the other two scales.

## Target-Retention Floor

| heldout scale | train scales | train harmful max R | train main min R | 0.01 inside margin | heldout non-harm | heldout main wins | heldout worst |
|---:|---|---:|---:|---:|---:|---:|---:|
| 100 | `25|400` | 0.005721 | 0.014797 | 1 | 372/372 | 186/186 | +0.0% |
| 25 | `100|400` | 0.005721 | 0.014797 | 1 | 372/372 | 186/186 | +0.0% |
| 400 | `100|25` | 0.005721 | 0.014797 | 1 | 372/372 | 186/186 | +0.0% |

## Rho Shrinkage

| heldout scale | train scales | train first safe rho | selected rho | selected safe-side | heldout non-harm | heldout main wins | heldout worst |
|---:|---|---:|---:|---:|---:|---:|---:|
| 100 | `25|400` | 0.35 | 0.50 | 1 | 372/372 | 186/186 | +0.0% |
| 25 | `100|400` | 0.40 | 0.50 | 1 | 372/372 | 186/186 | +0.0% |
| 400 | `100|25` | 0.40 | 0.50 | 1 | 372/372 | 186/186 | +0.0% |

## Readout

For every held-out scale, `0.01` remains inside the target-retention margin inferred from the other two scales (`3/3`), and the held-out scale has full non-harm plus full main-matrix wins. For rho, the first safe value inferred from the two training scales is at most the selected `0.50` in every split (`3/3`), and held-out evaluation remains fully non-harming with all main-matrix wins. This supports treating the constants as scale-stable within the current three-scale matrix, while still not replacing true external scale or schedule-family validation.
