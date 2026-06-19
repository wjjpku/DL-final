# Strict Cosine-Only MPL Backbone Audit

This audit refits the MPL backbone using only `cosine_24000.csv` and `cosine_72000.csv`, then fits the residual correction from `cosine_72000.csv` and evaluates WSD-family targets.

## Backbone Check

- Cosine-only MPL vs frozen official MPL on WSD: mean MAE change `+55.0%`, worst `+106.8%`.
- This means the strict cosine-only backbone is substantially weaker on WSD than the frozen MPL backbone used in the main audits.

## Best Single-Config Channel-Shrink Correction

- Mean / worst vs strict cosine-only MPL: `-33.12%` / `-13.16%`.
- Wins: `15/15`.

## Best Decoupled-Channel Correction

- Mean / worst vs strict cosine-only MPL: `-33.35%` / `-13.16%`.
- Wins/non-harm: `15/15` and `15/15`.
- Smooth config: `start=12000`, `lambda=4`, `mu=0.05`, `modes=8`, `p=0.25`, `rho=0.2`.
- Step config: `start=12000`, `lambda=20`, `mu=0.02`, `modes=12`, `p=0`, `rho=0`.

## Per-Target Decoupled Result

| target | mean delta | worst scale | wins |
|---|---:|---:|---:|
| WSD sharp | -38.8% | -29.5% | 3/3 |
| WSD linear | -26.8% | -16.9% | 3/3 |
| WSD-con 3e-5 | -56.0% | -51.7% | 3/3 |
| WSD-con 9e-5 | -27.4% | -26.3% | 3/3 |
| WSD-con 18e-5 | -17.8% | -13.2% | 3/3 |

## Reading

- The residual correction still works under a strict cosine-only MPL backbone, with non-harming improvement on all WSD rows.
- The absolute WSD baseline is much worse after refitting MPL only on cosine, so this audit should not replace the frozen-backbone main result unless the assignment requires a fully cosine-only backbone too.
