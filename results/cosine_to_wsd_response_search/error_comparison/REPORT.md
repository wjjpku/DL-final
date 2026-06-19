# Cosine-to-WSD Error Comparison

This visualization keeps the assignment protocol explicit: correction amplitudes are fitted from `cosine_72000.csv`; WSD-family losses are used only for evaluation and plotting.

## Compared Methods

- `MPL`: original MPL baseline.
- `MPL+old`: previous cosine-calibrated nextgen correction with S-time response `lambda=10` and target-retention gate.
- `MPL+response`: current cosine-calibrated response-kernel candidate.

Current response-kernel candidate:

```text
response_lambda = 20
nuisance_lambda = 0.07
max_mode = 8
ridge_tau = 0.05
retention_power = 0.5
rho = 0
```

## Aggregate Result

- Old mean / worst: `-17.2%` / `-2.2%`.
- New mean / worst: `-22.0%` / `-6.5%`.
- Wins: old `15/15`, new `15/15`.

## Target Breakdown

| target | old mean | old worst | new mean | new worst | new wins |
|---|---:|---:|---:|---:|---:|
| WSD sharp | -20.5% | -12.5% | -17.2% | -12.2% | 3/3 |
| WSD linear | -17.3% | -9.8% | -15.6% | -10.3% | 3/3 |
| WSD-con 3e-5 | -30.2% | -19.9% | -53.9% | -46.4% | 3/3 |
| WSD-con 9e-5 | -9.8% | -2.2% | -14.2% | -6.9% | 3/3 |
| WSD-con 18e-5 | -8.3% | -3.6% | -9.1% | -6.5% | 3/3 |

## Reading

- The new candidate remains a cosine-fitted model: WSD schedules only contribute their LR-derived response feature at prediction time.
- The visible improvement is mainly from changing the response time scale from the old general-purpose `lambda=10` to a cosine-to-WSD transfer value near `lambda=20`, plus stronger nuisance removal before estimating kappa from cosine.
- This is still a development result because the final candidate was selected by WSD-family ranking; a stronger final protocol should use a held-out split of WSD types or new schedules.
