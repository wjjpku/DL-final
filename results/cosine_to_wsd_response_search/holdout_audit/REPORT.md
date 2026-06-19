# Cosine-to-WSD Holdout Audit

This audit keeps the fitted amplitude source fixed to `cosine_72000.csv`. It only varies which WSD subset is allowed to select hyperparameters, then evaluates the selected configuration on the held-out subset.

| split | selected config | dev mean | dev worst | test mean | test worst | test wins |
|---|---|---:|---:|---:|---:|---:|
| dev_sharp_linear__test_wsdcon | `lambda=7, mu=0.1, tau=0.02, p=0.5, rho=0` | -46.8% | -29.0% | +55.2% | +116.2% | 1/9 |
| dev_wsdcon__test_sharp_linear | `lambda=20, mu=0.05, tau=0.03, p=0.5, rho=0.5` | -25.8% | -6.4% | -16.3% | -10.1% | 6/6 |
| leave_target__wsd_20000_24000 | `lambda=20, mu=0.05, tau=0.03, p=0.5, rho=0.5` | -23.2% | -6.4% | -17.1% | -12.0% | 3/3 |
| leave_target__wsdld_20000_24000 | `lambda=20, mu=0.07, tau=0.05, p=0.5, rho=0` | -23.6% | -6.5% | -15.6% | -10.3% | 3/3 |
| leave_target__wsdcon_3 | `lambda=14, mu=0.1, tau=0.05, p=0.5, rho=0.35` | -15.3% | -0.0% | -39.5% | -24.9% | 3/3 |
| leave_target__wsdcon_9 | `lambda=20, mu=0.1, tau=0.05, p=0.5, rho=0.2` | -24.1% | -5.9% | -13.3% | -5.2% | 3/3 |
| leave_target__wsdcon_18 | `lambda=20, mu=0.05, tau=0.05, p=0.25, rho=0.75` | -25.2% | -6.6% | -9.0% | -6.2% | 3/3 |
| leave_scale__25M | `lambda=20, mu=0.03, tau=0.02, p=0.5, rho=0.5` | -22.2% | -6.4% | -20.7% | -9.4% | 5/5 |
| leave_scale__100M | `lambda=20, mu=0.07, tau=0.03, p=0.5, rho=0.75` | -23.8% | -5.6% | -18.0% | -4.9% | 5/5 |
| leave_scale__400M | `lambda=20, mu=0.07, tau=0.05, p=0.5, rho=0` | -20.7% | -6.9% | -24.6% | -6.5% | 5/5 |

## Reading

- A healthy result is not that every split chooses the same hyperparameters; it is that held-out WSD targets remain below MPL after hyperparameters are chosen elsewhere.
- Failures here would mean the cosine-derived correction is too sensitive to WSD-family hyperparameter selection, even though kappa itself is still fitted only on cosine.
