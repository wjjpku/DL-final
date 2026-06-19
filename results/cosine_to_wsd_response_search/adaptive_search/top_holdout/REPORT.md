# Adaptive Top-Safe Holdout Check

This check uses the per-target rows for the top safe adaptive-search configurations. It is not a full hyperparameter holdout search; it tests whether the high-performing safe neighborhood is brittle.

| split | selected config | dev mean | dev worst | test mean | test worst | test wins |
|---|---|---:|---:|---:|---:|---:|
| dev_sharp_linear__test_wsdcon | `lambda_s=7, lambda_step=30, mu=0.07, tau=0.05, p=0.25, rho=0.5` | -46.4% | -28.4% | -20.1% | -0.6% | 9/9 |
| dev_wsdcon__test_sharp_linear | `lambda_s=2, lambda_step=20, mu=0.04, tau=0.05, p=0.25, rho=0.5` | -25.5% | -6.1% | -36.7% | -14.2% | 6/6 |
| leave_target__wsd_20000_24000.csv | `lambda_s=4, lambda_step=20, mu=0.04, tau=0.05, p=0.25, rho=0.5` | -28.1% | -6.1% | -43.9% | -25.2% | 3/3 |
| leave_target__wsdcon_18.csv | `lambda_s=7, lambda_step=30, mu=0.07, tau=0.05, p=0.25, rho=0.5` | -37.2% | -6.1% | -4.3% | -0.6% | 3/3 |
| leave_target__wsdcon_3.csv | `lambda_s=7, lambda_step=30, mu=0.07, tau=0.05, p=0.25, rho=0.5` | -26.7% | -0.6% | -46.3% | -43.5% | 3/3 |
| leave_target__wsdcon_9.csv | `lambda_s=7, lambda_step=30, mu=0.07, tau=0.05, p=0.25, rho=0.5` | -35.8% | -0.6% | -9.7% | -6.1% | 3/3 |
| leave_target__wsdld_20000_24000.csv | `lambda_s=4, lambda_step=20, mu=0.04, tau=0.05, p=0.25, rho=0.5` | -30.1% | -6.1% | -35.8% | -19.0% | 3/3 |
| leave_scale__25M | `lambda_s=4, lambda_step=20, mu=0.03, tau=0.05, p=0.25, rho=0.2` | -33.9% | -5.8% | -25.7% | -10.8% | 5/5 |
| leave_scale__100M | `lambda_s=4, lambda_step=20, mu=0.04, tau=0.05, p=0.25, rho=0.4` | -33.0% | -5.2% | -27.5% | -5.2% | 5/5 |
| leave_scale__400M | `lambda_s=7, lambda_step=30, mu=0.04, tau=0.05, p=0.25, rho=0` | -29.7% | -3.7% | -30.3% | -0.4% | 5/5 |

## Reading

- Within the top safe adaptive-search neighborhood, target-type and scale holdouts remain below MPL.
- This does not remove the need for new schedule families, but it reduces the concern that the best adaptive result is a single isolated WSD-family fit.
