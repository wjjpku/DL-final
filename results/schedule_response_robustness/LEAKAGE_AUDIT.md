# Target-Leakage Audit

| stage | files / quantities used | target WSD loss used? | notes |
|---|---|---:|---|
| Calibration | `cosine_72000.csv`, cosine LR schedule, frozen MPL prediction | no | Computes source residual and projected kappa. |
| Target feature construction | target LR schedule only | no | Computes `q2`, `lambda_s`, and `phi_s(t)`. |
| Prediction | frozen MPL target prediction, `phi_s(t)`, source-only `kappa_hat` | no | Outputs `L_MPL + kappa_hat phi`. |
| Evaluation | target loss curve | yes | Computes MAE, terminal error, and oracle diagnostics only. |
| Oracle kappa star | target residual | yes | Diagnostic; never used in deployable prediction. |
| Oracle lambda grid | target MAE | yes | Upper-bound diagnostic; separated from deployable q2 rule. |

Deployable rows in the main tables use no target WSD loss for calibration or prediction.
