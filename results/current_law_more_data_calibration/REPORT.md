# Current-Law More-Data Calibration

Fixed law: `MPL + kappa * DropRelaxS_lambda`. More calibration data, no new feature.

| protocol | lambda mode | lambda | MAE | vs MPL | wins |
|---|---|---:|---:|---:|---:|
| `other_sharp_only_leave_one_sharp` | `global_train` | 3 | 0.00184 | -49.9% | 6/6 |
| `other_sharp_only_leave_one_sharp` | `fixed10` | 10 | 0.00188 | -49.0% | 6/6 |
| `all_wsdcon_plus_other_sharp_leave_one_sharp` | `fixed10` | 10 | 0.00296 | -19.5% | 6/6 |
| `all_wsdcon_probes_to_both_sharp` | `fixed10` | 10 | 0.00303 | -17.6% | 6/6 |
| `all_wsdcon_plus_other_sharp_leave_one_sharp` | `global_train` | 20 | 0.00312 | -15.3% | 6/6 |
| `all_wsdcon_probes_to_both_sharp` | `global_train` | 20 | 0.00316 | -14.1% | 6/6 |

## Interpretation

- More data is not monotone. The strongest held-out sharp-decay prediction comes from the opposite sharp-decay shape alone; adding heterogeneous `wsdcon` probes dilutes the amplitude.
- Probe-only calibration is leakage-safe for the final `cosine -> wsd/wsdld` story, but its gain is smaller.
- Selecting lambda from the larger calibration set often prefers a smaller lambda than pure probe fitting; fixed lambda=10 remains a conservative theory-first setting.
