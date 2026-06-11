# Current-Law Calibration Search

Law is fixed: `MPL + kappa * DropRelaxS_lambda`. Only calibration changes.

Final target is still cosine-fit MPL evaluated on `wsd` and `wsdld`.

## Best final-target protocols

| rank | protocol | lambda | MAE | vs MPL | wins | kappa 25/100/400 |
|---:|---|---:|---:|---:|---:|---|
| 1 | `cross_scale_c_from_target_decays_LOO` | 5 | 0.00203 | -45.0% | 6/6 | 0.03862, 0.04669, 0.04782 |
| 2 | `cross_scale_c_from_target_decays_LOO` | 7 | 0.00203 | -44.9% | 6/6 | 0.04852, 0.05865, 0.06013 |
| 3 | `cross_scale_c_from_target_decays_LOO` | 3 | 0.00204 | -44.7% | 6/6 | 0.02923, 0.03533, 0.03614 |
| 4 | `cross_scale_c_from_target_decays_LOO` | 2 | 0.00205 | -44.3% | 6/6 | 0.02484, 0.03002, 0.03069 |
| 5 | `cross_scale_c_from_target_decays_LOO` | 10 | 0.00206 | -44.0% | 6/6 | 0.06377, 0.07708, 0.07913 |
| 6 | `cross_scale_c_from_target_decays_LOO` | 14 | 0.00210 | -43.1% | 6/6 | 0.08422, 0.1018, 0.1046 |
| 7 | `cross_scale_c_from_target_decays_LOO` | 20 | 0.00213 | -42.2% | 6/6 | 0.1145, 0.1384, 0.1424 |
| 8 | `cross_scale_c_from_target_decays_LOO` | 30 | 0.00216 | -41.4% | 6/6 | 0.1638, 0.1978, 0.2039 |
| 9 | `cross_scale_c_from_target_decays_LOO` | 50 | 0.00219 | -40.5% | 6/6 | 0.2591, 0.3128, 0.3229 |
| 10 | `fit_kappa_on_wsdcon_3+wsdcon_9+wsdcon_18` | 7 | 0.00302 | -17.9% | 6/6 | 0.01441, 0.01489, 0.02717 |
| 11 | `fit_kappa_on_wsdcon_3+wsdcon_9+wsdcon_18` | 10 | 0.00303 | -17.6% | 6/6 | 0.01774, 0.0202, 0.03469 |
| 12 | `fit_kappa_on_wsdcon_3+wsdcon_9+wsdcon_18` | 5 | 0.00304 | -17.4% | 6/6 | 0.01186, 0.01082, 0.02115 |

## Leave-one-noncos sanity check

These rows ask whether the same law predicts arbitrary held-out non-cosine curves.

| held-out | lambda | MAE | vs MPL | wins |
|---|---:|---:|---:|---:|
| `wsd_20000_24000` | 5 | 0.00310 | -21.7% | 3/3 |
| `wsd_20000_24000` | 10 | 0.00313 | -20.7% | 3/3 |
| `wsd_20000_24000` | 20 | 0.00333 | -15.8% | 3/3 |
| `wsdld_20000_24000` | 5 | 0.00278 | -18.5% | 3/3 |
| `wsdld_20000_24000` | 10 | 0.00280 | -18.0% | 3/3 |
| `wsdld_20000_24000` | 20 | 0.00291 | -14.7% | 3/3 |
| `wsdcon_3` | 5 | 0.00633 | +3.6% | 2/3 |
| `wsdcon_3` | 10 | 0.00568 | -7.0% | 2/3 |
| `wsdcon_3` | 20 | 0.00348 | -43.0% | 3/3 |
| `wsdcon_9` | 5 | 0.00531 | +9.4% | 0/3 |
| `wsdcon_9` | 10 | 0.00461 | -5.0% | 2/3 |
| `wsdcon_9` | 20 | 0.00423 | -12.7% | 3/3 |
| `wsdcon_18` | 5 | 0.00202 | -5.9% | 1/3 |
| `wsdcon_18` | 10 | 0.00193 | -10.0% | 3/3 |
| `wsdcon_18` | 20 | 0.00194 | -9.5% | 3/3 |
