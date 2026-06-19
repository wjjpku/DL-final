# Low-Tail Gate Config Route Audit

This audit keeps the joint LR-curvature prediction for all schedules except WSD-con targets whose `final_lr / peak_lr` is below a threshold. Those low-tail targets use one selected tail-gate configuration, fitted from cosine residuals only.

## Best Fully Non-Harming Route

- Threshold / tail config: `0.1` / `111`.
- Gate: `mode=late_lr`, `tau=0.03`, `sign=signed`, `shrink=0`.
- Mean / worst: `-37.67%` / `-10.80%`.
- Wins/non-harm: `15/15` and `15/15`.

## Best Worst-Case Route

- Threshold / tail config: `0.1` / `111`.
- Mean / worst: `-37.67%` / `-10.80%`.

## Per-Target Result For Best Route

| target | mean delta | worst scale | wins |
|---|---:|---:|---:|
| WSD sharp | -54.3% | -40.5% | 3/3 |
| WSD linear | -46.3% | -33.1% | 3/3 |
| WSD-con 3e-5 | -57.8% | -47.7% | 3/3 |
| WSD-con 9e-5 | -17.0% | -10.8% | 3/3 |
| WSD-con 18e-5 | -13.0% | -12.1% | 3/3 |

## Top-Safe Holdout Check

| split | selected route | dev mean | dev worst | test mean | test worst | test wins |
|---|---|---:|---:|---:|---:|---:|
| dev_sharp_linear__test_wsdcon | `thr=0.1, cfg=111, mode=late_lr, tau=0.03` | -50.3% | -33.1% | -29.3% | -10.8% | 9/9 |
| dev_wsdcon__test_sharp_linear | `thr=0.1, cfg=111, mode=late_lr, tau=0.03` | -29.3% | -10.8% | -50.3% | -33.1% | 6/6 |
| leave_target__wsd_20000_24000.csv | `thr=0.1, cfg=111, mode=late_lr, tau=0.03` | -33.5% | -10.8% | -54.3% | -40.5% | 3/3 |
| leave_target__wsdcon_18.csv | `thr=0.1, cfg=111, mode=late_lr, tau=0.03` | -43.8% | -10.8% | -13.0% | -12.1% | 3/3 |
| leave_target__wsdcon_3.csv | `thr=0.1, cfg=111, mode=late_lr, tau=0.03` | -32.6% | -10.8% | -57.8% | -47.7% | 3/3 |
| leave_target__wsdcon_9.csv | `thr=0.1, cfg=111, mode=late_lr, tau=0.03` | -42.8% | -12.1% | -17.0% | -10.8% | 3/3 |
| leave_target__wsdld_20000_24000.csv | `thr=0.1, cfg=111, mode=late_lr, tau=0.03` | -35.5% | -10.8% | -46.3% | -33.1% | 3/3 |
| leave_scale__25M | `thr=0.1, cfg=43, mode=sqrt_lr, tau=0.1` | -39.8% | -10.8% | -33.3% | -12.1% | 5/5 |
| leave_scale__100M | `thr=0.1, cfg=110, mode=late_lr, tau=0.03` | -38.5% | -12.1% | -36.1% | -10.8% | 5/5 |
| leave_scale__400M | `thr=0.1, cfg=43, mode=sqrt_lr, tau=0.1` | -34.8% | -10.8% | -43.5% | -13.6% | 5/5 |

## Reading

- This improves the low-tail WSD-con row without applying the same gate to the mid/high-tail WSD-con rows that it slightly harms.
- The route is schedule-only, but both the threshold and tail-gate configuration are selected in a development audit over available WSD-family targets.
