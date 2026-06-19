# MPL Final Term Multiplier Audit

问题：我们新增的 residual correction 是否只是 MPL 最后一项的若干倍？如果是，那么它可以写成

\[
\widehat e(t)\approx \rho L_G(t),\qquad L_G(t)=B D_{LD}(t),
\]

这等价于把 MPL 最后一项的系数从 \(B\) 改成 \((1+\rho)B\)。本实验直接拟合最优 \(\rho\)，并额外给一个更宽松的 affine check：\(c+\rho L_G(t)\)。affine 不是单独调 B，但可以判断形状是否至少接近。

## Main Result

结论：不是。直接的 \(\rho L_G(t)\) 基本不能解释我们的 residual correction；即使用更宽松的 \(c+\rho L_G(t)\)，在 WSD-con 目标上也明显不够。

### Aggregate

| group | rows | our mean delta | oracle scalar-LG mean delta | mean R2 to ours | affine mean R2 to ours | R2>0.8 count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all_core | 15 | -29.88% | -8.91% | +0.039 | +0.419 | 0/15 raw, 6/15 affine |
| wsd_final | 6 | -43.00% | -17.80% | +0.036 | +0.951 | 0/6 raw, 6/6 affine |
| wsdcon | 9 | -21.14% | -2.98% | +0.040 | +0.064 | 0/9 raw, 0/9 affine |

### Single Example: 100M WSD-con 3e-5

| quantity | value |
| --- | ---: |
| our correction MAE delta | -42.76% |
| oracle scalar `rho * L_G` MAE delta | -1.23% |
| best scalar R2 explaining our correction | +0.031 |
| best affine R2 explaining our correction | +0.147 |
| scalar alpha to ours | +0.0162245 |
| scalar relative RMSE to ours | 0.984 |

## Per-Target Table

| scale | target | our delta | oracle scalar-LG delta | scalar R2 to ours | affine R2 to ours | alpha to ours |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 25M | WSD sharp | -37.12% | -25.69% | +0.033 | +0.924 | +0.005742 |
| 25M | WSD linear | -29.81% | -29.65% | +0.040 | +0.976 | +0.004779 |
| 25M | WSD-con 3e-5 | -44.64% | +2.27% | +0.034 | +0.234 | +0.01162 |
| 25M | WSD-con 9e-5 | -21.19% | -0.50% | +0.047 | +0.026 | +0.00627 |
| 25M | WSD-con 18e-5 | -9.19% | -6.62% | +0.045 | +0.002 | +0.002431 |
| 100M | WSD sharp | -55.90% | -18.91% | +0.032 | +0.926 | +0.008021 |
| 100M | WSD linear | -49.01% | -24.65% | +0.039 | +0.978 | +0.006711 |
| 100M | WSD-con 3e-5 | -42.76% | -1.23% | +0.031 | +0.147 | +0.01622 |
| 100M | WSD-con 9e-5 | -9.84% | +3.10% | +0.044 | +0.013 | +0.008726 |
| 100M | WSD-con 18e-5 | -12.80% | -6.19% | +0.041 | +0.000 | +0.003254 |
| 400M | WSD sharp | -45.79% | -3.12% | +0.033 | +0.924 | +0.006624 |
| 400M | WSD linear | -40.38% | -4.76% | +0.039 | +0.976 | +0.005531 |
| 400M | WSD-con 3e-5 | -35.97% | +2.76% | +0.035 | +0.140 | +0.01398 |
| 400M | WSD-con 9e-5 | -9.17% | -2.17% | +0.046 | +0.011 | +0.007272 |
| 400M | WSD-con 18e-5 | -4.67% | -18.24% | +0.041 | +0.000 | +0.002674 |

## Figures

- `figs/single_100M_wsdcon_3.png`
- `figs/core_multiplier_grid_25M.png`
- `figs/core_multiplier_grid_100M.png`
- `figs/core_multiplier_grid_400M.png`
- `figs/r2_summary.png`

## Interpretation

调 MPL 最后一项的系数会给整个 \(L_G(t)=B D_{LD}(t)\) 乘一个全局倍数。这个方向包含很大的低频/全局形状；而我们的 correction 是一个更局部的 positive residual shape。因此二者不是同一个一维方向。尤其在 WSD-con 上，best scalar multiple 的 R2 只有约 0.04，即使加 intercept 的 affine 形状解释也很弱。

所以可以把“调 B”作为一个 baseline 或 ablation，但不能把它当作我们 residual 的等价替代。