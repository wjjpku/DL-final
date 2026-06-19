# Projection Ablation Error Curves

这些图比较同一个 q2 half-life 公式在有无 MPL-LD tangent projection 时预测出的 residual correction。黑线是真实 MPL residual，即 \(L-L_{MPL}\)；蓝线是有投影的预测误差项；红虚线是无投影的预测误差项。每个面板里的百分比是加入该 correction 后相对 MPL MAE 的变化。

## Figures

- `figs/predicted_residual_full_25M.png`: 完整纵轴，能看到无投影过冲幅度。
- `figs/predicted_residual_zoom_25M.png`: 以真实 residual 和有投影结果为主的放大图。
- `figs/remaining_residual_full_25M.png`: 校正后剩余 residual，完整纵轴。
- `figs/remaining_residual_zoom_25M.png`: 校正后剩余 residual，放大图。
- `figs/predicted_residual_full_100M.png`: 完整纵轴，能看到无投影过冲幅度。
- `figs/predicted_residual_zoom_100M.png`: 以真实 residual 和有投影结果为主的放大图。
- `figs/remaining_residual_full_100M.png`: 校正后剩余 residual，完整纵轴。
- `figs/remaining_residual_zoom_100M.png`: 校正后剩余 residual，放大图。
- `figs/predicted_residual_full_400M.png`: 完整纵轴，能看到无投影过冲幅度。
- `figs/predicted_residual_zoom_400M.png`: 以真实 residual 和有投影结果为主的放大图。
- `figs/remaining_residual_full_400M.png`: 校正后剩余 residual，完整纵轴。
- `figs/remaining_residual_zoom_400M.png`: 校正后剩余 residual，放大图。

## Same-Scale Metrics

| scale | target | with projection | without projection | no-proj / true residual L1 |
| --- | --- | ---: | ---: | ---: |
| 25M | WSD sharp | -37.12% | +258.70% | 3.45x |
| 25M | WSD linear | -29.81% | +191.11% | 2.67x |
| 25M | WSD-con 3e-5 | -44.64% | +2366.35% | 25.09x |
| 25M | WSD-con 9e-5 | -21.19% | +1036.77% | 11.04x |
| 25M | WSD-con 18e-5 | -9.19% | +345.52% | 4.07x |
| 100M | WSD sharp | -55.90% | +224.36% | 3.48x |
| 100M | WSD linear | -49.01% | +174.66% | 2.86x |
| 100M | WSD-con 3e-5 | -42.76% | +1847.59% | 19.73x |
| 100M | WSD-con 9e-5 | -9.84% | +604.46% | 6.37x |
| 100M | WSD-con 18e-5 | -12.80% | +675.24% | 7.17x |
| 400M | WSD sharp | -45.79% | +47.75% | 1.80x |
| 400M | WSD linear | -40.38% | +40.14% | 1.58x |
| 400M | WSD-con 3e-5 | -35.97% | +740.66% | 8.91x |
| 400M | WSD-con 9e-5 | -9.17% | +261.87% | 3.06x |
| 400M | WSD-con 18e-5 | -4.67% | +217.29% | 2.52x |

## Reading

无投影时，红线通常不是跟真实 residual 同量级的局部响应，而是把一个过大的、平滑的 cosine 残差方向迁移到了 WSD。这就是 summary 里 WSD 从全胜变成全败的直接视觉原因。
