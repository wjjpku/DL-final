# 等权与 1-3-3 方案测试集对比

本实验只对比两种方案：

- `equal_1_1_1`：训练集三条曲线等权
- `constant_wsdcon_1_3_3`：降低 `cosine_24000` 权重，提高 `constant_24000` 与 `wsdcon_9` 权重

其余设置全部保持与官方 `MPL` 一致。

## 平均测试 MAE

| Scale | Equal 1-1-1 | 1-3-3 | Better |
| --- | ---: | ---: | --- |
| 25M | 0.003760 | 0.002959 | 1-3-3 |
| 100M | 0.004348 | 0.003701 | 1-3-3 |
| 400M | 0.004835 | 0.004160 | 1-3-3 |

## 输出目录

- `results/weighted_compare_equal_vs_133/tables/equal_vs_133_curve_metrics.csv`
- `results/weighted_compare_equal_vs_133/tables/equal_vs_133_summary.csv`
- `results/weighted_compare_equal_vs_133/tables/equal_vs_133_best_params.json`
- `results/weighted_compare_equal_vs_133/figures/avg_test_mae_equal_vs_133.png`
- `results/weighted_compare_equal_vs_133/figures/*_test_curve_mae_compare.png`
- `results/weighted_compare_equal_vs_133/<scale>M/curve_compare/*.png`
