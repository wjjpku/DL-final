# 多比例训练权重方案测试集对比

本实验比较如下方案，顺序均对应训练集 `[cosine_24000, constant_24000, wsdcon_9]`：

- `111`: 等权
- `133`: 降低 `cosine_24000`，提高 `constant_24000` 与 `wsdcon_9`
- `144`
- `122`
- `124`
- `142`

所有方案都做了归一化，总权重保持一致，其余设置完全沿用官方 `MPL`。

## 跨尺度总排名

| Rank | Scheme | Avg Test MAE |
| --- | --- | ---: |
| 1 | 133 | 0.003607 |
| 2 | 124 | 0.003928 |
| 3 | 144 | 0.004230 |
| 4 | 111 | 0.004314 |
| 5 | 122 | 0.004474 |
| 6 | 142 | 0.006467 |

## 分尺度平均测试 MAE

| Scale | Scheme | Avg Test MAE |
| --- | --- | ---: |
| 25M | 111 | 0.003760 |
| 25M | 133 | 0.002959 |
| 25M | 144 | 0.003017 |
| 25M | 122 | 0.003075 |
| 25M | 124 | 0.003554 |
| 25M | 142 | 0.003029 |
| 100M | 111 | 0.004348 |
| 100M | 133 | 0.003701 |
| 100M | 144 | 0.003454 |
| 100M | 122 | 0.005777 |
| 100M | 124 | 0.003465 |
| 100M | 142 | 0.011671 |
| 400M | 111 | 0.004835 |
| 400M | 133 | 0.004160 |
| 400M | 144 | 0.006218 |
| 400M | 122 | 0.004569 |
| 400M | 124 | 0.004764 |
| 400M | 142 | 0.004701 |

## 输出目录

- `results/weighted_scheme_compare/tables/curve_metrics.csv`
- `results/weighted_scheme_compare/tables/summary.csv`
- `results/weighted_scheme_compare/tables/best_params.json`
- `results/weighted_scheme_compare/figures/avg_test_mae_all_schemes.png`
- `results/weighted_scheme_compare/figures/*_test_curve_mae_compare.png`
- `results/weighted_scheme_compare/<scale>M/curve_compare/*.png`
