# 官方 MPL 训练曲线加权实验

本实验保持官方 `MPL` 的初始化、模型结构、`AdamW` 超参数、训练/测试划分和评估流程不变，只修改训练集三条曲线在目标函数中的聚合权重。

## 方案定义

- `equal_1_1_1`: 等权基线
- `cosine_constant_3_3_1`: `cosine_24000` 与 `constant_24000` 偏重，`wsdcon_9` 较轻
- `cosine_wsdcon_3_1_3`: `cosine_24000` 与 `wsdcon_9` 偏重，`constant_24000` 较轻
- `constant_wsdcon_1_3_3`: `constant_24000` 与 `wsdcon_9` 偏重，`cosine_24000` 较轻

说明：所有 `3+3+1` 方案都做了归一化，使三条训练曲线的总权重仍为 `3`，避免仅仅因为总梯度尺度变化而影响比较。

## 总排名（跨 25M / 100M / 400M 的平均测试 MAE）

| Rank | Scheme | Avg Test MAE |
| --- | --- | ---: |
| 1 | constant_wsdcon_1_3_3 | 0.003607 |
| 2 | equal_1_1_1 | 0.004314 |
| 3 | cosine_wsdcon_3_1_3 | 0.007432 |
| 4 | cosine_constant_3_3_1 | 0.007934 |

## 分尺度平均测试 MAE

| Scale | Scheme | Avg Test MAE | Avg Test Huber |
| --- | --- | ---: | ---: |
| 25M | equal_1_1_1 | 0.003760 | 0.000223 |
| 25M | cosine_constant_3_3_1 | 0.014470 | 0.000953 |
| 25M | cosine_wsdcon_3_1_3 | 0.010446 | 0.000533 |
| 25M | constant_wsdcon_1_3_3 | 0.002959 | 0.000161 |
| 100M | equal_1_1_1 | 0.004348 | 0.000355 |
| 100M | cosine_constant_3_3_1 | 0.004141 | 0.000263 |
| 100M | cosine_wsdcon_3_1_3 | 0.005218 | 0.000464 |
| 100M | constant_wsdcon_1_3_3 | 0.003701 | 0.000273 |
| 400M | equal_1_1_1 | 0.004835 | 0.000350 |
| 400M | cosine_constant_3_3_1 | 0.005193 | 0.000418 |
| 400M | cosine_wsdcon_3_1_3 | 0.006632 | 0.000620 |
| 400M | constant_wsdcon_1_3_3 | 0.004160 | 0.000309 |

## 关键观察

- `constant_wsdcon_1_3_3` 是跨尺度最稳的方案：跨 `25M / 100M / 400M` 的平均测试 `MAE` 为 `0.003607`，优于等权基线 `0.004314`。
- 它在 `18` 个测试任务里拿到 `14` 个单项最优，只在 `25M-cosine_72000`、`25M-constant_72000`、`100M-cosine_72000`、`100M-wsdcon_18` 这 `4` 项上没有赢。
- `cosine_constant_3_3_1` 与 `cosine_wsdcon_3_1_3` 都明显弱于基线，尤其在 `25M` 上退化最明显，说明把 `cosine_24000` 过度放大并不会带来更好泛化。
- 从结果看，官方训练集三条曲线里，`constant_24000` 与 `wsdcon_9` 对测试集泛化更关键；相反，降低 `cosine_24000` 权重并没有伤害整体表现，反而在多数测试曲线上更稳。

## 结果目录

- `results/weighted_mpl_official/tables/weighted_scheme_metrics.csv`
- `results/weighted_mpl_official/tables/weighted_scheme_summary.csv`
- `results/weighted_mpl_official/tables/weighted_scheme_overall_ranking.csv`
- `results/weighted_mpl_official/figures/avg_test_mae_by_scheme.png`
- `results/weighted_mpl_official/<scale>M/<scheme>/loss_monitor.png`
- `results/weighted_mpl_official/<scale>M/<scheme>/train/*.png`
- `results/weighted_mpl_official/<scale>M/<scheme>/test/*.png`
