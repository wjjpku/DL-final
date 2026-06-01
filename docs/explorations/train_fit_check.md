# 训练集拟合能力检查

本报告只看论文口径官方训练集上的拟合效果。训练任务固定为：

- `cosine_24000`
- `constant_24000`
- `wsdcon_9`

训练集预测图目录：

- `results/paper_reproduction/mpl_only/{25M,100M,400M}`
- `results/paper_reproduction/mpl_vs_tissue_compare/` 中的 `*_compare_constant_24000.png`、`*_compare_cosine_24000.png`、`*_compare_wsdcon_9.png`

训练集平均指标如下：

| Scale | MPL Avg MAE | Tissue Avg MAE | MPL Avg RMSE | Tissue Avg RMSE | MPL Avg R2 | Tissue Avg R2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 25M | 0.003705 | 0.003020 | 0.005105 | 0.005358 | 0.998884 | 0.998828 |
| 100M | 0.003354 | 0.001781 | 0.004275 | 0.003108 | 0.999264 | 0.999513 |
| 400M | 0.005162 | 0.002341 | 0.006569 | 0.003564 | 0.998399 | 0.999514 |

结论：

- 两种方法在训练集上都拟合得很好，`R2` 基本都在 `0.9988` 以上。
- `25M` 上二者训练拟合接近，`Tissue` 的平均 `MAE` 略低，但 `MPL` 的平均 `RMSE` 略低。
- `100M` 和 `400M` 上，`Tissue` 在训练集上的平均误差明显更低。
- 因此，`MPL` 在部分测试任务上更强，并不是因为它单纯更会贴训练集；两者主要差别仍然来自泛化行为。

训练集逐图建议优先看：

- `25_compare_cosine_24000.png`
- `25_compare_constant_24000.png`
- `25_compare_wsdcon_9.png`
- `100_compare_wsdcon_9.png`
- `400_compare_wsdcon_9.png`