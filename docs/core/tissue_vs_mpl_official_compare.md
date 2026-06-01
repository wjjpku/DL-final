# Tissue 与 MPL 官方公开划分对比

## 目标

本实验在 `MultiPowerLaw` 官方公开数据与官方训练/测试划分上，对 `Tissue` 与 `MPL` 做同口径比较，并将两者的预测曲线叠加到同一张图中。

训练/测试划分保持为：

- 训练集：`cosine_24000.csv`、`constant_24000.csv`、`wsdcon_9.csv`
- 测试集：`constant_72000.csv`、`cosine_72000.csv`、`wsd_20000_24000.csv`、`wsdld_20000_24000.csv`、`wsdcon_3.csv`、`wsdcon_18.csv`

其中：

- `MPL` 参数来自已经成功复现的官方日志
- `Tissue` 参数在相同训练集上重新拟合

运行脚本：

- [compare_tissue_mpl_official.py](file:///Users/jiaju/Documents/github/DL-final/repro/compare_tissue_mpl_official.py)

## 输出位置

- 对比图目录：[official_compare/figures/compare](file:///Users/jiaju/Documents/github/DL-final/results/official_compare/figures/compare)
- 汇总图：[official_avg_test_mae_compare.png](file:///Users/jiaju/Documents/github/DL-final/results/official_compare/figures/official_avg_test_mae_compare.png)
- 指标表：[official_tissue_mpl_metrics.csv](file:///Users/jiaju/Documents/github/DL-final/results/official_compare/tables/official_tissue_mpl_metrics.csv)
- 参数表：[official_tissue_mpl_params.json](file:///Users/jiaju/Documents/github/DL-final/results/official_compare/tables/official_tissue_mpl_params.json)

## 关键结果

### 平均测试指标

| Scale | MPL Avg MAE | Tissue Avg MAE | MPL Avg RMSE | Tissue Avg RMSE | MPL Avg R2 | Tissue Avg R2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 25M  | 0.003760 | 0.004658 | 0.004652 | 0.006124 | 0.998802 | 0.998012 |
| 100M | 0.004348 | 0.003798 | 0.005919 | 0.005329 | 0.998301 | 0.998656 |
| 400M | 0.004835 | 0.005334 | 0.007305 | 0.007745 | 0.997762 | 0.997271 |

结论：

- `25M`：`MPL` 整体更强
- `100M`：`Tissue` 整体更强
- `400M`：`MPL` 整体更强

### 逐曲线胜负

按测试集 `MAE` 比较：

- `25M`：`MPL 4` 胜，`Tissue 2` 胜
- `100M`：`MPL 2` 胜，`Tissue 4` 胜
- `400M`：`MPL 3` 胜，`Tissue 3` 胜

更细一点看：

- `WSDCon` 两阶段切换曲线上，`MPL` 多数更稳，尤其 `wsdcon_3`
- `WSD / WSDLD` 上，`Tissue` 在 `100M` 和 `400M` 更常获胜
- `constant_72000` 上，`Tissue` 在 `100M / 400M` 更强

## 推荐先看的图

如果只看最有代表性的几张：

- [25_compare_wsdcon_3.png](file:///Users/jiaju/Documents/github/DL-final/results/official_compare/figures/compare/25_compare_wsdcon_3.png)
- [100_compare_wsd_20000_24000.png](file:///Users/jiaju/Documents/github/DL-final/results/official_compare/figures/compare/100_compare_wsd_20000_24000.png)
- [100_compare_wsdld_20000_24000.png](file:///Users/jiaju/Documents/github/DL-final/results/official_compare/figures/compare/100_compare_wsdld_20000_24000.png)
- [400_compare_wsdcon_3.png](file:///Users/jiaju/Documents/github/DL-final/results/official_compare/figures/compare/400_compare_wsdcon_3.png)
- [official_avg_test_mae_compare.png](file:///Users/jiaju/Documents/github/DL-final/results/official_compare/figures/official_avg_test_mae_compare.png)

这些图里：

- 黑线是 `Ground Truth`
- 橙色虚线是 `MPL`
- 蓝色点划线是 `Tissue`

## Tissue 拟合参数

参数顺序为：

`[L0, A, alpha, C, lam]`

| Scale | Tissue 参数 |
| --- | --- |
| 25M  | `[3.0374, 0.5212, 0.5055, 1.6535, 0.99568]` |
| 100M | `[2.6794, 0.5658, 0.4956, 0.7337, 0.99843]` |
| 400M | `[2.3941, 0.6221, 0.4593, 0.9302, 0.99832]` |

完整精度见：

- [official_tissue_mpl_params.json](file:///Users/jiaju/Documents/github/DL-final/results/official_compare/tables/official_tissue_mpl_params.json)

## 当前判断

- 在 `MultiPowerLaw` 官方公开划分上，`MPL` 并不是在所有尺度和所有曲线上都优于 `Tissue`。
- `MPL` 的优势主要出现在两阶段切换更明显的 `WSDCon` 曲线。
- `Tissue` 在中等尺度和较平滑的外推曲线上也能很强，尤其 `100M` 上的整体平均误差更低。
- 如果后续要分析误差结构，最值得优先检查的是：
  - `25M/400M` 的 `wsdcon_3`
  - `100M` 的 `wsd_20000_24000`
  - `100M` 的 `wsdld_20000_24000`
