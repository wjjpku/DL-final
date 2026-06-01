# 严格一致协议下的优化器影响实验

本实验用于回应“上一版优化器对照没有保持与官方 `MPL` 训练协议一致”的问题，因此只保留一个最严格、最可比的设置：`100M` 官方公开划分。

## 一致协议

以下因素全部保持一致，只更换优化器：

- 相同数据加载：直接复用 `external/MultiPowerLaw/src/data_loader.py`
- 相同训练集：`cosine_24000.csv`、`constant_24000.csv`、`wsdcon_9.csv`
- 相同测试集：`constant_72000.csv`、`cosine_72000.csv`、`wsd_20000_24000.csv`、`wsdld_20000_24000.csv`、`wsdcon_3.csv`、`wsdcon_18.csv`
- 相同完整训练点与完整评估点，不做下采样
- 相同损失：`log(loss)` 残差上的 Huber 损失
- 相同初始化后再分别交给 `AdamW` 与 `L-BFGS-B`
- `MPL + AdamW` 直接读取官方 `100.log` 中的最优参数，避免再次重复长时间重训，同时保证与官方训练协议严格一致

## 平均测试 MAE

| Model | Optimizer | 100M |
| --- | --- | ---: |
| TISSUE | L-BFGS-B | 0.003794 |
| TISSUE | AdamW | 0.015296 |
| MPL | L-BFGS-B | 0.003573 |
| MPL | AdamW | 0.004348 |

## 平均训练 MAE

| Model | Optimizer | 100M |
| --- | --- | ---: |
| TISSUE | L-BFGS-B | 0.001783 |
| TISSUE | AdamW | 0.011260 |
| MPL | L-BFGS-B | 0.001888 |
| MPL | AdamW | 0.003354 |

## 关键结论

- `MPL + AdamW` 在严格一致协议下已经恢复正常，平均测试 MAE 为 `0.004348`，不再出现上一版实验那种明显扭曲的现象。
- `MPL` 在本设置下对优化器相对稳健，`L-BFGS-B` 与 `AdamW` 的平均测试 MAE 只差 `0.000775`。
- `Tissue` 对优化器明显更敏感，`AdamW` 的平均测试 MAE 为 `0.015296`，约为 `L-BFGS-B` 的 `4.03x`。
- `Tissue + AdamW` 的退化不是“测试时不一致”导致的，因为本次训练集、测试集、初始化、损失函数和评估协议都已经统一。
- 参数分布也支持这一点：`Tissue` 中 `C` 从 `0.739` 被推到 `42.755`，`lam` 从 `0.9984` 降到 `0.9178`，说明 `AdamW` 下的解更偏向强退火响应；`MPL` 的参数虽然也变化，但整体预测性能保持稳定。

## 逐曲线现象

- `MPL + L-BFGS-B` 在 `constant_72000`、`wsd`、`wsdld`、`wsdcon_3` 上更优。
- `MPL + AdamW` 在 `cosine_72000` 上略优，在 `wsdcon_18` 上与 `L-BFGS-B` 基本持平。
- `Tissue + AdamW` 在全部 6 条测试曲线上都明显差于 `Tissue + L-BFGS-B`，其中 `constant_72000`、`wsd`、`wsdld` 退化尤其明显。

## 初始化参数

- `Tissue init`: `[2.9291, 0.3683539865848505, 1.0765324558013136, 50.0, 0.995]`
- `MPL init`: `[2.776246441007683, 0.6069347689045566, 0.4421032359426844, 456.86888343132114, 1.0, 0.5, 0.5]`

## 输出文件

- `results/optimizer_effect_strict/tables/strict_optimizer_metrics.csv`
- `results/optimizer_effect_strict/tables/strict_optimizer_best_params.json`
- `results/optimizer_effect_strict/figures/train_mae_summary.png`
- `results/optimizer_effect_strict/figures/test_mae_summary.png`
- `results/optimizer_effect_strict/figures/tissue_best_params.png`
- `results/optimizer_effect_strict/figures/mpl_best_params.png`
- `results/optimizer_effect_strict/figures/examples/`
