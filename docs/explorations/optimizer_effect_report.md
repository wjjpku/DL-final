# 优化器影响实验

本实验只在 `100M` 尺度上进行，目的不是重新比较模型大小，而是隔离“优化器选择”对同一公式模型的影响。

## 实验设置

- 训练集：`cosine_24000`、`constant_24000`、`wsdcon_9`
- 测试集：`constant_72000`、`cosine_72000`、`wsd_20000_24000`、`wsdld_20000_24000`、`wsdcon_3`、`wsdcon_18`
- 比较对象：
  - `Tissue + L-BFGS-B`
  - `Tissue + AdamW`
  - `MPL + L-BFGS-B`
  - `MPL + AdamW`
- 对照方式：同一训练/测试划分、同一 `100M` 数据、同一初始化模板，只比较优化器。
- 为了控制运行时间，拟合阶段使用训练曲线稀疏采样点；评估仍在完整曲线上进行。

## 平均测试 MAE

| Model | Optimizer | 100M |
| --- | --- | ---: |
| TISSUE | L-BFGS-B | 0.049209 |
| TISSUE | AdamW | 0.198958 |
| MPL | L-BFGS-B | 0.008416 |
| MPL | AdamW | 0.014045 |

## 平均训练 MAE

| Model | Optimizer | 100M |
| --- | --- | ---: |
| TISSUE | L-BFGS-B | 0.021427 |
| TISSUE | AdamW | 0.227205 |
| MPL | L-BFGS-B | 0.012705 |
| MPL | AdamW | 0.017077 |

## 参数对比

### Tissue 最优参数

- `L-BFGS-B`: `L0=1.2265, A=2.0479, alpha=0.1284, C=47.3012, lam=0.9000`
- `AdamW`: `L0=2.4417, A=0.0211, alpha=0.5441, C=99.0555, lam=0.9618`

### MPL 最优参数

- `L-BFGS-B`: `L0=2.6965, A=0.5931, alpha=0.4586, B=300.5582, C=4.3034, beta=5.0000, gamma=5.0000`
- `AdamW`: `L0=2.7563, A=0.5255, alpha=0.5528, B=296.5645, C=1.5458, beta=0.5309, gamma=0.5590`

## 当前观察

- `Tissue` 对优化器非常敏感：`AdamW` 在训练集和测试集上都明显劣于 `L-BFGS-B`。
- `MPL` 也存在优化器影响，但远小于 `Tissue`；两者测试误差同量级，`L-BFGS-B` 更优。
- `Tissue` 的两组最优参数差异很大，说明同一公式在不同优化器下会落到明显不同的参数区域。
- `MPL` 的 `L0/A/B` 变化不大，但 `C/beta/gamma` 差异明显，说明优化器主要改变了对 schedule 响应强度的估计。

## 输出文件

- `results/optimizer_effect/tables/optimizer_effect_metrics.csv`
- `results/optimizer_effect/tables/optimizer_effect_param_runs.csv`
- `results/optimizer_effect/tables/optimizer_effect_best_params.json`
- `results/optimizer_effect/figures/test_mae_summary.png`
- `results/optimizer_effect/figures/train_mae_summary.png`
- `results/optimizer_effect/figures/tissue_best_params.png`
- `results/optimizer_effect/figures/mpl_best_params.png`
- `results/optimizer_effect/figures/examples/`
