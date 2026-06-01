# 72k 到 144k 的连续学习后段调度搜索

本实验继续使用当前最优预测器：`133` 加权方案下的 `MPL` 参数。

## 设定

- 先单独学习一个 `0-72k` 的前段最优单调递减 schedule，约束仍为 `lr_max = 3e-4` 到 `lr_min = 3e-5`。
- 将这条 `72k` 前缀固定，并把它对应的预测 loss 作为后续连续学习问题的历史前段。
- 在 `72k-144k` 段，只优化新的后段 schedule。
- 后段初始化为全程 `lr_min = 3e-5`。
- 后段第 `72k` 点与第 `144k` 点都固定为 `lr_min`。
- 后段中间位置允许在 `[lr_min, lr_max]` 内上抬，从而形成可能的再加热形状。

## 目标

- 以前 `72k` 固定前缀为条件，最小化第 `144k` 步的代理预测 loss。
- 对照基线是：前 `72k` 使用同一最优前缀，后 `72k` 全部维持 `lr_min` 不变。

## 结果总表

| Scale | Pred Loss @72k Prefix End | Pred Loss @144k All-min Suffix | Pred Loss @144k Optimized Suffix | Improvement |
| --- | ---: | ---: | ---: | ---: |
| 25M | 3.068746 | 3.059047 | 3.035147 | 0.78% |
| 100M | 2.693481 | 2.681771 | 2.652155 | 1.10% |
| 400M | 2.423558 | 2.409484 | 2.376802 | 1.36% |

## 形状观察

- 三个尺度的最优后段都不是一直贴着 `lr_min`，而是会在 `72k` 之后很快重新抬升到接近 `lr_max`。
- `25M`、`100M`、`400M` 的后段峰值都出现在大约第 `72759` 步，峰值约为 `2.9994e-4`，几乎回到 `lr_max = 3e-4`。
- 这说明在当前代理模型下，如果允许 `72k-144k` 中间段自由再加热，那么最优解更接近“重新拉高学习率继续训练，最后再回落到最小值”，而不是“从 `72k` 之后一直维持极低学习率”。
- 与“后段全最小学习率”相比，优化后后段在三个尺度上都带来一致但不算巨大的改进，改善幅度约为 `0.78% ~ 1.36%`。

## 输出目录

- `results/continual_schedule_144k/<scale>M/prefix_schedule_72k.csv`
- `results/continual_schedule_144k/<scale>M/continual_schedule_compare.csv`
- `results/continual_schedule_144k/<scale>M/continual_predicted_curve_compare.csv`
- `results/continual_schedule_144k/<scale>M/continual_schedule_full.png`
- `results/continual_schedule_144k/<scale>M/continual_schedule_suffix_zoom.png`
- `results/continual_schedule_144k/<scale>M/continual_predicted_curve_full.png`

## 解释边界

- 这是在原有 `133-MPL` 代理模型上的 schedule search，不是带真实新数据重新训练后的真实最优控制。
- 这里的“新一批数据进来”被代理成：前 `72k` 训练历史固定，后 `72k` 允许重新设计学习率曲线。
- 本实验没有显式建模数据分布变化、遗忘约束或额外正则，只研究给定代理损失下的最优后段学习率形状。
