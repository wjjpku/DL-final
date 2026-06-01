# 固定端点下的最优单调递减调度搜索

本实验使用当前最优预测器：`133` 加权方案下的 `MPL` 参数。

约束条件：

- 固定初始学习率 `lr_max = 3e-4`
- 固定最终学习率 `lr_min = 3e-5`
- 固定总步数 `24000`
- 调度必须单调不增
- 初始化为线性递减调度

其中，`lr_max` 与 `lr_min` 都取自 `133` 模型对应训练集 `[cosine_24000, constant_24000, wsdcon_9]` 的端点范围：

- `lr_max = 3e-4`：三条训练曲线的共同峰值学习率
- `lr_min = 3e-5`：三条训练曲线中最小的终点学习率，来自 `cosine_24000`

说明：这里的 `lr_min` 指的是训练 schedule 的尾部最小学习率，而不是 warmup 起点的 `0`。

优化方法：

- 用 `64` 个控制点参数化整条单调 schedule
- 控制点之间线性插值，保证起点和终点精确满足约束
- 直接最小化 `MPL` 预测的最终一步 loss

## 结果总表

| Scale | Linear Final Loss | Optimized Final Loss | Improvement |
| --- | ---: | ---: | ---: |
| 25M | 3.212807 | 3.157283 | 1.73% |
| 100M | 2.858758 | 2.797192 | 2.15% |
| 400M | 2.602627 | 2.538974 | 2.45% |

## 形状观察

- 三个尺度的最优解都不是更早退火，而是明显偏向“尽量晚降”。
- `25M` 的最优 schedule 到大约第 `21835` 步才首次低于 `0.9 * lr_max + 0.1 * lr_min`，约占全程 `90.98%`。
- `100M` 的对应位置是第 `21751` 步，约占全程 `90.63%`。
- `400M` 的对应位置是第 `21355` 步，约占全程 `88.98%`。
- 因此，在“端点固定、单调递减、总步数固定”的约束下，当前最优 `133-MPL` 代理模型更偏好高学习率维持更久、尾部快速下落的调度。

## 解释边界

- 这是在代理模型上的 schedule search，不是真实训练曲线上的直接最优控制。
- 目标函数只最小化最终一步预测 loss，因此自然会更偏向改善 tail，而不保证整个训练过程最优。
- 本实验没有额外加入平滑度、二阶导或面积约束；如果继续收紧先验，最优调度形状可能会改变。

## 输出目录

- `results/optimized_monotone_schedule/<scale>M/schedule_compare.png`
- `results/optimized_monotone_schedule/<scale>M/predicted_curve_compare.png`
- `results/optimized_monotone_schedule/<scale>M/optimization_history.png`
- `results/optimized_monotone_schedule/<scale>M/schedule_compare.csv`
- `results/optimized_monotone_schedule/<scale>M/predicted_curve_compare.csv`
