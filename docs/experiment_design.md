# 实验设计

## 研究问题对应假设

### H1

在相同数据、相同架构族、相同 learning-rate schedule 下，加入小模型训练曲线后，大模型 loss curve 预测误差会显著下降。

### H2

不同辅助特征的信息量不同，其中“归一化后的完整小模型曲线”与“前缀曲线 + slope/curvature 统计量”预计最有效。

### H3

辅助特征不仅能提升同 schedule 内预测，也能提升跨 schedule 的泛化，尤其是在 warmup 长度和 decay 形状变化时。

## 任务定义

### 任务 1：最终 loss 预测

输入：

- 目标模型元数据：模型规模、训练进度、schedule 描述
- 可选辅助信息：小模型完整曲线或前缀曲线

输出：

- 目标模型在训练终点的 `final_loss`

### 任务 2：未来轨迹预测

输入：

- 目标模型当前前缀
- 小模型辅助曲线
- schedule 特征

输出：

- 目标模型未来若干步的 loss trajectory

### 任务 3：跨 schedule 泛化

训练：

- 在若干 schedule 上拟合

测试：

- 在未见过的 schedule 形状或 horizon 上预测

## Baseline 设计

### B0: 元数据经验式

只使用：

- 模型规模
- 训练步数或 token 数
- schedule 简单描述

不使用任何辅助曲线。

作用：

- 作为最弱但必要的基线，检验辅助特征是否真的带来增益。

### B1: Annealing-law 风格模型

将 schedule 压缩为少量统计量，例如累计学习率面积与衰减面积，再拟合 loss。

作用：

- 作为轻量、可解释、schedule-aware baseline。

### B2: Multi-power-law 风格模型

显式建模 warmup 与每次 decay 对 loss trajectory 的影响。

作用：

- 作为当前 proposal 最直接的 backbone。

### Ours: Auxiliary-Feature-Conditioned Predictor

在 B1 或 B2 的骨架上，引入小模型辅助特征，对公式参数进行条件化。

候选条件化器：

- 线性回归
- GBDT
- 浅层 MLP

## 辅助特征设计

### F1: 完整小模型曲线

直接输入小模型的归一化 loss curve。

### F2: 小模型前缀曲线

只输入训练早期一段曲线，模拟更便宜的现实场景。

### F3: 统计特征

从小模型曲线提取：

- 初始下降速度
- warmup 结束点斜率
- 最大曲率
- decay 后恢复速度
- 最终稳定段平均斜率

### F4: 尺度与配对元数据

- 小模型参数量
- 目标模型参数量
- scale ratio
- token/parameter ratio

## 数据切分协议

### 切分 1：同来源随机切分

用于验证方法是否能学到基本轨迹规律。

### 切分 2：按模型规模切分

训练使用较小尺度，测试使用较大尺度。

这是验证“小模型辅助大模型”的关键切分。

### 切分 3：按 schedule 切分

训练只看部分 schedule，测试时使用未见过的 schedule。

### 切分 4：按来源切分

例如训练在 `Pythia + OLMo`，测试在 `LLM360`。

用于验证跨项目泛化能力。

## 评价指标

### 点预测指标

- `MAE(final_loss)`
- `RMSE(final_loss)`
- `MAPE(final_loss)`，仅在量纲合适时使用

### 轨迹预测指标

- 整段曲线 MAE
- 未来窗口 MAE
- DTW 距离
- 归一化面积误差

### 排序与选择指标

- rank correlation
- top-k schedule selection accuracy

## 消融实验

至少做以下消融：

1. 去掉辅助特征，只保留 backbone
2. 只保留小模型最终 loss
3. 只保留小模型前缀曲线
4. 只保留 slope/curvature 统计量
5. 不输入 schedule 特征
6. 不输入 scale metadata

## 首轮实验矩阵

### 实验组 A：辅助特征是否有效

- 目标：验证 H1
- 对比：B2 vs B2 + F1 / F2 / F3

### 实验组 B：哪类辅助特征最有效

- 目标：验证 H2
- 对比：F1 / F2 / F3 / F4 的单独与组合效果

### 实验组 C：跨 schedule 泛化

- 目标：验证 H3
- 重点：固定模型族，只改变 warmup、peak LR、decay 形状

### 实验组 D：跨来源泛化

- 目标：验证是否只是在单一公开来源上过拟合

## 预期结果判据

如果以下条件同时满足，可以认为 proposal 得到初步支持：

- 相比 B2，加入辅助特征后 `final_loss` 误差稳定下降
- 在未来轨迹预测中，优势不只出现在训练末期，也出现在中前期 horizon
- 在跨 schedule 设置下，辅助特征模型比纯元数据模型退化更慢

## 执行顺序建议

1. 先做 `Pythia` 上的尺度切分实验
2. 再做 `OLMo` 上的真实公开日志验证
3. 最后做跨来源外测和有限小规模可控实验
