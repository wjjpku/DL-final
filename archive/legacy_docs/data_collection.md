# 数据收集方案

## 目标

本项目需要的不是单纯“有模型权重”的公开仓库，而是“带训练过程信息”的公开来源。优先收集以下三类信息：

- 轨迹轴：`step`、`tokens_seen`、`epoch_fraction`
- 损失指标：`train_loss`、`val_loss`、`perplexity`
- schedule 描述：`warmup_steps`、`peak_lr`、`decay_type`、`total_steps`

## 统一 schema

建议将所有公开来源统一为如下字段：

| 字段 | 含义 |
| --- | --- |
| `source_name` | 数据来源，如 OLMo / Pythia |
| `model_family` | 模型家族 |
| `run_id` | 训练 run 唯一标识 |
| `scale_label` | 尺度标签，如 160M / 1B / 7B |
| `schedule_type` | cosine / linear / WSD / custom |
| `warmup_steps` | warmup 长度 |
| `peak_lr` | 峰值学习率 |
| `step` | 训练步数 |
| `tokens_seen` | 已见 token 数 |
| `train_loss` | 训练损失 |
| `val_loss` | 验证损失 |
| `perplexity` | 困惑度 |
| `aux_curve_available` | 是否有可用的小模型辅助曲线 |
| `notes` | 清洗或字段备注 |

## 公开来源优先级

### 第一梯队

1. `OLMo`
   - 优点：公开日志和 checkpoint 较完整，适合直接抽取轨迹监督样本。
   - 用途：主训练源之一，重点用于 schedule-aware curve prediction。

2. `LLM360`
   - 优点：强调全过程公开，可能额外包含 gradient norm 等有价值特征。
   - 用途：做 richer trajectory feature 的补充源。

3. `Pythia`
   - 优点：控制变量强，尺度族清晰，checkpoint 密。
   - 用途：做跨尺度迁移和小模型辅助特征验证。

### 第二梯队

4. `BLOOM`
   - 用途：外部泛化测试，验证方法在超大模型单次训练上的鲁棒性。

5. `TinyLlama`
   - 用途：补充长 token、小规模模型的中间 checkpoint 轨迹。

6. `OpenLM`
   - 用途：用于稀疏采样曲线补全测试。

## 数据采集策略

### 阶段 A：只采元数据与可确认轨迹字段

目标是先建立一张可靠索引表，而不是一开始就追求完整原始日志。

- 记录每个来源是否公开：
  - 可下载日志
  - 可访问 W\&B 历史
  - 可通过 checkpoint/revision 恢复训练进度
- 记录每个来源的：
  - 模型规模集合
  - checkpoint 频率
  - 训练 token 范围
  - schedule 是否可识别

### 阶段 B：落地统一的轨迹表

按来源分别写抓取器，将结果转为统一长表：

| source_name | run_id | scale_label | step | tokens_seen | train_loss | val_loss |
| --- | --- | --- | --- | --- | --- | --- |

优先原则：

- 先收集 `train_loss + step/tokens_seen`
- 再补 `val_loss/perplexity`
- 最后补 `schedule` 细节与附加特征

### 阶段 C：为辅助特征建配对关系

项目核心不是单独拟合大模型曲线，而是建立“小模型 -> 大模型”的配对样本。每条样本应额外维护：

- `target_run_id`
- `aux_run_ids`
- `shared_dataset`
- `shared_arch_family`
- `shared_schedule`
- `scale_ratio`

## 数据清洗原则

- 统一横轴：优先使用 `tokens_seen`，缺失时退回 `step`。
- 统一 loss：若来源只给 perplexity，则保留原字段并标记不可直接比较。
- 去除异常点：明显的日志重启点、重复 step、断裂 checkpoint 单独标记。
- 统一 schedule 名称：将来源自定义命名映射到 `cosine`、`linear_decay`、`wsd`、`custom_piecewise`。

## 最小可执行数据包

第一周内应至少形成如下最小数据包：

- `OLMo`: 至少 1 个中大尺度 run
- `Pythia`: 至少 3 个不同尺度 run
- `LLM360`: 至少 1 个可确认有训练曲线的 run

形成后即可支持：

- 跨尺度曲线可视化
- 曲线归一化形状分析
- 简单辅助特征回归

## 风险与应对

- 风险 1：公开日志字段命名不统一
  - 应对：使用来源专属 parser，再映射到统一 schema

- 风险 2：部分来源只提供 checkpoint 不提供 loss 历史
  - 应对：先登记为“可恢复来源”，后续必要时离线重算 loss

- 风险 3：schedule 信息不完整
  - 应对：优先选择论文或模型卡中明确给出 schedule 的来源作为主实验集
