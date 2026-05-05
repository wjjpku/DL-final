# DL-final

本仓库用于推进一项研究型 proposal：利用更便宜的小模型训练曲线作为辅助特征，提升大模型预训练 loss curve prediction，尤其关注对 learning-rate schedule 变化的感知能力。

## 研究目标

- 预测目标：更准确地预测大模型在预训练中的未来 loss trajectory 与最终 loss。
- 核心假设：在相同数据、相同架构族、相同 schedule 下，小模型曲线中包含可迁移的优化动力学信息。
- 方法偏好：以可解释、轻量的经验公式为骨架，再用辅助特征对公式参数进行条件化。

## 当前阶段

当前仓库只包含 proposal 源文件 `task.tex`。基于 proposal，已经完成第一阶段研究启动工作：

- 整理了公开训练曲线数据源优先级与采集 schema。
- 设计了首轮实验矩阵、baseline、评价指标与消融方案。
- 给出了阶段性研究判断与执行建议。

## 目录说明

- `task.tex`: 原始 proposal。
- `docs/data_collection.md`: 数据收集方案与公开来源清单。
- `docs/experiment_design.md`: 首轮实验设计。
- `docs/stage1_conclusions.md`: 阶段性结论与建议。
- `data/public_sources.csv`: 结构化公开来源表，便于后续程序化抓取。

## 当前建议路线

优先构建一个统一 schema：

`model_family, run_id, scale, schedule_type, step, tokens_seen, train_loss, val_loss, perplexity`

然后按下面顺序推进：

1. 先采 `OLMo`、`LLM360`、`Pythia` 三类公开源，建立统一表。
2. 先做不依赖目标模型前缀的预测，再逐步加入目标模型的早期 prefix 作为增强项。
3. 先验证 schedule-aware backbone 是否成立，再验证 small-model auxiliary feature 是否稳定带来增益。

## 首轮实验重点

- Baseline 1: 不使用辅助特征的 step-only / scale-only 经验曲线模型。
- Baseline 2: schedule-aware 的 annealing-law 风格模型。
- Baseline 3: multi-power-law 风格模型。
- Ours: 在上述 backbone 上加入小模型曲线特征条件化。

## 下一步执行

- 补充脚本，抓取公开日志与 checkpoint 元数据。
- 先形成最小训练样本表，再决定采用解析公式还是浅层回归器。
- 在小规模可控实验中复现 warmup / decay / horizon 外推三类现象。
