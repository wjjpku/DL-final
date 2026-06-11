# docs 索引

本文档按与当前主论文《Learning-Rate Schedules Are Not Adiabatic》的相关性分类。

## core/ — 主线文档（保留）

| 文件 | 内容 |
| --- | --- |
| [core/reproduction_report.md](core/reproduction_report.md) | 课程任务设定下（cosine 拟合、WSD 测试）的复现报告 |
| [core/mpl_official_reproduction.md](core/mpl_official_reproduction.md) | MPL 官方公开划分的严格复现报告 |
| [core/tissue_vs_mpl_official_compare.md](core/tissue_vs_mpl_official_compare.md) | 官方划分上 Tissue 与 MPL 的对比 |
| [core/derivation.md](core/derivation.md) | 从 SGD-谱模型推导 MPL 形式及 SC-MPL 理论 |
| [core/scaling_law_theory.md](core/scaling_law_theory.md) | SC-MPL 的统一公式设计与尺度分离论证 |
| [core/river_valley_derivation.md](core/river_valley_derivation.md) | River-Valley EoS 重新推导与诚实负结果 |
| [core/sc_mpl_report.md](core/sc_mpl_report.md) | SC-MPL 实验报告（含被推翻的早期结论与诚实版结论） |

## explorations/ — 周边诊断与探索

| 文件 | 内容 |
| --- | --- |
| [explorations/optimizer_effect_report.md](explorations/optimizer_effect_report.md) | 优化器（L-BFGS-B vs AdamW）影响对照 |
| [explorations/optimizer_effect_strict_report.md](explorations/optimizer_effect_strict_report.md) | 严格一致协议下的优化器对照 |
| [explorations/weighted_mpl_official_report.md](explorations/weighted_mpl_official_report.md) | 官方 MPL 训练曲线加权实验 |
| [explorations/weighted_scheme_compare.md](explorations/weighted_scheme_compare.md) | 多比例权重方案对比 |
| [explorations/optimized_monotone_schedule.md](explorations/optimized_monotone_schedule.md) | 固定端点下最优单调递减调度搜索 |
| [explorations/continual_schedule_144k.md](explorations/continual_schedule_144k.md) | 72k→144k 连续学习后段调度搜索 |
| [explorations/public_large_model_curves.md](explorations/public_large_model_curves.md) | 公开大模型曲线抓取与可视化 |
| [explorations/train_fit_check.md](explorations/train_fit_check.md) | 训练集拟合检查 |

## 已归档的历史文档

以下文档已被移至 `archive/legacy_docs/`，它们反映项目早期的 proposal 阶段设想，部分内容已被后续工作修正或替代：

- `data_collection.md` — 早期数据收集方案（实际数据来自 MPL 官方仓库与自行训练）
- `experiment_design.md` — 首轮实验设计（方向已迭代）
- `stage1_conclusions.md` — 阶段性结论（部分结论在严格验证后被修正）

原始 proposal 见 `archive/legacy_proposals/task.tex`。
