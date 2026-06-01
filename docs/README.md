# docs 索引

文档按与 proposal 主线的相关性分为两类。

## core/ — 主线文档

直接服务 proposal 三个研究问题与核心复现。

| 文件 | 内容 |
| --- | --- |
| [core/data_collection.md](core/data_collection.md) | 数据收集方案与公开来源清单 |
| [core/experiment_design.md](core/experiment_design.md) | 首轮实验设计（baseline / 辅助特征 / 切分 / 消融） |
| [core/stage1_conclusions.md](core/stage1_conclusions.md) | 阶段性结论与执行建议 |
| [core/reproduction_report.md](core/reproduction_report.md) | cosine 拟合、WSD 测试设定下的复现报告 |
| [core/mpl_official_reproduction.md](core/mpl_official_reproduction.md) | MPL 官方公开划分的严格复现报告 |
| [core/tissue_vs_mpl_official_compare.md](core/tissue_vs_mpl_official_compare.md) | 官方划分上 Tissue 与 MPL 的对比 |

## explorations/ — 周边探索

围绕 backbone 的诊断与 schedule 探索，非 proposal 主线，但记录了有用的经验观察。

| 文件 | 内容 |
| --- | --- |
| [explorations/optimizer_effect_report.md](explorations/optimizer_effect_report.md) | 优化器（L-BFGS-B vs AdamW）影响对照 |
| [explorations/optimizer_effect_strict_report.md](explorations/optimizer_effect_strict_report.md) | 严格一致协议下的优化器对照 |
| [explorations/weighted_mpl_official_report.md](explorations/weighted_mpl_official_report.md) | 官方 MPL 训练曲线加权实验 |
| [explorations/weighted_scheme_compare.md](explorations/weighted_scheme_compare.md) | 多比例权重方案对比 |
| [explorations/weighted_compare_equal_vs_133.md](explorations/weighted_compare_equal_vs_133.md) | 等权 vs 133 加权对比 |
| [explorations/optimized_monotone_schedule.md](explorations/optimized_monotone_schedule.md) | 固定端点下最优单调递减调度搜索 |
| [explorations/continual_schedule_144k.md](explorations/continual_schedule_144k.md) | 72k→144k 连续学习后段调度搜索 |
| [explorations/public_large_model_curves.md](explorations/public_large_model_curves.md) | 公开大模型曲线抓取与可视化 |
| [explorations/train_fit_check.md](explorations/train_fit_check.md) | 训练集拟合检查 |
