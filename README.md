# DL-final

**Schedule-aware loss-curve prediction**: reproduce Tissue (2024) and the Multi-Power
Law (MPL, Luo 2025), then derive / improve / bound MPL from SGD dynamics.

论文(可在 Overleaf 或本地用 pdfLaTeX 编译):`paper/main.tex`。

## 快速开始

```bash
pip install -r requirements.txt          # numpy scipy matplotlib torch tqdm scikit-learn ...
# 数据:MPL 官方公开曲线已随仓库提供于 external/MultiPowerLaw/loss_curve_repo/csv_{25,100,400}
```

## 复现论文中的每个结果

| 论文 | 命令 | 产物 |
|---|---|---|
| 表1 复现 (cosine→WSD) | `python3 repro/reproduce_cosine_to_wsd.py --scales 25 100 400` | `results/tables/cosine_to_wsd_metrics.csv`, `results/figures/` |
| 图2 / §5.2 universality 塌缩 | `python3 repro/universal_collapse.py` | `results/universal_collapse.png` (R²=0.997) |
| 表2上 / §6.1 SC-MPL 公平对比 | `python3 repro/validate_theory.py` | `results/validate_theory/validation.json` (E-WIN, E-GAMMA, …) |
| 表2下 / §6.2 Q-MPL 新渐进 | `python3 repro/qmpl.py` | stdout: MPL vs Q-MPL test MAE |
| 表3 / §7 γ essential + 模拟 | `python3 repro/validate_theory.py`(real)+ `python3 repro/sgd_spectrum_sim.py`(sim) | `results/eos_gamma.json` |
| 表4上 / §7 RV-EoS 显式 river-valley 律 | `python3 repro/river_valley.py` | `results/river_valley.log`(RV vs MPL,0/15) |
| 表4下 / §7 sharpness-lag 修正(冻结 MPL) | `python3 repro/river_floor_lag.py` | `results/river_floor_lag.log`(留出 WSD,7/12) |
| §4 完整数学推导 | 见 `docs/core/derivation.md` | — |
| §7 river-valley / Adaptive-EoS 推导 | 见 `docs/core/river_valley_derivation.md` | — |

理论推导文档:[`docs/core/derivation.md`](docs/core/derivation.md)、[`docs/core/scaling_law_theory.md`](docs/core/scaling_law_theory.md)、[`docs/core/river_valley_derivation.md`](docs/core/river_valley_derivation.md)(§7 γ 机理:river-valley + Adaptive-EoS,含两个显式律的诚实负结果)。

---

> 历史背景(原始 proposal):利用更便宜的小模型曲线作为辅助特征改进大模型 loss 预测。该方向(SC-MPL)经严格评估后未超过拟合良好的 MPL,详见论文 §6 与 `docs/core/sc_mpl_report.md`。`task.tex` 为原始 proposal。

## 目录说明

- `task.tex`: 原始 proposal。
- `requirements.txt`: 运行 `repro/` 与 `scripts/` 的 Python 依赖。
- `docs/README.md`: 文档索引，按 `core`（主线）与 `explorations`（周边探索）分类。
- `docs/core/data_collection.md`: 数据收集方案与公开来源清单。
- `docs/core/experiment_design.md`: 首轮实验设计。
- `docs/core/stage1_conclusions.md`: 阶段性结论与建议。
- `docs/core/reproduction_report.md`: 按“cosine 拟合、WSD 测试”设置完成的复现实验报告。
- `docs/core/mpl_official_reproduction.md`: 按 `MultiPowerLaw` 官方仓库公开训练/测试划分完成的严格复现报告。
- `docs/core/tissue_vs_mpl_official_compare.md`: 在官方公开划分上对比 `Tissue` 与 `MPL` 的预测结果与可视化。
- `data/public_sources.csv`: 结构化公开来源表，便于后续程序化抓取。
- `repro/reproduce_cosine_to_wsd.py`: 复现实验主脚本。
- `repro/compare_tissue_mpl_official.py`: 在官方公开划分上拟合 `Tissue` 并叠加 `MPL/Tissue/Ground Truth` 对比图。
- `results/paper_reproduction/`: 论文口径统一结果目录，集中存放完整的 `MPL` 原始预测图、`MPL vs Tissue` 对比图和任务索引。
- `results/`: 复现实验输出，包括图表、预测结果和指标表。
- `external/`: 第三方 vendored 仓库（含 `MultiPowerLaw` 官方数据与代码），来源见 `external/README.md`。

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

## 已完成复现

本仓库已在本地完成如下复现实验：

### A. 课程任务自定义设定

- 训练集：`cosine_24000.csv`、`cosine_72000.csv`
- 测试集：`wsd_20000_24000.csv`、`wsdld_20000_24000.csv`、`wsdcon_3.csv`、`wsdcon_9.csv`、`wsdcon_18.csv`
- 覆盖尺度：`25M`、`100M`、`400M`
- 对比方法：`Tissue et al., 2024` 的 annealing-law 与 `Luo et al.` 的 Multi-Power Law

运行方式：

```bash
python3 repro/reproduce_cosine_to_wsd.py --scales 25 100 400
```

主要输出：

- `results/tables/cosine_to_wsd_metrics.csv`
- `results/tables/fitted_params.json`
- `results/figures/*.png`
- `docs/core/reproduction_report.md`

### B. MPL 官方仓库公开实验严格复现

- 官方仓库：`external/MultiPowerLaw`
- 官方训练集：`cosine_24000.csv`、`constant_24000.csv`、`wsdcon_9.csv`
- 官方测试集：`constant_72000.csv`、`cosine_72000.csv`、`wsd_20000_24000.csv`、`wsdld_20000_24000.csv`、`wsdcon_3.csv`、`wsdcon_18.csv`
- 覆盖尺度：`25M`、`100M`、`400M`
- 入口脚本：`external/MultiPowerLaw/main.py`

运行方式：

```bash
cd external/MultiPowerLaw
python3 main.py --folder_path 25
python3 main.py --folder_path 100
python3 main.py --folder_path 400
```

主要输出：

- `external/MultiPowerLaw/logs/25.log`
- `external/MultiPowerLaw/logs/100.log`
- `external/MultiPowerLaw/logs/400.log`
- `external/MultiPowerLaw/25M/fit/`
- `external/MultiPowerLaw/100M/fit/`
- `external/MultiPowerLaw/400M/fit/`
- `docs/core/mpl_official_reproduction.md`

复现结论：

- `25M / 100M / 400M` 三组公开实验均已成功跑通。
- 平均测试指标与官方 README 基本一致。
- 最优参数与官方 README 一致到三位小数。

### C. Tissue vs MPL 官方公开划分对比

- 对比脚本：`repro/compare_tissue_mpl_official.py`
- 数据划分：与官方 `MPL` 公开实验完全一致
- 输出目录：`results/official_compare/`

主要输出：

- `results/official_compare/figures/compare/*.png`
- `results/official_compare/figures/official_avg_test_mae_compare.png`
- `results/official_compare/tables/official_tissue_mpl_metrics.csv`
- `results/official_compare/tables/official_tissue_mpl_params.json`
- `docs/core/tissue_vs_mpl_official_compare.md`

当前结论：

- `25M` 与 `400M` 上 `MPL` 的平均测试误差更低
- `100M` 上 `Tissue` 的平均测试误差更低
- `WSDCon` 两阶段切换曲线多数仍然是 `MPL` 更稳

### D. 论文口径统一结果目录

- 目录：`results/paper_reproduction/`
- 任务索引：`results/paper_reproduction/TASK_INDEX.md`
- `MPL` 官方原始图：`results/paper_reproduction/mpl_only/`
- `MPL + Tissue + Ground Truth` 对比图：`results/paper_reproduction/mpl_vs_tissue_compare/`

说明：

- 该目录覆盖 `25M / 100M / 400M` 三个尺度下全部 `9` 个任务
- 每个尺度包含 `3` 个训练任务和 `6` 个测试任务
- 用于统一查看论文口径的完整预测任务，不再依赖分散在 `external/` 与 `results/official_compare/` 的输出

### E. `G(x)` 替换实验

- 脚本：`repro/compare_g_replacements.py`
- 目标：在保持 `MPL` 其余结构不变的前提下，仅替换 `LD(t)` 中的 `G(x)` 形式；当前支持 `power`、`power+theta`、`hill`、`weibull` 四种响应，其中 `power+theta` 对应 `1 - (1 + C x^theta)^(-beta)`。
- 数据划分：沿用 `MultiPowerLaw` 官方公开训练/测试划分。
- 选择方式：先在训练集三条曲线上做 leave-one-out 选择，再用入选变体在完整训练集上重拟合并评估全部训练/测试曲线。
- 输出目录：`results/g_replacement_official/`

快速试跑：

```bash
python3 repro/compare_g_replacements.py --scales 25 --fit-stride 8 --maxiter 120
```

完整运行：

```bash
python3 repro/compare_g_replacements.py --scales 25 100 400
```

只比较 `pow` 与 `pow_theta`：

```bash
python3 repro/compare_g_replacements.py --scales 25 100 400 --variants pow pow_theta
```

主要输出：

- `results/g_replacement_official/tables/official_g_replacement_metrics.csv`
- `results/g_replacement_official/tables/official_g_replacement_cv.csv`
- `results/g_replacement_official/tables/official_g_replacement_params.json`
- `results/g_replacement_official/figures/`

诊断可视化：

```bash
python3 scripts/plot_gx_diagnostics.py --scale 100 --split test
```

- 输出 `pow` 与 `pow_theta` 的 `x(step)` 摘要曲线图与 loss curve 对比图
- 输出目录：`results/g_replacement_official/diagnostics/`

### F. 优化器影响对照实验

- 脚本：`repro/optimizer_effect_experiment.py`
- 当前实验范围：`100M` 单尺度
- 目的：在同一官方公开训练/测试划分上，单独比较 `L-BFGS-B` 与 `AdamW` 对 `Tissue` 和 `MPL` 的影响
- 输出目录：`results/optimizer_effect/`

主要输出：

- `results/optimizer_effect/tables/optimizer_effect_metrics.csv`
- `results/optimizer_effect/tables/optimizer_effect_best_params.json`
- `results/optimizer_effect/figures/test_mae_summary.png`
- `results/optimizer_effect/figures/train_mae_summary.png`
- `results/optimizer_effect/figures/tissue_best_params.png`
- `results/optimizer_effect/figures/mpl_best_params.png`
- `results/optimizer_effect/figures/examples/`
- `docs/explorations/optimizer_effect_report.md`

当前结论：

- `Tissue` 对优化器非常敏感，`L-BFGS-B` 明显优于 `AdamW`
- `MPL` 也存在优化器差异，但幅度小于 `Tissue`
- 两个模型的最优参数向量都会随优化器改变，其中 `Tissue` 的变化更剧烈

### G. 严格一致协议下的优化器对照实验

- 脚本：`repro/optimizer_effect_strict.py`
- 实验范围：`100M` 单尺度
- 目标：消除上一版轻量实验中的协议差异，只在完全一致的设置下比较优化器
- 输出目录：`results/optimizer_effect_strict/`

一致设置：

- 相同官方数据加载方式：直接复用 `external/MultiPowerLaw/src/data_loader.py`
- 相同官方训练/测试划分
- 相同完整训练点与完整评估点，不做下采样
- 相同目标函数：`log(loss)` 残差上的 Huber 损失
- 相同初始化，再分别交给 `AdamW` 与 `L-BFGS-B`
- `MPL + AdamW` 直接读取官方 `external/MultiPowerLaw/logs/100.log` 中的最优参数，保证与官方训练协议一致

运行方式：

```bash
python3 repro/optimizer_effect_strict.py
```

主要输出：

- `results/optimizer_effect_strict/tables/strict_optimizer_metrics.csv`
- `results/optimizer_effect_strict/tables/strict_optimizer_best_params.json`
- `results/optimizer_effect_strict/figures/test_mae_summary.png`
- `results/optimizer_effect_strict/figures/train_mae_summary.png`
- `results/optimizer_effect_strict/figures/tissue_best_params.png`
- `results/optimizer_effect_strict/figures/mpl_best_params.png`
- `results/optimizer_effect_strict/figures/examples/`
- `docs/explorations/optimizer_effect_strict_report.md`

当前结论：

- `MPL + AdamW` 在严格一致协议下恢复正常，平均测试 MAE 为 `0.004348`
- `MPL + L-BFGS-B` 略优，平均测试 MAE 为 `0.003573`
- `Tissue + L-BFGS-B` 仍显著优于 `Tissue + AdamW`，平均测试 MAE 分别为 `0.003794` 与 `0.015296`
- 因此，上一次扭曲结果的主要原因确实是训练协议不一致；在保持一致后，`MPL` 对优化器较稳，而 `Tissue` 仍高度敏感

### H. 官方 MPL 训练曲线加权实验

- 脚本：`repro/weighted_mpl_official.py`
- 实验范围：`25M`、`100M`、`400M`
- 目标：保持官方 `MPL` 初始化、模型结构、`AdamW` 超参数、训练/测试划分和评估流程不变，只修改训练集三条曲线在目标函数中的聚合权重
- 输出目录：`results/weighted_mpl_official/`

方案定义：

- `equal_1_1_1`：等权基线
- `cosine_constant_3_3_1`：偏重 `cosine_24000` 和 `constant_24000`
- `cosine_wsdcon_3_1_3`：偏重 `cosine_24000` 和 `wsdcon_9`
- `constant_wsdcon_1_3_3`：偏重 `constant_24000` 和 `wsdcon_9`

说明：

- 所有 `3+3+1` 方案都先归一化到总权重为 `3`，避免只因为总梯度尺度变化而影响比较
- 训练集和测试集的预测图、每个方案的 `loss_monitor.png` 都已保留

运行方式：

```bash
python3 repro/weighted_mpl_official.py
```

主要输出：

- `results/weighted_mpl_official/tables/weighted_scheme_metrics.csv`
- `results/weighted_mpl_official/tables/weighted_scheme_summary.csv`
- `results/weighted_mpl_official/tables/weighted_scheme_overall_ranking.csv`
- `results/weighted_mpl_official/figures/avg_test_mae_by_scheme.png`
- `results/weighted_mpl_official/<scale>M/<scheme>/loss_monitor.png`
- `results/weighted_mpl_official/<scale>M/<scheme>/train/*.png`
- `results/weighted_mpl_official/<scale>M/<scheme>/test/*.png`
- `docs/explorations/weighted_mpl_official_report.md`

当前结论：

- `constant_wsdcon_1_3_3` 是跨尺度最优方案，跨 `25M/100M/400M` 的平均测试 MAE 为 `0.003607`
- 等权基线排第二，平均测试 MAE 为 `0.004314`
- 其余两种偏重 `cosine_24000` 的 `3+3+1` 方案都更差
- 因此，若只允许在官方三条训练曲线上做偏心加权，优先提高 `constant_24000` 与 `wsdcon_9` 权重、降低 `cosine_24000` 权重更有利于测试泛化

### I. 多比例权重对比图

- 脚本：`repro/compare_weight_schemes.py`
- 实验范围：`111 / 133 / 144 / 122 / 124 / 142`
- 顺序约定：比例始终对应训练集 `[cosine_24000, constant_24000, wsdcon_9]`
- 输出目录：`results/weighted_scheme_compare/`

主要输出：

- `results/weighted_scheme_compare/figures/avg_test_mae_all_schemes.png`
- `results/weighted_scheme_compare/figures/25M_test_curve_mae_compare.png`
- `results/weighted_scheme_compare/figures/100M_test_curve_mae_compare.png`
- `results/weighted_scheme_compare/figures/400M_test_curve_mae_compare.png`
- `results/weighted_scheme_compare/<scale>M/curve_compare/*.png`
- `results/weighted_scheme_compare/tables/curve_metrics.csv`
- `results/weighted_scheme_compare/tables/summary.csv`
- `results/weighted_scheme_compare/tables/overall_ranking.csv`
- `docs/explorations/weighted_scheme_compare.md`

当前结论：

- 跨尺度总排名为：`133 > 124 > 144 > 111 > 122 > 142`
- `133` 仍然是整体最优，平均测试 MAE 为 `0.003607`
- `124` 在 `100M` 上几乎追平 `144` 和 `133`，但在 `25M/400M` 上不如 `133` 稳
- `144` 在 `100M` 单尺度上最好，但在 `400M` 上明显退化
- `142` 在 `100M` 上出现明显失稳，不适合作为通用权重方案

### J. 固定端点下的最优单调递减调度搜索

- 脚本：`repro/optimize_monotone_schedule.py`
- 使用模型：当前最优的 `133` 加权 `MPL`
- 固定约束：
  - 初始学习率固定为 `3e-4`
  - 最终学习率固定为 `3e-5`
  - 总步数固定为 `24000`
  - 调度必须单调不增
  - 初始调度为线性递减
- 输出目录：`results/optimized_monotone_schedule/`

端点来源：

- `lr_max = 3e-4` 取自 `133` 训练集 `[cosine_24000, constant_24000, wsdcon_9]` 的共同峰值学习率
- `lr_min = 3e-5` 取自这三条训练曲线里最小的终点学习率，即 `cosine_24000` 的尾部
- 这里不把 warmup 起点的 `0` 当作 `lr_min`

优化方法：

- 用 `64` 个控制点参数化整条 schedule
- 控制点之间做线性插值
- 直接最小化代理模型预测的最终一步 loss

主要输出：

- `results/optimized_monotone_schedule/summary.csv`
- `results/optimized_monotone_schedule/<scale>M/schedule_compare.png`
- `results/optimized_monotone_schedule/<scale>M/predicted_curve_compare.png`
- `results/optimized_monotone_schedule/<scale>M/optimization_history.png`
- `results/optimized_monotone_schedule/<scale>M/schedule_compare.csv`
- `results/optimized_monotone_schedule/<scale>M/predicted_curve_compare.csv`
- `docs/explorations/optimized_monotone_schedule.md`

当前结论：

- 相对线性递减初值，优化后的最终预测 loss 分别改善：
  - `25M`: `1.73%`
  - `100M`: `2.15%`
  - `400M`: `2.45%`
- 三个尺度的最优 schedule 都偏向“尽量晚降”
- 例如 `25M` 直到约第 `21835` 步才首次明显低于高学习率区间，说明当前代理模型更偏好长时间保持接近 `lr_max`，在尾部再快速下探到 `lr_min`
- 这是一项代理模型上的 schedule search 结果，不等同于真实训练最优解

### K. 公开大参数模型曲线抓取与可视化

- 脚本：`scripts/visualize_public_large_model_curves.py`
- 输出目录：`results/public_large_model_curves/`
- 图像形式：每个来源一张双轴图，同图展示 `loss curve` 和 `lr schedule curve`
- 标题格式：`任务类型 | 模型名 | 参数量`

当前已稳定抓取并可视化的公开来源：

- `OLMoE SFT`：`1.3B active / 6.9B total`，官方 GitHub 原始日志，逐步 `LR/Loss`
- `OLMoE DPO`：`1.3B active / 6.9B total`，官方 GitHub 原始日志，逐步 `LR/Loss`
- `Zephyr-7B alpha`：`7B`，Hugging Face 原始模型卡训练表，`LR` 由公开超参数重建
- `Zephyr-7B beta`：`7B`，Hugging Face 原始模型卡训练表，`LR` 由公开超参数重建
- `Mistral-7B SFT beta`：`7B`，Hugging Face 原始模型卡训练表，但只公开了 `1` 个训练点

主要输出：

- `results/public_large_model_curves/olmoe_sft.png`
- `results/public_large_model_curves/olmoe_dpo.png`
- `results/public_large_model_curves/zephyr_7b_alpha.png`
- `results/public_large_model_curves/zephyr_7b_beta.png`
- `results/public_large_model_curves/mistral_7b_sft_beta.png`
- `results/public_large_model_curves/*.csv`
- `results/public_large_model_curves/manifest.json`
- `docs/explorations/public_large_model_curves.md`

说明：

- `OLMoE` 两条曲线最完整，因为原始日志里直接有逐步 `Step/LR/Loss`
- `Zephyr` 和 `Mistral` 来自 Hugging Face 自动生成模型卡，`Loss` 是公开训练表，`LR` 依据公开 `learning_rate + scheduler + warmup` 重建
- 其中 `Mistral-7B SFT beta` 只公开了一个训练采样点，因此可视化有效信息明显少于其他来源

### L. 72k 到 144k 的连续学习后段调度搜索

- 脚本：`repro/optimize_continual_schedule.py`
- 输出目录：`results/continual_schedule_144k/`
- 使用模型：当前最优的 `133` 加权 `MPL`

问题设定：

- 先重新学习一个 `0-72k` 的前段最优单调递减 schedule，作为固定前缀
- 把前 `72k` 的预测 loss 视为连续学习问题的历史前段
- 在 `72k-144k` 段重新设计学习率
- 后段初始化：全程 `lr_min = 3e-5`
- 固定边界：第 `72k` 点与第 `144k` 点都固定为 `lr_min`
- 中间位置允许在 `[lr_min, lr_max]` 内上抬，形成可能的再加热 schedule

结果总表：

- `25M`: `72k` 前缀末端预测 loss `3.068746`，后段全最小 lr 基线 `3.059047`，优化后 `3.035147`，改善 `0.78%`
- `100M`: `72k` 前缀末端预测 loss `2.693481`，后段全最小 lr 基线 `2.681771`，优化后 `2.652155`，改善 `1.10%`
- `400M`: `72k` 前缀末端预测 loss `2.423558`，后段全最小 lr 基线 `2.409484`，优化后 `2.376802`，改善 `1.36%`

形状观察：

- 三个尺度的最优后段都不是停在 `lr_min`
- 它们都会在 `72k` 之后快速再加热到接近 `lr_max = 3e-4`
- 后段峰值都出现在大约第 `72759` 步，峰值约 `2.9994e-4`
- 因此在当前代理模型下，最优解更接近“重新拉高学习率继续训练，最后再回落到最小值”

主要输出：

- `results/continual_schedule_144k/<scale>M/continual_schedule_full.png`
- `results/continual_schedule_144k/<scale>M/continual_schedule_suffix_zoom.png`
- `results/continual_schedule_144k/<scale>M/continual_predicted_curve_full.png`
- `results/continual_schedule_144k/400M/continual_schedule_window_57k_72k.png`
- `results/continual_schedule_144k/400M/continual_schedule_window_129k_144k.png`
- `results/continual_schedule_144k/<scale>M/continual_schedule_compare.csv`
- `results/continual_schedule_144k/<scale>M/continual_predicted_curve_compare.csv`
- `docs/explorations/continual_schedule_144k.md`

## 下一步执行

- 补充脚本，抓取公开日志与 checkpoint 元数据。
- 先形成最小训练样本表，再决定采用解析公式还是浅层回归器。
- 在小规模可控实验中复现 warmup / decay / horizon 外推三类现象。
