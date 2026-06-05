# DL-final

**Learning-Rate Schedules Are Not Adiabatic: A Rate-Dependent Correction for Loss-Curve Prediction**

主论文：`paper/main.tex`（NeurIPS 风格，pdfLaTeX 编译）。

独立复现与扩展：`represent/REPORT.md`（含受控 NQM 模拟 + 真实 ~10M transformer + 5 个扩展 G1–G5）。

---

## 快速开始

```bash
pip install -r requirements.txt          # numpy scipy matplotlib torch tqdm scikit-learn ...
# 数据:MPL 官方公开曲线已随仓库提供于 external/MultiPowerLaw/loss_curve_repo/csv_{25,100,400}
```

## 复现主论文中的每个结果

| 论文 | 命令 | 产物 |
|---|---|---|
| 表1 复现 (cosine→WSD) | `python3 repro/reproduce_cosine_to_wsd.py --scales 25 100 400` | `results/tables/cosine_to_wsd_metrics.csv`, `results/figures/` |
| 图1 / §3 非绝热残差 | `python3 repro/reproduce_cosine_to_wsd.py` + `repro/validate_theory.py` | `results/figures/fig_residual_fit.png` |
| 图2 / §4 τ∝1/η | `python3 repro/universal_collapse.py` | `results/universal_collapse.png` |
| 表2上 / §6.1 SC-MPL 公平对比 | `python3 repro/validate_theory.py` | `results/validate_theory/validation.json` |
| 表2下 / §6.2 Q-MPL 新渐进 | `python3 repro/qmpl.py` | stdout: MPL vs Q-MPL test MAE |
| 表3 / §7 γ essential + 模拟 | `python3 repro/validate_theory.py`(real)+ `python3 repro/sgd_spectrum_sim.py`(sim) | `results/eos_gamma.json` |
| 表4上 / §7 RV-EoS 显式 river-valley 律 | `python3 repro/river_valley.py` | `results/river_valley.log`(RV vs MPL,0/15) |
| 表4下 / §7 sharpness-lag 修正(冻结 MPL) | `python3 repro/river_floor_lag.py` | `results/river_floor_lag.log`(留出 WSD,7/12) |
| §3 完整数学推导 | 见 `docs/core/derivation.md` | — |
| §7 river-valley / Adaptive-EoS 推导 | 见 `docs/core/river_valley_derivation.md` | — |

理论推导文档:[`docs/core/derivation.md`](docs/core/derivation.md)、[`docs/core/scaling_law_theory.md`](docs/core/scaling_law_theory.md)、[`docs/core/river_valley_derivation.md`](docs/core/river_valley_derivation.md)。

## 独立复现（represent/）

`represent/` 目录包含一次完整的**独立对抗复现**：在作者原始数据不可获得的情况下，从第一性原理和真实模型两端验证论文机制。

| 部分 | 内容 | 脚本/报告 |
|---|---|---|
| A. NQM 模拟 | 从零 AdamW 噪声二次模型，复现 τ∝1/η、β₂ 无关性、幅度恒等 | `represent/repro/E*.py`, `represent/results/NQM_REPORT.md` |
| B. 扩展 G1–G5 | λ_slow 慢模诊断、跨形状迁移、有效性边界、双峰谱、动量/WD | `represent/repro/G*.py`, `represent/results/EXTENSIONS_REPORT.md` |
| C. 真实 transformer | 自行训练的 ~10M byte-level Llama-2 风格模型，免拟合验证非绝热滞后 | `represent/repro/train.py`, `represent/repro/analyze_curves.py`, `represent/results/AUDIT_PARTC.md` |
| 综合报告 | 三段独立审计后的最终裁决 | `represent/REPORT.md` |

---

> 历史背景（原始 proposal）：利用更便宜的小模型曲线作为辅助特征改进大模型 loss 预测。该方向（SC-MPL）经严格评估后未超过拟合良好的 MPL，详见 `docs/core/sc_mpl_report.md`。原始 proposal 已归档至 `archive/legacy_proposals/task.tex`。

## 目录说明

- `paper/main.tex`: 主论文（NeurIPS 风格，双栏 article）。
- `slides/main_zh.tex`: 中文答辩幻灯片（ctexbeamer，XeLaTeX 编译）。
- `slides/main.tex`: 英文幻灯片。
- `requirements.txt`: 运行 `repro/` 与 `scripts/` 的 Python 依赖。
- `docs/README.md`: 文档索引。
- `docs/core/`: 主线理论推导与实验报告。
- `docs/explorations/`: 周边诊断实验（加权、优化器、调度搜索等）。
- `repro/`: 主论文复现脚本（48 个 Python 文件）。
- `represent/`: 独立复现代码、结果与报告（对抗审计后）。
- `results/`: 主论文实验输出（图表、预测结果、指标表）。
- `external/`: 第三方 vendored 仓库（含 `MultiPowerLaw` 官方数据与代码）。
- `archive/`: 已归档的历史文档与脚本。

## 已完成工作

### A. 课程任务自定义设定

- 训练集：`cosine_24000.csv`、`cosine_72000.csv`
- 测试集：`wsd_20000_24000.csv`、`wsdld_20000_24000.csv`、`wsdcon_3.csv`、`wsdcon_9.csv`、`wsdcon_18.csv`
- 覆盖尺度：`25M`、`100M`、`400M`

```bash
python3 repro/reproduce_cosine_to_wsd.py --scales 25 100 400
```

### B. MPL 官方仓库公开实验严格复现

```bash
cd external/MultiPowerLaw
python3 main.py --folder_path 25
python3 main.py --folder_path 100
python3 main.py --folder_path 400
```

复现结论：三组公开实验均已成功跑通，平均测试指标与官方 README 基本一致，最优参数一致到三位小数。详见 `docs/core/mpl_official_reproduction.md`。

### C. Tissue vs MPL 官方公开划分对比

- 对比脚本：`repro/compare_tissue_mpl_official.py`
- 输出目录：`results/official_compare/`

### D. 论文口径统一结果目录

- 目录：`results/paper_reproduction/`
- 覆盖 `25M / 100M / 400M` 三个尺度下全部 `9` 个任务

### E–L. 周边诊断实验（加权、G(x) 替换、优化器效应、调度搜索等）

详见本文件历史版本或 `docs/explorations/` 中的各报告。这些实验服务于对 MPL 行为边界的理解，非主论文核心结果。

## 当前论文状态

主论文 `paper/main.tex` 已完成以下核心内容：

1. **问题定义**：MPL/FSL 等现有定律是绝热的，无法表示快速衰减时的非绝热弛豫滞后。
2. **理论推导**：在弱稳定性假设下，推导出线性响应修正项 DropRelaxS；AdamW 特化给出经典 τ∝1/η 与幅度恒等 dL_eq/dη。
3. **实证验证**：在公开 LLM 曲线上确认 τ∝1/η（p=1.00±0.18），用低校准的跨尺度迁移将陡降误差削减 44%。
4. **独立复现**：受控 NQM 模拟（p=0.97，幅度误差 <3%）与真实 ~10M transformer（免拟合验证效应存在）双重确认。
5. **诚实范围**：在数据丰富的 MPL 上修正无益；价值在数据稀缺、领头阶 regime。

### 已知局限与开放问题（论文已诚实声明）

- `τ ∝ 1/η` 在 ~10M 真实 transformer 上**未复现**（τ≈平），与论文"信号随尺度增长"一致。
- `λ_slow` 和 `c` 仍是**测量**而非第一性原理计算（开放问题 (a)）。
- 单条曲线上 DropRelaxS 核与绝热基线**数值简并**；速率依赖只在跨调度对比中真实存在。
- 跨*形状*迁移未通过预注册门槛（G2 负结果）。

## 下一步工作（短期）

1. **统一交付口径**：检查 `paper/`、`slides/`、`README.md` 与 `represent/REPORT.md`，确保 `λ_slow`、`c` 被表述为经验测量/低校准常数，而非第一性原理已预测常数。
2. **固定 DropRelaxS 归一化约定**：论文已区分 raw LR drops 与 `/η_peak` 归一化 drops；下一步需要同步检查脚本、图注和报告，避免 `c≈0.5` 与 `c≈60–86` 混用。
3. **真实模型幅度恒等检验**：在 ~10M 模型上训练 3–4 条 constant LR 曲线，独立测量 `dL_eq/dη`，再与 DropRelaxS 拟合幅度比较。
4. **S-time 操控实验**：改变 batch size 或等效 LR 积分，使 step-time profile 与 S-time profile 解耦，直接检验记忆变量是 `S` 而非步数 `t`。
5. **更大真实模型或谱诊断**：若算力允许，在 ≥25M 尺度验证 `τ ∝ 1/η`；若能记录 Hessian/梯度噪声谱，则直接检验 `λ_slow ≈ 2(λ/s)_eff` 的慢模解释。
