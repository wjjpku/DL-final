# Loss Curve Prediction 复现实验报告

## 1. Problem Introduction

大语言模型预训练极其昂贵，而学习率调度器会显著影响整个训练过程中的 loss trajectory。传统 scaling law 更关注训练终点的单个 loss 点，但在真实研发中，更有价值的问题往往是：

- 能否在训练尚未结束时预测完整 loss curve；
- 能否利用少量已知 schedule 的曲线，外推出未见过 schedule 的训练表现；
- 能否据此提前比较不同学习率调度器，减少昂贵试错。

这个问题的重要性主要体现在三方面：

1. **降低算力成本**
   - 如果能只用少量已有曲线拟合经验公式，再外推到新 schedule，就能减少重复预训练实验。

2. **提升训练可控性**
   - 研究者不仅关心最终 loss，还关心中期 loss drop、annealing 带来的额外收益，以及不同 horizon 下的训练动态。

3. **服务 schedule 设计**
   - 如果公式既能拟合曲线、又能外推到 WSD 等未见过 schedule，就能进一步用于反推更优 schedule。

本次实验聚焦两类代表性方法：

- **Tissue et al., 2024**: `Scaling Law with Learning Rate Annealing`
- **Luo et al.** 的 **Multi-Power Law**

核心复现任务是：

- 只用 `cosine` 学习率调度下的 loss curves 拟合模型；
- 在 `WSD` 及其变体曲线上评估预测性能。

## 2. 论文与方法

### 2.1 Tissue et al., 2024

该方法提出一个显式包含 annealing 影响的公式：

```text
L(s) = L0 + A * S1^(-alpha) - C * S2
```

其中：

- `S1` 是到当前 step 为止的累计学习率面积；
- `S2` 是带 annealing 动量衰减的面积项；
- 直觉上，`S1` 描述“总体训练推进量”，`S2` 描述“学习率衰减带来的额外 loss reduction”。

它的优势是：

- 公式简单；
- 可解释性强；
- 参数少，拟合容易。

### 2.2 Luo et al., Multi-Power Law

Multi-Power Law 使用更强的非线性 loss-drop 项：

```text
L(t) = L0 + A * (S1(t))^(-alpha) + B * LD(t)
```

其中 `LD(t)` 进一步考虑每次学习率变化后，对后续损失下降的累计影响，并加入形如 `C, beta, gamma` 的非线性参数。

相比 Tissue 公式，它的特点是：

- 表达能力更强；
- 更适合建模复杂的学习率衰减效应；
- 参数更多，拟合更重，但跨 schedule 外推通常更强。

## 3. 数据与实验设置

### 3.1 数据来源

本次使用的数据来自：

- 你提供的 PKU 网盘临时链接；
- `MultiPowerLaw` 官方仓库中的 `loss_curve_repo/csv_{25,100,400}`。

实际检查后，`MultiPowerLaw` 仓库已经自带以下曲线文件：

- `cosine_24000.csv`
- `cosine_72000.csv`
- `wsd_20000_24000.csv`
- `wsdld_20000_24000.csv`
- `wsdcon_3.csv`
- `wsdcon_9.csv`
- `wsdcon_18.csv`

分别覆盖 3 种模型尺度：

- `25M`
- `100M`
- `400M`

### 3.2 训练/测试划分

按照你的要求，本次复现实验采用：

- **训练集**
  - `cosine_24000.csv`
  - `cosine_72000.csv`

- **测试集**
  - `wsd_20000_24000.csv`
  - `wsdld_20000_24000.csv`
  - `wsdcon_3.csv`
  - `wsdcon_9.csv`
  - `wsdcon_18.csv`

也就是说，模型**只看 cosine 曲线拟合参数**，然后**完全外推到 WSD 家族曲线**。

### 3.3 硬件与运行环境

本地运行环境为：

- macOS
- Apple Silicon `arm64`
- Python `3.12.10`
- PyTorch `2.11.0`
- `MPS available = True`

为保证稳定性，本次复现实验脚本使用 `NumPy + SciPy + Matplotlib` 的 CPU 方案，不依赖 CUDA；因此可直接在你的 M5 Mac 上复现。

## 4. 复现实现

### 4.1 复现实验脚本

主脚本：

- [reproduce_cosine_to_wsd.py](file:///Users/jiaju/Documents/github/DL-final/repro/reproduce_cosine_to_wsd.py)

它完成了以下工作：

- 加载 25M / 100M / 400M 曲线；
- 重建 cosine / WSD / WSDLD / WSDCon 的学习率序列；
- 拟合 Tissue 公式；
- 拟合 Multi-Power Law 公式；
- 输出逐曲线指标、逐曲线图片、参数文件。

### 4.2 Apple Silicon 上的实现取舍

为了让实验在本地 M5 上可控完成，我做了两点工程化处理：

1. `Tissue` 中的 `S2` 从朴素高复杂度实现改为线性递推；
2. 拟合阶段对 `cosine` 训练曲线做了稀疏采样，但**评估仍在完整测试曲线上完成**。

这两点不会改变实验目标：

- 训练仍然只用 cosine；
- 测试仍然在 WSD-family 完整曲线上。

## 5. 评价指标

本次记录指标包括：

- `MAE`
- `RMSE`
- `MAPE`
- `R2`
- `Huber(log residual)`

其中最重要的是：

- `MAE`：直接衡量预测误差；
- `R2`：衡量曲线拟合优度；
- `RMSE`：对较大偏差更敏感。

## 6. 主要结果

### 6.1 主任务：Cosine 拟合，WSD 测试

`wsd_20000_24000.csv` 上的结果如下：

| Scale | Model | MAE | RMSE | R2 |
| --- | --- | ---: | ---: | ---: |
| 25M | Tissue | 0.00696 | 0.00800 | 0.99728 |
| 25M | MPL | **0.00430** | **0.00518** | **0.99886** |
| 100M | Tissue | 0.00436 | 0.00653 | 0.99838 |
| 100M | MPL | **0.00434** | **0.00615** | **0.99856** |
| 400M | Tissue | 0.00864 | 0.01189 | 0.99506 |
| 400M | MPL | **0.00728** | **0.01069** | **0.99601** |

结论：

- 在你指定的主任务 `fit on cosine, test on WSD` 上，**Multi-Power Law 在 3 个尺度上都优于 Tissue 公式**。

### 6.2 所有 WSD-family 测试曲线的平均表现

各尺度在 5 条测试曲线上的平均 `MAE`：

| Scale | Tissue Avg MAE | MPL Avg MAE | Better |
| --- | ---: | ---: | --- |
| 25M | 0.00783 | **0.00539** | MPL |
| 100M | **0.00486** | 0.00496 | Tissue 略优 |
| 400M | 0.00979 | **0.00852** | MPL |

结论：

- `25M` 和 `400M` 上，MPL 的整体优势明显；
- `100M` 上，两者接近，Tissue 的平均 MAE 略低，但差距很小；
- 综合三种尺度，**MPL 的跨 schedule 泛化更稳定**。

### 6.3 按测试曲线逐个比较

按 `MAE` 比较每个尺度 5 条测试曲线的胜负：

| Scale | Tissue Wins | MPL Wins |
| --- | ---: | ---: |
| 25M | 0 | 5 |
| 100M | 1 | 4 |
| 400M | 1 | 4 |

结论：

- MPL 在大多数测试曲线上占优；
- Tissue 主要在个别 `wsdcon` 两阶段曲线上还能保持竞争力。

## 7. 结果分析

### 7.1 为什么 MPL 在主任务上更强

这次任务的关键不只是“拟合一条曲线”，而是：

- 只看 `cosine`；
- 外推到**形状明显不同**的 `WSD`。

Tissue 公式把 annealing 影响压缩到一个较简单的 `S2` 项中，因此：

- 在训练曲线整体比较平滑时，它非常有效；
- 但当测试 schedule 的衰减结构更复杂时，表达能力会受限。

MPL 的 `LD(t)` 对“每次学习率变化后如何影响后续 loss”建模得更细，因此更适合：

- 从 `cosine` 外推到 `WSD`；
- 从一种 decay 机制外推到另一种更复杂的 decay 机制。

### 7.2 为什么 100M 上两者差距最小

100M 结果显示两种公式都已经相当准确，说明这个尺度上：

- 曲线规律足够稳定；
- Tissue 的简单公式已经能捕捉大部分变化；
- MPL 的额外自由度未必总能转化为显著收益。

这也是一个重要结论：

- **更复杂的公式并不总是显著更好**；
- 但在跨 schedule 外推更难的设置上，MPL 的收益更稳定。

### 7.3 难点曲线：WSDCon-3

对三个尺度来说，`wsdcon_3.csv` 往往是误差最大的测试曲线之一，说明：

- 两阶段学习率切换带来的动态变化更陡；
- 从 `cosine` 外推到这种“突变式” schedule 更困难；
- 即使是 MPL，在该曲线上也会出现明显误差上升。

这与论文动机是一致的：真正困难的不是插值，而是**跨 schedule shape 的外推**。

## 8. 结论

本次复现实验支持以下结论：

1. **问题本身重要且成立**
   - loss curve prediction 在预训练中有直接计算价值；
   - 仅从少量已知 schedule 曲线外推未见 schedule 是一个高价值任务。

2. **Tissue 公式可作为强简单基线**
   - 参数少；
   - 拟合稳定；
   - 在中等尺度上已能取得很强结果。

3. **Multi-Power Law 在主任务上更强**
   - 对你指定的 `fit cosine -> evaluate WSD`，MPL 在主任务 WSD 上三种尺度全部更优；
   - 在全部 15 个测试比较中，MPL 赢下 13 个。

4. **跨 schedule 外推是区分方法优劣的关键**
   - 如果只看 cosine 内拟合，两种方法都很好；
   - 真正拉开差距的是从 cosine 到 WSD-family 的外推。

## 9. 产出文件

### 报告与表格

- [reproduction_report.md](file:///Users/jiaju/Documents/github/DL-final/docs/reproduction_report.md)
- [cosine_to_wsd_metrics.csv](file:///Users/jiaju/Documents/github/DL-final/results/tables/cosine_to_wsd_metrics.csv)
- [fitted_params.json](file:///Users/jiaju/Documents/github/DL-final/results/tables/fitted_params.json)

### 关键可视化

- 平均测试 MAE: [avg_test_mae.png](file:///Users/jiaju/Documents/github/DL-final/results/figures/avg_test_mae.png)
- 平均测试 RMSE: [avg_test_rmse.png](file:///Users/jiaju/Documents/github/DL-final/results/figures/avg_test_rmse.png)
- 主任务 WSD 的跨尺度 MAE: [wsd_mae_by_scale.png](file:///Users/jiaju/Documents/github/DL-final/results/figures/wsd_mae_by_scale.png)
- 主任务 WSD 的跨尺度 R2: [wsd_r2_by_scale.png](file:///Users/jiaju/Documents/github/DL-final/results/figures/wsd_r2_by_scale.png)

### 逐曲线图

每个尺度、每个方法、每条曲线都已生成对应对比图，保存在：

- [results/figures](file:///Users/jiaju/Documents/github/DL-final/results/figures)

### 逐点预测结果

逐 step 的 `ground truth / prediction` 文件保存在：

- [results/predictions](file:///Users/jiaju/Documents/github/DL-final/results/predictions)

## 10. 参考

- Tissue, H., Wang, V., Wang, L. *Scaling Law with Learning Rate Annealing*. AlphaXiv / arXiv 2408.11029. [https://www.alphaxiv.org/abs/2408.11029](https://www.alphaxiv.org/abs/2408.11029)
- Luo, K. et al. *A Multi-Power Law for Loss Curve Prediction Across Learning Rate Schedules*. [https://github.com/thu-yao-01-luo/MultiPowerLaw](https://github.com/thu-yao-01-luo/MultiPowerLaw)
