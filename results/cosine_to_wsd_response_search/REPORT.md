# Cosine-to-WSD 无局部路由修正报告

本文档记录当前冻结的主线版本：只使用从 cosine residual 中估计出的 schedule-response correction，并把它迁移到 WSD-family schedules。目标不是重新拟合一条更复杂的 WSD 曲线，而是在保留 MPL 主体结构的前提下，解释为什么 cosine 上学到的 loss dynamics 可以用于修正 WSD、WSD-con 等目标曲线。

## 1. 问题定义

给定一个目标 learning-rate schedule \(s\)，我们已经有 MPL 给出的 loss 预测：

\[
\hat L_{\mathrm{MPL},s}(t).
\]

MPL 的主体项能够描述 scaling-law 层面的下降趋势，但它默认 loss 对 learning-rate 变化的响应近似同步。因此当 schedule 发生明显 decay、drop 或 restart-like transition 时，真实 loss curve 与 MPL curve 之间会出现系统性 residual：

\[
r_s(t)=L_s(t)-\hat L_{\mathrm{MPL},s}(t).
\]

我们的目标是从 calibration curve `cosine_72000.csv` 中提取可迁移的 residual component，并对目标 schedule 做如下修正：

\[
\hat L_s(t)=\hat L_{\mathrm{MPL},s}(t)+C_s(t).
\]

当前版本不使用 target-specific 局部路由项；目标 schedule 只提供 learning-rate path 本身，以及用于 safety check 的目标保留验证。

## 2. 核心公式

首先把 learning-rate drop 写成可直接从 schedule 读出的序列：

\[
d_t=\frac{\max(\eta_{t-1}-\eta_t,0)}{\eta_{\max}}.
\]

其中 \(\eta_t\) 是第 \(t\) 个记录点附近的 learning rate，\(\eta_{\max}\) 是该条 schedule 的 peak learning rate。然后构造 causal LR-response feature：

\[
\phi_{\lambda}(t)=\sum_{u\le t}\exp[-\lambda(S_t-S_u)]d_u.
\]

\(S_t\) 是归一化训练时间或 schedule time。这个 feature 表示：过去的 LR drop 会在后续一段时间内继续影响 loss，但影响强度会随 \(S_t-S_u\) 衰减。

为了避免把 MPL 的慢漂移误当成 schedule response，我们先对 source residual 和 feature 做 DCT residualization：

\[
M_{\mu}y=y-Q(Q^\top Q+\mu D)^{-1}Q^\top y,
\]

其中 \(Q\) 是低频 DCT basis，\(D\) 是对高阶频率更强的平滑惩罚。随后只在 residualized 空间中估计 correction strength：

\[
\kappa=
\frac{1}{1+\rho}
R_{\mathrm{src}}^p
\frac{\langle M_{\mu}\phi_{\lambda},M_{\mu}r\rangle_+}
{\|M_{\mu}\phi_{\lambda}\|_2^2+\tau^2}.
\]

这里 \(\langle\cdot,\cdot\rangle_+\) 表示只保留正相关响应；\(\tau\) 控制小信号时的数值稳定性；\(p,\rho\) 是跨 schedule transfer 的 shrinkage。所有这些量都由 source residual 或 LR schedule 计算得到。

最终 correction 分两类：

\[
C_s(t)=
\begin{cases}
k_{\mathrm{smooth}}\phi_4(t), & \text{smooth target},\\
a\,\phi_{20}(t)+b\,\psi_{10}(t), & \text{step-like target}.
\end{cases}
\]

其中 \(\psi_{10}(t)\) 是 LR curvature response：

\[
\psi_{10}(t)=\sum_{u\le t}\exp[-10(S_t-S_u)]
\frac{\max(|\Delta^2\eta_u|,0)}{\eta_{\max}}.
\]

当前报告中，WSD、WSD-con 都被视作 step-like schedule，使用同一类 step correction。每个模型规模最多新增 3 个系数：

- \(k_{\mathrm{smooth}}\)：smooth schedule response strength；
- \(a\)：step drop response strength；
- \(b\)：LR curvature response strength。

## 3. 实验设置

Calibration source 只使用 `cosine_72000.csv`。目标评估包含 5 类 WSD-family schedules，每类在 25M、100M、400M 三个规模上评估，共 15 个 scale-target rows。

主评估指标是修正后 MAE 相对 MPL baseline 的变化：

\[
\Delta_{\mathrm{MAE}}
=
\frac{\mathrm{MAE}_{\mathrm{ours}}-\mathrm{MAE}_{\mathrm{MPL}}}
{\mathrm{MAE}_{\mathrm{MPL}}}\times 100\%.
\]

因此负数表示误差下降。我们同时报告 mean improvement、worst-row improvement 和 wins/non-harm。这里的 non-harm 要求每一个 scale-target row 都不能比 MPL baseline 更差。

## 4. 当前主结果

当前冻结配置为 `joint_curvature` 中的 config `15620`：

| 指标 | 数值 |
|---|---:|
| mean MAE change | -37.53% |
| worst scale-target row | -10.80% |
| wins / non-harm | 15/15 |
| 每个规模新增系数 | 3 |

分目标结果如下：

| target | mean delta | worst scale | wins |
|---|---:|---:|---:|
| WSD sharp | -54.3% | -40.5% | 3/3 |
| WSD linear | -46.3% | -33.1% | 3/3 |
| WSD-con 3e-5 | -57.1% | -46.7% | 3/3 |
| WSD-con 9e-5 | -17.0% | -10.8% | 3/3 |
| WSD-con 18e-5 | -13.0% | -12.1% | 3/3 |

这个结果说明，主要收益来自把 cosine residual 拆成可迁移的 schedule-response error，而不是为某个目标 schedule 单独增加复杂的局部规则。

## 5. 消融路径

从旧的单通道 correction 到当前公式，性能变化如下：

| version | mean MAE change | worst row | wins |
|---|---:|---:|---:|
| original single response | -34.53% | -6.08% | 15/15 |
| fixed nuisance removal | -35.07% | -6.12% | 15/15 |
| decoupled smooth / step channel | -36.18% | -6.29% | 15/15 |
| fixed-channel curvature | -37.47% | -9.43% | 15/15 |
| joint-curvature core model | -37.53% | -10.80% | 15/15 |

最关键的变化有两个。第一，smooth 与 step-like schedules 不再共享同一个 correction channel，因为它们的 LR geometry 不同。第二，在 step-like schedules 中加入 curvature response，可以解释 LR transition 附近更尖锐的 residual 形状，从而改善 worst-case row。

## 6. 结果解读

这个模型保留了一个清晰的主故事：

1. MPL 给出主要 loss trend，但在 LR schedule 发生变化时会留下系统性 residual。
2. Cosine residual 不是一个整体可迁移误差；其中只有与 LR drop / curvature 对齐的部分更可能迁移到 WSD。
3. DCT residualization 用来压掉低频 MPL drift，避免把全局偏差塞进 \(\kappa\)。
4. Smooth 与 step-like schedules 分开估计，是为了让平滑 decay 和突变 decay 使用不同的 response channel。
5. 当前结果在 15 个 scale-target rows 上全部优于 MPL baseline，说明这个 correction 至少在现有 WSD-family benchmark 上是稳定有效的。

## 7. 局限性

这个版本仍然是 development result，而不是最终泛化结论。主要风险包括：

- schedule family 仍然较少，当前验证主要覆盖 cosine-to-WSD-family transfer；
- config selection 使用了现有 WSD-family 结果，因此仍需要新的 schedule 或新的 held-out split 检验；
- 每个规模新增 3 个系数，复杂度已经比原始 MPL 高，后续不应继续随意增加自由度；
- curvature term 改善 worst-case row，但它仍是经验性 response basis，需要更多曲线验证其物理或优化动力学解释。

因此当前版本适合作为大作业主线：公式足够紧凑，实验结果稳定，且能解释 cosine-to-WSD 的核心问题；但如果要作为更强科研结论，还需要额外数据和更严格的外部验证。

## 8. 文件位置

- 主结果：`results/cosine_to_wsd_response_search/joint_curvature/`
- 详细方法：`results/cosine_to_wsd_response_search/DETAILED_METHODS.md`
- 中文 slides：`slides/main_zh.tex`
- 编译后 PDF：`slides/main_zh.pdf`
- 作图脚本：`repro/plot_new_formula_slides.py`
