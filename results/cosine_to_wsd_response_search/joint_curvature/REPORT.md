# Joint-Curvature Core Model 实验报告

这个目录保存当前主线公式的搜索与评估结果。当前版本只保留 smooth / step response 与 LR-curvature correction，不引入 target-specific 局部路由项。所有 correction coefficient 都从 `cosine_72000.csv` 的 residual 中估计；WSD-family 曲线用于 development ranking、目标保留验证和最终汇报。

## 1. 模型形式

最终预测写成：

\[
\hat L_s(t)=\hat L_{\mathrm{MPL},s}(t)+C_s(t).
\]

其中 \(C_s(t)\) 由目标 schedule 的 LR path 直接计算：

\[
C_s(t)=
\begin{cases}
k_{\mathrm{smooth}}\phi_4(t), & \text{smooth target},\\
a\,\phi_{20}(t)+b\,\psi_{10}(t), & \text{step-like target}.
\end{cases}
\]

\(\phi_{\lambda}\) 是 LR-drop response：

\[
\phi_{\lambda}(t)=\sum_{u\le t}\exp[-\lambda(S_t-S_u)]
\frac{\max(\eta_{u-1}-\eta_u,0)}{\eta_{\max}}.
\]

\(\psi_{10}\) 是 LR-curvature response：

\[
\psi_{10}(t)=\sum_{u\le t}\exp[-10(S_t-S_u)]
\frac{|\Delta^2\eta_u|}{\eta_{\max}}.
\]

因此每个规模只新增 3 个训练系数：\(k_{\mathrm{smooth}},a,b\)。其他量，例如 LR drop、LR curvature、causal decay kernel、DCT residualization basis，都由 schedule 或 source residual 直接计算。

## 2. 最优非伤害配置

当前冻结使用 config `15620`：

- mean / worst MAE change: `-37.53%` / `-10.80%`
- wins / non-harm: `15/15` / `15/15`
- source decoupled pair: `9995`
- curvature mode: `signed_d2_lr`
- curvature decay: `lambda2=10`
- curvature stability: `tau2=0.003`
- shrink: `1`
- signed: `0`
- mean step coefficients: primary `0.04074`, curvature `0.01870`

这里的 non-harm 指每一个 scale-target row 都没有比 MPL baseline 更差。

## 3. Channel Calibration

Smooth channel：

- fit start: `12000`
- response decay: `lambda=4`
- residualization strength: `mu=0.05`
- DCT modes: `8`
- ridge: `tau=0.05`
- transfer exponent: `p=0.25`
- shrinkage: `rho=0.2`

Step channel：

- fit start: `3000`
- response decay: `lambda=20`
- residualization strength: `mu=0.01`
- DCT modes: `8`
- ridge: `tau=0.05`
- transfer exponent: `p=0`
- shrinkage: `rho=0.35`

Smooth channel 更慢，是因为 cosine-like decay 的 residual 主要表现为较平滑的响应滞后；step channel 更快，是因为 WSD-family transition 附近的 residual 更集中，需要更短的 memory kernel。

## 4. 与前序版本对比

| version | mean MAE change | worst row | wins |
|---|---:|---:|---:|
| decoupled-channel | -36.18% | -6.29% | 15/15 |
| fixed-channel LR-curvature | -37.47% | -9.43% | 15/15 |
| joint-channel LR-curvature | -37.53% | -10.80% | 15/15 |

主要提升来自 LR-curvature term。它对 mean improvement 的增益不大，但显著改善 worst-row performance，说明它更像一个局部 transition-shape 修正，而不是全局重拟合。

## 5. 分目标结果

| target | mean delta | worst scale | wins |
|---|---:|---:|---:|
| WSD sharp | -54.3% | -40.5% | 3/3 |
| WSD linear | -46.3% | -33.1% | 3/3 |
| WSD-con 3e-5 | -57.1% | -46.7% | 3/3 |
| WSD-con 9e-5 | -17.0% | -10.8% | 3/3 |
| WSD-con 18e-5 | -13.0% | -12.1% | 3/3 |

WSD sharp、WSD linear 和 WSD-con 3e-5 的收益最大，说明当目标 schedule 的 LR transition 更明显时，schedule-response correction 能消掉 MPL residual 中较大的一部分。WSD-con 9e-5 和 18e-5 的改进较小，但仍然保持全部 scale 上不伤害 baseline。

## 6. Top-Safe Holdout Check

| split | selected config | dev mean | dev worst | test mean | test worst | test wins |
|---|---|---:|---:|---:|---:|---:|
| dev sharp/linear, test WSD-con | `lambda2=4, tau2=0.01, shrink=0` | -50.3% | -33.1% | -28.9% | -9.4% | 9/9 |
| dev WSD-con, test sharp/linear | `lambda2=10, tau2=0.003, shrink=1` | -29.0% | -10.8% | -50.3% | -33.1% | 6/6 |
| leave WSD sharp | `lambda2=10, tau2=0.003, shrink=1` | -33.5% | -10.8% | -52.8% | -39.1% | 3/3 |
| leave WSD-con 18e-5 | `lambda2=4, tau2=0.01, shrink=0` | -43.7% | -8.4% | -12.5% | -11.2% | 3/3 |
| leave WSD-con 3e-5 | `lambda2=10, tau2=0.003, shrink=1` | -32.6% | -10.8% | -57.1% | -46.6% | 3/3 |
| leave WSD-con 9e-5 | `lambda2=4, tau2=0.01, shrink=0` | -42.7% | -11.2% | -16.3% | -8.4% | 3/3 |
| leave WSD linear | `lambda2=10, tau2=0.003, shrink=1` | -35.3% | -10.8% | -46.3% | -33.1% | 3/3 |
| leave scale 25M | `lambda2=10, tau2=0.003, shrink=1` | -39.7% | -10.8% | -33.2% | -12.1% | 5/5 |
| leave scale 100M | `lambda2=4, tau2=0.01, shrink=0` | -38.5% | -12.5% | -35.5% | -8.4% | 5/5 |
| leave scale 400M | `lambda2=10, tau2=0.003, shrink=0` | -34.7% | -11.0% | -43.0% | -12.7% | 5/5 |

这个检查说明，当前 correction 在现有 split 上不是只靠某一个 target 或某一个 scale 支撑；不过这些 split 仍然来自同一组 WSD-family 数据，因此只能作为内部稳定性证据。

## 7. 结论与风险

当前公式的优点是主线清楚：MPL 负责全局 loss trend，schedule-response correction 负责 LR transition 引入的 residual。相比继续加入更细的 target-specific 项，当前版本更容易解释，也更不容易被质疑为为了少数目标曲线做过度拟合。

需要保留的风险判断：

- 这仍然是基于现有 WSD-family 的 development search；
- curvature response 改善了 worst-case row，但其解释还应通过更多 schedule family 验证；
- 每个规模新增 3 个系数是目前可以接受的复杂度上限，后续不建议再增加缺少明确动力学含义的项；
- 若要声称更强泛化，需要新增 schedule 或严格冻结配置后做外部 held-out evaluation。
