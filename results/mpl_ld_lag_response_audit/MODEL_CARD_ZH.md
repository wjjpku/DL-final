# MPL-LD Cooldown Finite-Response Model Card

这份 model card 只描述当前最干净的候选，不把 observation-bracket MPL-LD 写成主方法。核心原则是：只修改 MPL 自己的 LR-dependent decay term，不新增 residual basis，不从 cosine residual 拟合推荐模型参数。

## 推荐公式

MPL baseline 写作

\[
L_{\mathrm{MPL},s}(t)=L_{0,s}+A_sS_s(t)^{-\alpha_s}+B_sD_s(t).
\]

将 MPL 的 \(D_s(t)\) 按 LR 变化方向拆成

\[
D_s(t)=D_{\uparrow,s}(t)+D_{\downarrow,s}(t),
\]

只对 cooldown 子项引入有限响应：

\[
D_{\downarrow,\tau_s,s}(t_i)
=\rho_iD_{\downarrow,\tau_s,s}(t_{i-1})+(1-\rho_i)D_{\downarrow,s}(t_i),
\quad \rho_i=\exp[-(t_i-t_{i-1})/\tau_s].
\]

最终预测为

\[
\hat L_s(t)=L_{\mathrm{MPL},s}(t)+a_sB_s[D_{\downarrow,\tau_s,s}(t)-D_{\downarrow,s}(t)].
\]

响应时间和边界项都只由 LR schedule / logging resolution 计算：

\[
\tau_s=\Delta_{\mathrm{obs}}\left(1+\min\left(1,\frac{\ell_\downarrow}{\Delta_{\mathrm{obs}}}\right)\right),
\qquad
a_s=\left[1-\frac{\ell_\downarrow}{T-W}\right]_+.
\]

其中 \(\ell_\downarrow\) 是 post-warmup 的 LR-drop support span。这个模型没有 residual-fitted coefficient。

## 当前结果

### Frozen official MPL backbone

| model | WSD mean | WSD worst | WSD wins | controls | fitted residual params |
|---|---:|---:|---:|---:|---:|
| support-bracket cooldown finite-response | -13.77% | -6.29% | 15/15 | 9/9 non-harm | 0 |
| fixed tau=128 cooldown finite-response | -8.73% | -6.22% | 15/15 | 9/9 non-harm | 0 |
| cosine-fitted amplitude negative control | +525.54% | +1166.45% | 0/15 | fails | 1, not recommended |

这组结果只能说明 finite-response 修正在 frozen official MPL backbone 上有稳定机制信号。它不能单独作为严格 cosine-to-WSD protocol 的最终结论，因为 `MPL_PRECOMPUTED_INIT` 来自官方公开 split，不是只用 cosine curves 拟合。

### Strict cosine-only MPL backbone

同一个公式、同一套 schedule-only \(\tau_s,a_s\)，把 MPL backbone 换成只由 `cosine_24000.csv` 和 `cosine_72000.csv` 拟合得到的参数：

| protocol | WSD mean | WSD worst | WSD wins | controls | fitted residual params |
|---|---:|---:|---:|---:|---:|
| cosine-only MPL + finite response | -11.44% | -6.40% | 15/15 | 9/9 non-harm | 0 |

这说明公式本身不是完全依赖 official frozen backbone 才有效。但 cosine-only MPL backbone 本身比 official MPL 在 WSD 上平均差 `+55.05%` MAE；修正后仍比 official MPL baseline 平均差 `+37.34%` MAE。因此当前方法只能说是一个解释性较强的机制修正，还不能说完整解决 cosine-to-WSD。

## 消融含义

1. Full \(D_\tau-D\) 能改善 WSD，但会伤 short-cosine / constant controls，说明不能把 MPL 的 warmup/increase 与 cooldown/decrease 混在一起 lag。
2. Cooldown-only 分解让 constant controls 变为 0，说明误差主要来自 LR 下降子项。
3. Adiabatic boundary 让 full-horizon cosine decay 不再被当成本地 cooldown transient，恢复 controls non-harm。
4. Support-bracket \(\tau_s\) 解释了为什么 4k-step WSD cooldown 需要比 single-step WSD-con 更长的响应时间。
5. 从 cosine residual 拟合 amplitude 会灾难性失败，说明 cosine residual contamination 仍然存在，不能自由学习幅度。

## 当前限制

- \(a_s\) 是 schedule-level boundary prior，不是 MPL 内部唯一推出的定理。
- 当前收益低于 observation-bracket MPL-LD 诊断模型，但解释性更强。
- strict cosine-only backbone 下虽然 WSD 15/15 正优化，但绝对误差仍落后 official MPL baseline。
- 仍缺少新训练 run 或新 schedule 的外部验证。
- 因为推荐模型不拟合 residual 参数，它更像一个机制修正 baseline，而不是最终性能上限。
