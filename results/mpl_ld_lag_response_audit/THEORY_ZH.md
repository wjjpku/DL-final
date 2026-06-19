# MPL-LD Cooldown Finite-Response 理论说明

这份说明只为当前最干净候选服务：

\[
\hat L_s(t)
=
L_{\mathrm{MPL},s}(t)
+
a_sB_s[D_{\downarrow,\tau_s,s}(t)-D_{\downarrow,s}(t)].
\]

它不是把 residual 曲线看完后再拼一个外部 basis，而是对 MPL 自己的 learning-rate dependent term 做最小动力学修正。

## 1. 从 MPL 出发

MPL baseline 写成

\[
L_{\mathrm{MPL},s}(t)
=
L_{0,s}
+A_sS_s(t)^{-\alpha_s}
+B_sD_s(t).
\]

其中

\[
S_s(t)=\sum_{u\le t}\eta_u
\]

表示 cumulative learning-rate time，\(D_s(t)\) 是 MPL 中由 LR history 计算的 LR-dependent decay term。

当前 MPL 隐含了一个强假设：只要 LR schedule 给定，\(D_s(t)\) 对 loss 的影响可以被瞬时读出。也就是说，MPL 把

\[
B_sD_s(t)
\]

当成即时响应项。

这个假设在 LR 平滑变化或 constant LR 时通常问题不大，但在 LR drop / cooldown 后可能过强：真实训练动态可能需要一段时间才能跟上新的 LR regime。

## 2. 为什么只处理 cooldown 子项

MPL 的 \(D_s(t)\) 本质上来自 LR changes 的累积贡献。按 LR 变化方向拆开：

\[
D_s(t)=D_{\uparrow,s}(t)+D_{\downarrow,s}(t),
\]

其中

- \(D_{\uparrow,s}(t)\)：来自 \(\Delta\eta_t>0\) 的 warmup / LR increase；
- \(D_{\downarrow,s}(t)\)：来自 \(\Delta\eta_t<0\) 的 cooldown / LR decrease。

我们要解释的是 cosine-to-WSD transfer 中 LR 下降后的误差，因此只对 \(D_{\downarrow,s}(t)\) 做有限响应修正。

这个拆分不是新增 channel，也不是按任务类型 route。它只是 MPL 原有 \(D(t)\) 求和项中 \(\Delta\eta\) 的符号分解。

审计结果支持这个决定：

| variant | WSD | controls |
|---|---:|---:|
| full \(D_\tau-D\), \(\tau=128\) | -9.52%, 15/15 | worst +6.01% |
| cooldown-only, \(\tau=128\) | -9.44%, 15/15 | constant controls become 0, but short-cosine still harms |
| cooldown + adiabatic boundary, \(\tau=128\) | -8.73%, 15/15 | 9/9 non-harm |

这说明 full \(D\) 的 finite response 混入了不该 lag 的部分；cooldown-only 更符合问题来源。

## 3. 一阶响应方程

把 \(D_{\downarrow,s}(t)\) 看成 MPL 给出的 quasi-static cooldown target。真实训练中的 effective cooldown state 记为 \(Z_s(t)\)。假设 \(Z_s(t)\) 以一阶响应追踪 \(D_{\downarrow,s}(t)\)：

\[
\frac{dZ_s}{dt}
=
\frac{D_{\downarrow,s}(t)-Z_s(t)}{\tau_s}.
\]

在离散 logging steps \(t_i\) 上，解析更新为

\[
Z_s(t_i)
=
\rho_iZ_s(t_{i-1})
+(1-\rho_i)D_{\downarrow,s}(t_i),
\]

\[
\rho_i=\exp[-(t_i-t_{i-1})/\tau_s].
\]

记

\[
D_{\downarrow,\tau_s,s}(t_i)=Z_s(t_i),
\]

则有限响应版本的 MPL-LD 项为

\[
B_sD_{\downarrow,\tau_s,s}(t).
\]

相对原 MPL baseline 的修正就是

\[
B_s[D_{\downarrow,\tau_s,s}(t)-D_{\downarrow,s}(t)].
\]

这一步没有新拟合系数。幅度仍然是 MPL 自己的 \(B_s\)。

## 4. Response time 如何得到

不能从 WSD loss 选择 \(\tau_s\)。当前使用 observation-resolution support bracket：

\[
\tau_s
=
\Delta_{\mathrm{obs}}
\left(
1+\min(1,\ell_\downarrow/\Delta_{\mathrm{obs}})
\right).
\]

其中：

- \(\Delta_{\mathrm{obs}}\)：loss logging 的主要间隔，当前为 128 steps；
- \(\ell_\downarrow\)：post-warmup LR-drop support span；
- \(\ell_\downarrow/\Delta_{\mathrm{obs}}\)：cooldown 持续时间相对一个 observation window 的长度。

解释：

- single-step cooldown 只占一个 observation window 的很小一部分，因此 \(\tau_s\) 约为一个 observation interval；
- 持续至少一个 observation interval 的 cooldown 使用 two-observation response upper bound；
- 不再让所有 schedule 共享同一个 \(\tau\)，避免 WSD sharp/linear 与 WSD-con 的响应时间被错误绑在一起。

当前数据中：

| target | \(\ell_\downarrow\) | \(\tau_s\) |
|---|---:|---:|
| WSD sharp / linear | 4000 | 256 |
| WSD-con | 2 | 130 |
| Cosine 24k control | full horizon | irrelevant after \(a_s=0\) |

## 5. Adiabatic boundary

full-horizon cosine decay 不应被当成本地 cooldown transient。为此使用 schedule-only boundary：

\[
a_s=
\left[
1-\frac{\ell_\downarrow}{T-W}
\right]_+.
\]

其中 \(T-W\) 是 post-warmup horizon。

含义：

- 如果 LR drop support 很短，cooldown 是局部事件，finite-response correction 应保留；
- 如果 LR drop support 覆盖整个 post-warmup horizon，schedule 更接近 quasi-adiabatic decay，不应施加 local transient correction；
- constant LR 没有 cooldown support，修正自然为 0。

这个 \(a_s\) 不是 learned gate，也不使用 target loss。它是当前模型中最需要谨慎表述的 boundary prior。

## 6. 最终公式和参数来源

最终公式：

\[
\hat L_s(t)
=
L_{\mathrm{MPL},s}(t)
+
a_sB_s[D_{\downarrow,\tau_s,s}(t)-D_{\downarrow,s}(t)].
\]

参数来源：

| quantity | source | residual-fitted? | target loss? |
|---|---|---:|---:|
| \(L_{\mathrm{MPL}},B_s,D_s\) | existing MPL baseline | 0 | outside error model |
| \(D_{\downarrow,s}\) | signed decomposition of MPL \(D_s\) | 0 | 0 |
| \(\Delta_{\mathrm{obs}}\) | logging interval | 0 | 0 |
| \(\ell_\downarrow\) | LR schedule support | 0 | 0 |
| \(\tau_s\) | support-bracket rule | 0 | 0 |
| \(a_s\) | adiabatic boundary | 0 | 0 |
| residual amplitude | not used | 0 | 0 |

## 7. 当前实验证据

推荐模型：

| model | WSD mean | WSD worst | wins | controls |
|---|---:|---:|---:|---:|
| support-bracket cooldown finite-response | -13.77% | -6.29% | 15/15 | 9/9 non-harm |

负控：

| negative control | WSD mean | WSD worst | wins |
|---|---:|---:|---:|
| cosine-fitted cooldown amplitude, \(\tau=128\) | +525.54% | +1166.45% | 0/15 |

这个负控说明：如果允许从 cosine residual 自由拟合 amplitude，模型会吸收 global MPL drift 并严重过转移。因此推荐模型坚持不拟合 residual amplitude。

## 8. 当前边界

可以安全说：

- MPL 的 cooldown 子项存在有限响应误差；
- 只修改 \(D_{\downarrow}\) 比修改 full \(D\) 更符合 residual 来源；
- support-bracket \(\tau_s\) 是一个不使用 loss 的 schedule-only response-time rule；
- 推荐模型在现有 WSD-family 上全面正优化，并保持 controls non-harm。

不能过度说：

- 这已经是通用训练 loss 定律；
- \(a_s\) 已经从 MPL 内部唯一推出；
- 不需要外部 schedule 或新训练 run 验证。
