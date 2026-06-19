# 解释性重置说明

这份说明用于纠正前面模型过复杂、解释性不足的问题。结论先写清楚：之前带 gate、channel 选择、正弦/频域展开、DCT residual basis、二阶段 residual 拟合的版本，都不应该作为主方法。它们可以作为诊断或负控材料，但不能被包装成核心贡献。

## 1. 为什么前面的模型不合格

前面几类模型的问题不一样，但本质相同：它们是在看见 residual 形状之后再补结构，解释链条不够硬。

| 模型做法 | 问题 | 当前处理 |
|---|---|---|
| gate / route | schedule 分类边界可以人为调，像规则工程 | 不作为主方法 |
| channel 选择 | “选择哪个 channel”缺少从 MPL 原式唯一推出的理由 | 不作为主方法 |
| 正弦 / DCT 展开 | 可以拟合误差形状，但物理或优化解释弱 | 只作为残差形状诊断 |
| cosine residual 拟合 amplitude | 会吸收全局 MPL drift，迁移到 WSD 灾难性失败 | 作为负控 |
| 二阶段 MPL/residual 轮流拟合 | residual 会吃掉 first-stage MPL 假设误差，解释混杂 | 不作为主方法 |

所以主线必须收缩到一个更严格的标准：

1. 不能新增看 residual 形状后设计的 basis；
2. 不能从 WSD target loss 选择参数；
3. 最好不从 cosine residual 学幅度；
4. 新项必须能从 MPL 原式中的已有量推出；
5. 每个变量都必须能从 LR schedule、logging resolution 或 MPL backbone 直接计算。

## 2. 当前唯一可保留的公式

当前最干净的候选只修改 MPL 自己的 LR-dependent term。

MPL baseline 写作

\[
L_{\mathrm{MPL},s}(t)
=
L_{0,s}
+A_sS_s(t)^{-\alpha_s}
+B_sD_s(t).
\]

这里 \(D_s(t)\) 是 MPL 中由 learning-rate history 计算出的 learning-rate dependent term。

把它按 LR 变化方向拆开：

\[
D_s(t)=D_{\uparrow,s}(t)+D_{\downarrow,s}(t).
\]

其中 \(D_{\downarrow,s}(t)\) 只包含 \(\Delta\eta_t<0\) 的 LR drop / cooldown 贡献。这个拆分不是新 channel，也不是 route；它只是 MPL 原求和项中按 \(\Delta\eta\) 的符号拆分。

我们只对 \(D_{\downarrow,s}(t)\) 加一阶有限响应：

\[
D_{\downarrow,\tau_s,s}(t_i)
=
\rho_iD_{\downarrow,\tau_s,s}(t_{i-1})
+(1-\rho_i)D_{\downarrow,s}(t_i),
\]

\[
\rho_i=\exp[-(t_i-t_{i-1})/\tau_s].
\]

最终预测：

\[
\hat L_s(t)
=
L_{\mathrm{MPL},s}(t)
+a_sB_s
\left[
D_{\downarrow,\tau_s,s}(t)
-D_{\downarrow,s}(t)
\right].
\]

这条式子的含义是：MPL 假设 \(D_{\downarrow}\) 对 loss 的影响可以瞬时体现；我们只把这部分改成一个 causal first-order response。幅度仍然使用 MPL 自己的 \(B_s\)，不额外拟合 residual amplitude。

## 3. 新变量怎样计算

响应时间：

\[
\tau_s
=
\Delta_{\mathrm{obs}}
\left(
1+
\min(1,\ell_\downarrow/\Delta_{\mathrm{obs}})
\right).
\]

其中：

| 变量 | 来源 | 是否拟合 |
|---|---|---:|
| \(\Delta_{\mathrm{obs}}\) | loss curve logging 的主要间隔，当前为 128 steps | 否 |
| \(\ell_\downarrow\) | post-warmup LR drop support span | 否 |
| \(\tau_s\) | observation-resolution support bracket | 否 |

直觉是：single-step drop 只能在一个 observation interval 内被看见，所以给约一个 observation interval；持续时间超过一个 observation interval 的 cooldown，给 two-observation response upper bracket。

adiabatic boundary：

\[
a_s
=
\left[
1-\frac{\ell_\downarrow}{T-W}
\right]_+.
\]

这里 \(T-W\) 是 post-warmup horizon。这个项的作用是避免把 full-horizon cosine decay 当成局部 cooldown transient。它不是 learned gate，也不使用 loss，但它确实是当前公式中最需要谨慎表述的 schedule-level prior。

## 4. 当前真实结果

在 frozen official MPL backbone 上：

| protocol | WSD mean | WSD worst | WSD wins | controls |
|---|---:|---:|---:|---:|
| official frozen MPL + finite response | -13.77% | -6.29% | 15/15 | 9/9 non-harm |

但是这还不够，因为 official MPL backbone 不是严格 cosine-only 训练。

在 strict cosine-only backbone 上，同一个公式得到：

| protocol | WSD mean | WSD worst | WSD wins | controls |
|---|---:|---:|---:|---:|
| cosine-only MPL + finite response | -11.44% | -6.40% | 15/15 | 9/9 non-harm |

这说明公式本身不是完全靠 official backbone 才有效；它在严格 cosine-only backbone 上仍然对所有 WSD-family target 正优化。

但也必须同时承认：

| comparison | result |
|---|---:|
| cosine-only MPL baseline vs official MPL baseline on WSD | +55.05% MAE |
| corrected cosine-only MPL vs official MPL baseline on WSD | +37.34% MAE |

也就是说，finite-response 修正确实降低了 strict cosine-only MPL 的 WSD 误差，但还没有把它拉回 official MPL baseline 的水平。当前不能说已经完整解决 cosine-to-WSD，只能说找到了一个可解释、无 residual-fitted 参数、方向稳定的机制修正。

## 5. 现在可以怎样讲

可以讲：

- MPL 的 LR-dependent term 中，cooldown contribution 存在 finite-response 误差；
- 只修正 \(D_{\downarrow}\) 比修正 full \(D\) 更有解释性；
- 当前公式没有 gate、channel selection、sinusoid、DCT residual basis；
- 在 official frozen backbone 和 strict cosine-only backbone 下，WSD-family 都是 15/15 正优化；
- cosine-fitted amplitude 是强负控，说明不能自由从 cosine residual 学幅度。

不能讲：

- 已经解决了 cosine-to-WSD；
- \(a_s\) 是从 MPL 唯一严格推出的定理；
- 当前结果足够支撑强科研主张；
- 这些复杂 residual 模型有清晰机制解释。

## 6. 下一步应该做什么

下一步不应该继续加项。更合理的是做三件事：

1. 用 leave-one-schedule 或新增 schedule 检验 \(a_s,\tau_s\) 是否稳定；
2. 尝试直接在 MPL 最后一项内部联合拟合 finite-response 形式，而不是二阶段 residual 拟合；
3. 如果 strict cosine-only backbone 仍明显弱于 official MPL，需要把问题拆成两个部分：MPL backbone 参数泛化问题，以及 cooldown finite-response 问题。

当前主线应当被表述为：一个解释性强的 MPL-LD finite-response diagnostic，不是最终性能模型。
