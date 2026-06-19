# 解释性审计：当前模型为什么还不够好

这份记录是对当前 residual-error 模型路线的降级说明。结论先说清楚：observation-bracket MPL-LD 在现有数据上有效，但还不应该被包装成最终主方法。它可以作为强消融结果，说明 MPL 对 LR 下降后的 loss 响应确实有系统误差；但作为“可解释模型”，它仍然不够干净。

## 1. 当前公式的问题

当前公式写作

\[
\hat L_s(t)=L_{\mathrm{MPL},s}(t)+a_s\hat\kappa_s\phi_{\lambda_s,s}(t).
\]

其中

\[
\phi_{\lambda,s}(t)
=
\sum_{u\le t}
\exp[-\lambda\eta_u]
\frac{[\eta_{u-1}-\eta_u]_+}{\eta_{\max}},
\qquad
\lambda_s=\lambda_{\mathrm{obs}}\frac{1+q_s}{2}.
\]

这个形式比 gate、channel、正弦展开干净很多，但仍有四个解释性弱点。

第一，\(\phi_{\lambda,s}\) 是 MPL 外部新增的 basis。它表达了“LR drop 后有滞后响应”，但没有直接从 MPL 的最后一项 \(B D(t)\) 推出来。读者会问：为什么不是别的 kernel？为什么是这个 exponential memory？

第二，\(\lambda_s\) 是 observation-resolution prior，不是模型内部推导。用一到两个 logging interval 夹住响应时间是合理的工程约束，但它不是从 loss dynamics 或 MPL 参数中严格推出的。

第三，MPL-LD tangent projection 是必要的，但也暴露了问题本身。如果 raw projection 在 WSD 上严重失败，而加上 tangent projection 后才成功，说明 correction 很依赖 nuisance-removal protocol。这个 protocol 可以解释为去掉 MPL 的 LR-dependent 参数误差方向，但仍然会显得像“先清洗 residual，再投影”的后处理。

第四，locality boundary \(a_s\) 不是学习参数，也不用 target loss，但它仍是公式外部的 schedule rule。它主要用于阻止 full-horizon cosine control 被错误修正，不应该被写成核心理论贡献。

## 2. 数值有效不等于解释性足够

已有结果说明当前模型不是纯粹随机拟合：

| method | same-scale WSD | cross-scale WSD | controls |
|---|---:|---:|---:|
| observation-bracket MPL-LD + locality | -29.87%, 15/15 | -24.95%, 30/30 | 9/9 non-harm |
| old MPL-LD fixed tau | -27.25%, 15/15 | -23.07%, 30/30 | 9/9 non-harm |
| no-nuisance raw projection | +602.17%, 0/15 | +572.52%, 0/30 | not useful |

但这些数字只能支持较弱结论：

1. MPL 的 cosine residual 不能直接转移到 WSD。
2. MPL 的 LR-dependent tangent directions 确实解释了很多污染项。
3. 去掉这些污染后，LR drop response 能改善 WSD-family prediction。

它们还不能支持较强结论：

1. 当前 \(\phi_{\lambda,s}\) 就是正确的物理/优化动力学项。
2. \(\lambda_s=\lambda_{\mathrm{obs}}(1+q_s)/2\) 是唯一合理的响应时间。
3. locality boundary 是真正的理论组成部分。

## 3. 更可解释的收缩方向

更好的主线应该从 MPL 自己的最后一项开始，而不是在 MPL 外面另加 residual basis。

MPL 当前写作

\[
L_{\mathrm{MPL}}(t)
=
L_0
+A S(t)^{-\alpha}
+B D(t),
\]

其中 \(S(t)=\sum_{u\le t}\eta_u\)，\(D(t)\) 是 MPL 中由 LR history 计算出的 learning-rate-dependent decay term。当前 MPL 实际假设：当 LR schedule 变化时，\(D(t)\) 对 loss 的影响可以被瞬时读出。

如果真实训练存在有限响应时间，更自然的修改是把 \(D(t)\) 替换为一个 causal lagged version：

\[
D_\tau(t_i)
=
\rho_iD_\tau(t_{i-1})
+(1-\rho_i)D(t_i),
\qquad
\rho_i=\exp\left[-\frac{t_i-t_{i-1}}{\tau}\right].
\]

于是预测为

\[
\hat L_\tau(t)
=
L_0
+A S(t)^{-\alpha}
+B D_\tau(t).
\]

等价地，如果保留已有 MPL baseline，

\[
\hat L_\tau(t)
=
L_{\mathrm{MPL}}(t)
+B\left[D_\tau(t)-D(t)\right].
\]

这个公式比当前 residual-response 公式更容易解释：

1. \(D(t)\) 已经是 MPL 的最后一项，不是新发明的 basis。
2. \(B\) 已经是 MPL 参数，不新增 residual amplitude。
3. \(\tau\) 是唯一需要讨论的机制量，含义是 MPL decay term 的响应时间。
4. constant LR 没有新的外部 gate；如果 \(D(t)\) 不发生有效变化，correction 自然很小。

这条路线的代价是数值可能没有当前 observation-bracket MPL-LD 好。但如果目标是可解释性，这个代价是可以接受的；反过来，如果它完全无效，就应该诚实承认目前还没有足够硬的可解释模型。

## 4. 收缩实验结果

已新增 `repro/mpl_ld_lag_response_audit.py`，只测试上面的 \(B[D_\tau-D]\) 机制，不加入外部 residual basis。

固定 \(\tau\)、不拟合 residual 幅度时：

| \(\tau\) steps | WSD mean | WSD worst | WSD wins | controls worst |
|---:|---:|---:|---:|---:|
| 32 | -0.42% | -0.28% | 15/15 | +0.19% |
| 64 | -3.11% | -2.38% | 15/15 | +1.62% |
| 128 | -9.52% | -5.95% | 15/15 | +6.01% |
| 256 | -15.09% | +18.57% | 14/15 | +15.90% |
| 512 | -6.59% | +92.63% | 10/15 | +36.20% |

这说明 \(B[D_\tau-D]\) 确实抓到了 WSD 的一部分系统误差：在 \(\tau=64\) 或 \(\tau=128\) 附近，所有 WSD-family target 都有改善。但它还不是可交付主模型，因为 controls 已经出现 harm，而且稍大的 \(\tau\) 会造成严重过修正。

更关键的是，如果在同一个 \(B[D_\tau-D]\) feature 上从 cosine residual 拟合一个幅度系数，结果会灾难性过转移：

| setting | same-scale WSD | cross-scale WSD | controls |
|---|---:|---:|---:|
| \(\tau=128\), cosine-fitted amplitude | +565.16%, 0/15 | +548.27%, 0/30 | +656.40% worst |

这条负结果很重要：它证明问题不只是“公式不够强”，而是 cosine residual 中确实混入了很重的全局 MPL drift。只要让幅度从 cosine residual 自由吸收误差，泛化就会崩掉。

进一步把 MPL 的 \(D(t)\) 按 LR 变化方向拆开：

\[
D(t)=D_\uparrow(t)+D_\downarrow(t),
\]

其中 \(D_\uparrow\) 来自 \(\Delta\eta_t>0\) 的 warmup / LR increase，\(D_\downarrow\) 来自 \(\Delta\eta_t<0\) 的 cooldown / LR decrease。只 lag \(D_\downarrow\) 后，constant controls 自动为 0，但 short cosine 仍会被当成 diffuse cooldown transient。

因此再加一个 schedule-only adiabatic boundary：

\[
a_s=\left[1-\frac{\ell_\downarrow}{T-W}\right]_+,
\]

其中 \(\ell_\downarrow\) 是 post-warmup positive LR-drop support span。最终候选写成

\[
\hat L_\tau(t)
=
L_{\mathrm{MPL}}(t)
+a_sB[D_{\downarrow,\tau}(t)-D_\downarrow(t)].
\]

这个 \(a_s\) 不是拟合参数，也不是按 curve family 选择的 gate；它表达的是：如果 LR decay 支撑集覆盖整个 post-warmup horizon，就应该视为 quasi-adiabatic schedule，而不是 local cooldown transient。这个边界项仍需谨慎，因为它是 schedule-level prior，不是从 MPL 公式内部唯一推出。

固定 \(\tau=\Delta_{\mathrm{obs}}=128\) 时：

| variant | WSD mean | WSD worst | wins | controls |
|---|---:|---:|---:|---:|
| full \(D_\tau-D\) | -9.52% | -5.95% | 15/15 | worst +6.01% |
| cooldown only | -9.44% | -6.22% | 15/15 | worst +6.53% |
| cooldown + adiabatic boundary | -8.73% | -6.22% | 15/15 | 9/9 non-harm |

为了避免 sharp/linear WSD 的 4k-step cooldown 和 WSD-con 的 two-step cooldown 使用同一个响应时间，再把 \(\tau\) 改为 schedule-only support bracket：

\[
\tau_s
=
\Delta_{\mathrm{obs}}
\left(
1+\min\left(1,\frac{\ell_\downarrow}{\Delta_{\mathrm{obs}}}\right)
\right).
\]

这个规则只用 logging interval 和 LR-drop support span。单步 drop 几乎是一倍 observation interval；持续至少一个 observation interval 的 cooldown 使用两倍 observation interval。当前数据中它给出：

| target type | \(\ell_\downarrow\) | effective \(\tau_s\) |
|---|---:|---:|
| WSD sharp / linear | 4000 | 256 |
| WSD-con | 2 | 130 |
| Cosine 24k control | full horizon | irrelevant because \(a_s=0\) |

最终当前 clean candidate：

\[
\hat L_s(t)
=
L_{\mathrm{MPL},s}(t)
+a_sB_s[D_{\downarrow,\tau_s,s}(t)-D_{\downarrow,s}(t)].
\]

结果：

| variant | WSD mean | WSD worst | wins | controls |
|---|---:|---:|---:|---:|
| fixed \(\tau=128\) cooldown + boundary | -8.73% | -6.22% | 15/15 | 9/9 non-harm |
| support-bracket \(\tau_s\) cooldown + boundary | -13.77% | -6.29% | 15/15 | 9/9 non-harm |

因此新的判断是：

1. \(B[D_\tau-D]\) 是目前解释性最干净的方向；
2. \(D_\downarrow\) 分解进一步说明误差主要来自 LR 下降子项，而不是 warmup 子项；
3. support-bracket \(\tau_s\) 能解释为什么 long cooldown 比 single-step cooldown 需要更长响应时间；
4. adiabatic boundary 可以在不拟合参数的情况下修复 controls；
5. 但当前 \(a_s\) 仍是 schedule-level prior，因此还不能宣称最终完成；
6. 不能再把更复杂的 observation-bracket MPL-LD 包装成最终理论，只能作为“加了 nuisance control 后数值上能做到什么”的参考。

## 5. 后续实验必须满足的标准

后续只应测试少量预注册版本：

| variant | fitted quantity | allowed source | target loss used? | reading |
|---|---:|---|---:|---|
| support-bracket cooldown MPL-LD lag + adiabatic boundary | 0 | \(\tau_s\) and \(a_s\) from LR schedule support | 0 | current cleanest candidate |
| fixed-\(\tau\) cooldown MPL-LD lag + adiabatic boundary | 0 | \(\tau=\Delta_{\mathrm{obs}}\), \(a_s\) from LR schedule support | 0 | conservative lower version |
| fixed-\(\tau\) MPL-LD lag | 0 | choose \(\tau\) from observation interval, e.g. 128/256/512 | 0 | hardest no-boundary test |
| cosine-calibrated \(\tau\) | 1 | cosine residual only | 0 | acceptable if stable across scales |
| cosine-calibrated scalar on \(B[D_\tau-D]\) | 1 extra amplitude | cosine residual only | 0 | weaker, only if fixed amplitude fails |
| observation-bracket MPL-LD | 1 \(\kappa\) plus protocol rules | cosine residual + schedule prior | 0 | ablation/reference, not main |

不能继续作为主线的内容：

1. gate 或 channel routing；
2. 正弦/DCT basis 作为核心方法；
3. 为某个 curve family 单独选择公式；
4. target WSD loss 参与模型选择；
5. 多个 residual term 叠加后只用结果好来解释。

## 6. 当前写作结论

现在不应该把 slides 或 paper 写成“我们已经得到一个最终可解释 error model”。更准确的说法是：

1. 我们发现 MPL 在 cosine-to-WSD transfer 中有系统 residual contamination。
2. 简单 LR-drop response 如果直接从 cosine residual 拟合会失败。
3. MPL-LD tangent projection 可以显著改善转移，说明误差和 MPL 的 LR-dependent 部分有关。
4. 但当前最佳数值模型仍有 protocol engineering 成分，因此只能作为过渡结果。
5. 下一步主线应收缩到 MPL 最后一项 \(B D(t)\) 的有限响应时间模型；当前最干净候选是 cooldown-only finite response + support-bracket \(\tau_s\) + adiabatic boundary，但还需要继续解释 \(a_s\) 并做更多外部 schedule 验证。

这比继续堆公式更诚实，也更容易让老师判断问题到底在哪里。
