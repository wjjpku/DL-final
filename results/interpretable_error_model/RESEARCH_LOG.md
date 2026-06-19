# 可解释误差模型研究日志

本目录用于重新整理 cosine-to-WSD 误差估计研究线。当前原则是：先不修改 slides，不继续堆叠经验公式；只研究少参数、变量来源清楚、可以从 MPL residual 和 LR schedule 推导的 correction。

## 2026-06-19: 重新收束问题

前一阶段的高分模型使用了 smooth/step channel split、LR-curvature、局部 route/ratio/gate、alternating refit 等开发集技巧。它们的共同问题是：每一项单独能给出一点直觉，但整体不像一个统一的机制模型，容易被质疑为为了 WSD-family 结果而调出来。

因此当前研究线放弃把这些作为主方法，只保留它们作为消融或失败尝试。

## 建模约束

新的主模型必须满足：

1. 不使用 target loss 拟合参数；
2. 不使用 target-specific route/gate；
3. 参数数量保持在每个 scale 1 到 2 个；
4. 每个变量必须能从 MPL、cosine residual 或 LR schedule 直接计算；
5. 公式优先解释 residual 产生机制，而不是优先追最高开发集指标；
6. 若引入超参数，必须有固定理由或小范围稳定性检查，不能依赖大量搜索。

## 现有残差证据

已有 residual gallery 显示：

- constant schedule 有 residual，但 LR-drop feature 为零，说明并非所有 MPL residual 都是 LR response；
- cosine residual 很宽、低频，直接从 cosine 拟合一个大 kappa 会混入 MPL backbone drift；
- WSD / WSD-con residual 更局部，更像 LR decay 或 step transition 后的有限响应；
- forced step-only 统一公式稳定但收益小，说明它有解释性但表达力不足；
- alternating MPL/residual refit 没有稳定改善，说明问题不是简单二阶段估计误差。

## 候选 1: residualized causal LR-drop response

最小公式：

\[
\hat L_s(t)=L_{\mathrm{MPL},s}(t)+\kappa \phi_{\lambda,s}(t),
\]

其中

\[
\phi_{\lambda,s}(t)=
\sum_{u\le t}
\exp[-\lambda \eta_u]
\frac{[\eta_{u-1}-\eta_u]_+}{\eta_{\max}}.
\]

\(\kappa\) 只从 cosine residual 拟合：

\[
\hat\kappa=
\frac{
\langle M_\mu \phi_{\lambda,\cos},M_\mu r_{\cos}\rangle_+
}{
\|M_\mu\phi_{\lambda,\cos}\|_2^2+\tau^2
}.
\]

解释：MPL 负责主趋势，\(\phi\) 表示 LR drop 之后 loss 需要有限时间响应。

问题：单一 \(\phi\) 对 WSD-con 有一定作用，但对 WSD sharp / linear 的宽 residual 不够强。

## 候选 2: slope-modulated lag response

更强但仍然统一的机制公式：

\[
\hat L_s(t)
=
L_{\mathrm{MPL},s}(t)
+
\delta\,
g_s(t)\,
\phi_{\lambda,s}(t),
\]

其中

\[
g_s(t)=\left[-\frac{dL_{\mathrm{MPL},s}}{dS}(t)\right]_+.
\]

解释：LR 下降后，真实 loss 不是凭空增加，而是沿 MPL 下降方向产生时间滞后。若 MPL 曲线局部下降很快，同样的响应滞后会造成更大的 loss error；若 MPL 曲线局部较平，误差自然更小。

这个模型比直接加 curvature 更有解释性，因为它把 correction 的 loss 单位交给 MPL slope，而不是额外构造一个经验形状项。

待验证问题：

- 能否在不做 smooth/step route 的情况下恢复 WSD sharp / linear 的收益；
- \(\lambda\) 是否可以固定在少量物理含义明确的响应时间上；
- \(\delta\) 在 25M / 100M / 400M 上是否稳定。

## 候选 3: two-timescale relaxation spectrum

若单一响应时间不足，可以考虑：

\[
\hat L_s(t)
=
L_{\mathrm{MPL},s}(t)
+
a\,g_s(t)\phi_{\lambda_1,s}(t)
+
b\,g_s(t)\phi_{\lambda_2,s}(t).
\]

这里不是 route，也不是按 schedule 类型切换，而是所有 target 共用同一个慢/快 relaxation spectrum。它最多 2 个系数，解释为两个响应时间常数。

这个模型可以作为候选 2 的上限对照，但不能继续扩展更多 basis。

## 当前判断

推荐研究顺序：

1. 先评估候选 1，作为最小 baseline；
2. 再评估候选 2，看 slope modulation 是否提供真实机制增益；
3. 如果候选 2 明显不足，再用候选 3 做少参数上限；
4. 所有 channel split、curvature、alternating、gate 只保留为对照，不进入主公式。

当前还不能宣称完成。下一步需要用缓存版脚本跑固定候选，并保存 summary / per-target / coefficient stability。

## 2026-06-19: 固定候选实验结果

已实现并运行 `repro/interpretable_error_model.py`。该脚本不做大规模结构搜索，只评估预先定义的机制候选：

- raw causal LR-drop response；
- slope-modulated lag response；
- two-timescale lag response；
- MPL 内部 LR-dependent term 的一阶敏感度；
- continuous response-rate calibration。

主要结果：

| model | mean | worst | wins | 结论 |
|---|---:|---:|---:|---|
| raw drop, \(\lambda=20\) | -22.06% | -5.30% | 15/15 | 最小强基线，稳定但对 WSD sharp/linear 不够强 |
| slope-modulated lag | about -1% 或更差 | 不稳定 | 最高 15/15 | “有限时间滞后乘 MPL slope”解释不足 |
| MPL sensitivity | 约 0% | 不稳定 | 不足 | 残差不是简单 MPL 内部 \(B,C\) 参数一阶偏差 |
| continuous projected response | -34.45% | -5.30% | 15/15 | 当前最有前途的解释性模型 |

当前最佳公式为：

\[
q_s=\frac{\max_t [\eta_{t-1}-\eta_t]_+}
{\sum_t [\eta_{t-1}-\eta_t]_+},
\qquad
\lambda_s=7+13q_s.
\]

对目标 schedule \(s\)，先由 LR geometry 得到 \(\lambda_s\)，再在 cosine residual 上用同一个 response operator 拟合：

\[
\hat\kappa_s
=
\frac{
\langle M_\mu \phi_{\lambda_s,\cos},M_\mu r_{\cos}\rangle_+
}{
\|M_\mu\phi_{\lambda_s,\cos}\|_2^2+\tau^2
},
\qquad
\hat L_s(t)=L_{\mathrm{MPL},s}(t)+\hat\kappa_s\phi_{\lambda_s,s}(t).
\]

注意这里没有 target loss fitting。目标 schedule 只决定 response operator \(\lambda_s\)，\(\kappa_s\) 仍然来自 cosine residual projection。

结果：

| target | mean | worst | wins |
|---|---:|---:|---:|
| WSD sharp | -50.85% | -35.17% | 3/3 |
| WSD linear | -46.18% | -33.82% | 3/3 |
| WSD-con 3e-5 | -53.08% | -41.40% | 3/3 |
| WSD-con 9e-5 | -13.04% | -5.30% | 3/3 |
| WSD-con 18e-5 | -9.11% | -6.43% | 3/3 |

Holdout selection 检查：

- dev sharp/linear, test WSD-con: test mean -25.07%, test worst -5.30%, 9/9 wins；
- dev WSD-con, test sharp/linear: test mean -17.54%, test worst -12.57%, 6/6 wins；
- leave-target: 每个 held-out target 都保持 3/3 wins；
- leave-scale: 每个 held-out scale 都保持 5/5 wins。

解释性优势：

1. 去掉了离散 smooth/step channel split；
2. 去掉了 curvature、gate、ratio 和 alternating；
3. 每个目标只由 LR schedule 的连续 drop concentration 决定 response rate；
4. correction amplitude 仍然从 cosine residual 中投影得到；
5. 参数数量保持为每个 scale / 每个 response operator 一个非负 \(\kappa\)，没有 target loss 参数。

当前风险：

- \(\lambda_{\min}=7,\lambda_{\max}=20\) 仍需要更强的理论或外部验证；
- \(t_{\mathrm{fit}}\ge8000\) 是合理的 warmup/early-transient 去除规则，但需要在报告中固定为 protocol，而不是搜索结果；
- 这个结果目前仍基于 WSD-family development suite，不能直接声称外部泛化。

## 2026-06-19: 参数来源审计与半衰期解释

上面的 \(7/20\) 端点虽然比 gate/channel 更干净，但仍然像开发集数字。为此新增
`repro/interpretable_parameter_origin_audit.py`，专门检查 response-time 参数能否从
observable quantity 得到，而不是从 WSD target loss 调出来。

### 负结果 1: 不能直接让 cosine residual 自己选择 \(\lambda\)

若在 cosine residual 上扫描 \(\lambda\)，并选择 source objective 最小的值，三个 scale 都选择
\(\lambda=1\)。这个结果在 cosine 自身看起来合理，因为它吸收了很宽的低频 MPL drift；但迁移到
WSD-family 后完全失败：

| rule | mean | worst | wins |
|---|---:|---:|---:|
| cosine-source selected \(\lambda=1\) | +200.29% | +452.79% | 0/15 |

结论：\(\lambda\) 不能通过 raw source fit objective 学。cosine residual 的低频成分会把
response time 拉得过慢，从而把与 LR decay 无关的 MPL backbone error 注入到 WSD transfer。
这也解释了为什么早期 cosine-to-WSD 的误差估计会滞后、振荡、并在 WSD 上失真。

### 负结果 2: step-time 几何规则不适合 cosine-only calibration

旧的 step-time geometry tau 在同族/多源校准里有解释性，但若强制只用 cosine 作为 calibration
source，再把 target tau 由 WSD 几何给出，会过度转移长记忆 correction：

| rule | mean | worst | wins |
|---|---:|---:|---:|
| step-time geometry tau, cosine-only | +30.39% | +58.56% | 1/15 |

结论：这个任务不能简单搬用旧的 step-time route/tau。cosine-to-WSD 需要的是 LR-time response
operator，并且 amplitude 必须在同一个 response operator 下从 cosine residual 投影出来。

### 新解释: 用观测半衰期确定 response-rate 端点

loss curve 的主观测间隔为 128 steps。若 response 在一个观测间隔内完成半衰减，则在 LR-time
recursion

\[
\phi_{\lambda}(t)=
\sum_{u\le t}\exp[-\lambda\eta_u]
\frac{[\eta_{u-1}-\eta_u]_+}{\eta_{\max}}
\]

中有

\[
\lambda_{\mathrm{obs}}
=
\frac{\log 2}{\eta_{\max}\Delta_{\mathrm{obs}}}
=
\frac{\log 2}{3\times10^{-4}\times128}
=18.0507.
\]

因此 fast endpoint 不再是任意数字：\(\lambda=20\) 可解释为 one-observation half-life 的圆整值。
smooth decay 的 response 不应短到只看一个点，否则会把观测噪声和低频 residual 混到一起；当前使用
2.5 个观测间隔作为 slow endpoint：

\[
\lambda_{\mathrm{slow}}=\lambda_{\mathrm{obs}}/2.5=7.2203,\qquad
\lambda_{\mathrm{fast}}=20.
\]

最终 response rate 仍由 schedule drop concentration 连续决定：

\[
q_s=
\frac{\max_t[\eta_{t-1}-\eta_t]_+}
{\sum_t[\eta_{t-1}-\eta_t]_+},
\qquad
\lambda_s=\lambda_{\mathrm{slow}}+
(\lambda_{\mathrm{fast}}-\lambda_{\mathrm{slow}})q_s.
\]

这不是 smooth/step channel 选择。所有 target 使用同一个连续公式；target 只提供 LR schedule，
不提供 target loss。

### 新主结果

将该半衰期端点规则加入主审计后，当前最佳候选为
`obs_half_life_projected_2p5_roundfast20`：

| model | mean | worst | wins |
|---|---:|---:|---:|
| observation half-life projected | -34.56% | -5.30% | 15/15 |
| old continuous \(7/20\) projected | -34.45% | -5.30% | 15/15 |
| fixed \(\lambda=20\) | -22.06% | -5.30% | 15/15 |

Per target:

| target | mean | worst | wins |
|---|---:|---:|---:|
| WSD sharp | -51.51% | -36.50% | 3/3 |
| WSD linear | -46.06% | -33.84% | 3/3 |
| WSD-con 3e-5 | -53.08% | -41.40% | 3/3 |
| WSD-con 9e-5 | -13.04% | -5.30% | 3/3 |
| WSD-con 18e-5 | -9.11% | -6.43% | 3/3 |

### 敏感性

半衰期倍数不是刀尖调参。使用 exact fast endpoint
\(\lambda_{\mathrm{fast}}=\lambda_{\mathrm{obs}}\)，slow multiplier 从 2.0 到 4.0 都保持
15/15 wins；使用 rounded fast endpoint 20 时也保持 15/15 wins，mean 在约 -27% 到 -35% 范围内。
这说明主要机制来自 response operator 和 cosine-only projection，而不是某个精确端点。

### 当时仍需补强

- \(t_{\mathrm{fit}}\ge8000\)、DCT nuisance bandwidth、ridge \(\tau=0.05\) 仍需要做同样的来源审计（下一节已补充）；
- 2.5-observation slow half-life 需要在最终报告中冻结为 protocol，并解释为“最小可辨认 smooth transient”；
- 目前证据仍是 WSD-family 内部验证，不应声称外部 schedule 已经验证；
- slides 暂时不要同步，等这些 protocol-level 参数也审计完再统一改。

## 2026-06-19: protocol 参数敏感性审计

新增 `repro/interpretable_protocol_sensitivity.py`，固定当前 observation half-life response 公式，只改变
protocol 参数：

- fit start；
- DCT nuisance bandwidth；
- nuisance regularization \(\mu\)；
- ridge \(\tau\)。

### fit start

| fit start | mean | worst | wins |
|---:|---:|---:|---:|
| 3000 | -26.97% | -2.15% | 15/15 |
| 5000 | -31.72% | -4.09% | 15/15 |
| 6500 | -33.85% | -4.95% | 15/15 |
| 8000 | -34.56% | -5.30% | 15/15 |
| 10000 | -35.14% | -6.06% | 15/15 |
| 12000 | -34.72% | -5.75% | 15/15 |

结论：丢掉前期 transient 是必要方向，但不是 8000 这个单点在支撑结果。5000 到 12000 都全胜，
8000/10000 附近更强。报告中可以把 \(t\ge8000\) 解释为 warmup 后再额外去除早期 MPL transient
的保守 protocol。

### DCT nuisance bandwidth

在 \(\mu=0.01,\tau=0.05\) 下，DCT modes 从 4 到 12 都给出约 -34.56% mean 且 15/15 wins。
在 \(\mu=0.005\) 下仍全胜但更保守，mean 约 -27%。在 \(\mu=0.02\) 下失败，说明 nuisance
projection 过强会把 response 也吃掉或使 amplitude 估计失真。

结论：DCT modes 不是敏感参数；\(\mu=0.01\) 是当前合适的低频 residualizer 强度，但最终还需写成
固定 protocol。

### ridge \(\tau\)

这是最敏感也最需要解释的参数。结果：

| ridge \(\tau\) | mean | worst | wins |
|---:|---:|---:|---:|
| 0 | +138.78% | +408.55% | 0/15 |
| 0.02 | +49.12% | +162.28% | 3/15 |
| 0.04 | -19.85% | +6.55% | 12/15 |
| 0.045 | -28.99% | -0.11% | 15/15 |
| 0.05 | -34.56% | -5.30% | 15/15 |
| 0.055 | -34.25% | -5.62% | 15/15 |
| 0.08 | -20.69% | -3.36% | 15/15 |
| 0.2 | -4.06% | -1.02% | 15/15 |

低 \(\tau\) 会严重失败，因为 cosine residualized feature 的可辨认能量很小，raw projection 会把
cosine drift 放大成过大的 \(\kappa\)。当前 response features 在 cosine source 上经过 DCT
residualization 后的范数范围为：

| target type | \(\lambda\) | \(\|M\phi\|\) | identifiable fraction |
|---|---:|---:|---:|
| WSD sharp/linear operator | about 7.22 | about 0.0422 | about 0.0098 |
| WSD-con operator | 20.0 | about 0.0216 | about 0.0200 |

因此 \(\tau=0.05\) 可以解释为略高于最大可辨认 response-feature norm 的 round threshold。它不是新的
loss-fitted 参数，而是“若 response direction 在 cosine 中可辨认能量不足，则不要信任 raw projection”的
保守 prior strength。敏感性上，\(\tau\in[0.045,0.08]\) 都保持 15/15 wins。

### 当前评价

这一步显著改善了解释性：原来最危险的 `7/20` 和 \(\tau=0.05\) 都有了 observable 或
identifiability-based 来源，并且有敏感性证据。但还不能说完全完成，因为：

- \(\mu=0.01\) 还只是 sensitivity-backed protocol，不是理论唯一值；
- 2.5-observation slow endpoint 仍需更精炼的文字解释；
- 需要把 exact no-round endpoint 与 rounded endpoint 的取舍写清楚；
- 外部 schedule 或新训练 run 仍然缺失。

## 2026-06-19: short-cosine 控制与 localized deployable 公式

新增 `repro/interpretable_strict_vs_rounded_audit.py`，比较：

- strict exact endpoint: \(\lambda_{\mathrm{fast}}=\lambda_{\mathrm{obs}}\)；
- rounded fast endpoint: \(\lambda_{\mathrm{fast}}=20\)；
- localized variants；
- short cosine / constant extra controls。

### 发现的问题

unlocalized observation-half-life 公式在 WSD-family 上很好：

| variant | WSD mean | WSD worst | WSD wins |
|---|---:|---:|---:|
| strict exact | -31.97% | -1.09% | 15/15 |
| rounded fast 20 | -34.56% | -5.30% | 15/15 |

但它会伤害 `cosine_24000`：

| variant | Cosine 24k mean | worst | non-harm |
|---|---:|---:|---:|
| strict exact | +42.06% | +56.43% | 0/3 |
| rounded fast 20 | +42.06% | +56.43% | 0/3 |

constant controls 没问题，因为 positive LR drop feature 为零，correction 自动为零。

这个结果说明：当前公式确实捕捉了 WSD-family 的局部 cooldown residual，但如果目标 schedule 是
full-run diffuse cosine decay，它不应该被当成局部非绝热 transient 来修正。

### 无拟合参数的 locality 修正

不能加 gate，也不应该按 schedule 类型分类。因此采用一个连续的 schedule-locality factor。

令

\[
d_t=[\eta_{t-1}-\eta_t]_+,
\]

positive-drop support span 为

\[
\ell_s=\max\{t:d_t>0\}-\min\{t:d_t>0\}+2.
\]

其中 `+2` 是因为 \(d_t\) 表示从 \(t-1\) 到 \(t\) 的 LR transition，support 要包含 transition 的两个端点。
post-warmup 长度为 \(T_s-W\)。定义

\[
a_s=
\left[
1-\frac{\ell_s}{T_s-W}
\right]_+.
\]

最终 deployable 公式变为

\[
\hat L_s(t)=
L_{\mathrm{MPL},s}(t)
+
a_s\hat\kappa_s\phi_{\lambda_s,s}(t).
\]

解释：

- WSD sharp / linear 的 cooldown 只占 post-warmup 的约 18%，因此 \(a_s\approx0.817\)；
- WSD-con 是 single-step drop，\(a_s\approx1\)；
- cosine 24k 的 LR 在整个 post-warmup 区间 diffuse decay，\(a_s=0\)；
- constant schedule 没有 positive drop，\(\phi=0\)。

这不是 gate，因为没有可调阈值，也不是按曲线类型分类；它只是把“局部 transient correction”限制在
LR drop 局部发生的 schedule 上。

### localized 结果

| variant | group | mean | worst | wins/non-harm |
|---|---|---:|---:|---:|
| rounded fast 20, unlocalized | WSD | -34.56% | -5.30% | 15/15 |
| rounded fast 20, sqrt-localized | WSD | -34.15% | -5.30% | 15/15 |
| rounded fast 20, linear-localized | WSD | -32.83% | -5.30% | 15/15 |
| rounded fast 20, unlocalized | extra controls | +14.02% | +56.43% | 6/9 non-harm |
| rounded fast 20, sqrt-localized | extra controls | +0.00% | +0.00% | 9/9 non-harm |

Per-target sqrt-localized:

| target | mean | worst | wins |
|---|---:|---:|---:|
| WSD sharp | -51.72% | -40.88% | 3/3 |
| WSD linear | -43.80% | -32.99% | 3/3 |
| WSD-con 3e-5 | -53.08% | -41.41% | 3/3 |
| WSD-con 9e-5 | -13.04% | -5.30% | 3/3 |
| WSD-con 18e-5 | -9.11% | -6.43% | 3/3 |

### localization shape sensitivity

新增 `repro/interpretable_localization_sensitivity.py` 后，结论更清楚：

| power \(p\) in \(a_s=\alpha_s^p\) | WSD mean | WSD worst | extra non-harm |
|---:|---:|---:|---:|
| 0, no localization | -34.56% | -5.30% | 6/9 |
| 0.5, sqrt amplitude | -34.15% | -5.30% | 9/9 |
| 1.0, linear conservative | -32.83% | -5.30% | 9/9 |
| 2.0 | -29.84% | -5.30% | 9/9 |

因此当时建议：把 sqrt-localized 版本作为 slides/paper 的主公式；把 unlocalized 版本作为 WSD-only
upper bound，把 linear-localized 作为保守下界。这个建议现已撤回，见下一节。sqrt 的解释是
energy-locality fraction 到 amplitude-locality factor 的转换，不是额外拟合参数。

## 2026-06-19: 解释性重置，撤回 sqrt-localized 主公式建议

上面的建议现在撤回。原因不是指标不好，而是解释性不够硬：`sqrt-localized` 虽然没有新增拟合参数，
但 energy-to-amplitude 的说法仍然像事后安全项。对于当前作业和后续科研叙事，主线不能依赖这种
解释偏软的 amplitude factor。

新的主线记录在 `results/interpretable_error_model/INTERPRETABILITY_RESET.md`。核心公式收缩为：

\[
\hat L_s(t)
=
L_{\mathrm{MPL},s}(t)
+
\hat\kappa_s\phi_{\lambda_s,s}(t).
\]

只保留一个 response term：

\[
\phi_{\lambda,s}(t)
=
\sum_{u\le t}
\exp[-\lambda\eta_u]
\frac{[\eta_{u-1}-\eta_u]_+}{\eta_{\max}}.
\]

当前解释性优先的版本排序：

| version | role | WSD mean | WSD worst | wins |
|---|---|---:|---:|---:|
| fixed_lambda_obs | minimal sanity | -20.55% | -1.09% | 15/15 |
| fixed_lambda_20 | minimal rounded | -22.06% | -5.30% | 15/15 |
| strict_exact | recommended theory | -31.97% | -1.09% | 15/15 |
| rounded_fast20 | performance variant | -34.56% | -5.30% | 15/15 |
| rounded_fast20_localized | optional control-safety caveat | -32.83% | -5.30% | 15/15 |
| rounded_fast20_sqrtlocalized | ablation only | -34.15% | -5.30% | 15/15 |

当前写作建议：

- 主文只讲 `MPL + causal LR-drop response`；
- 先展示 fixed observed-half-life kernel 已经 15/15 改善，说明机制本身有效；
- 再展示 schedule-geometry \(\lambda_s(q_s)\) 带来更强 WSD transfer；
- `linear-localized` 只在讨论 short-cosine control 时作为 deployment caveat；
- `sqrt-localized`、gate、channel routing、正弦展开、curvature patch 都不进入主公式。

这个版本牺牲了一点指标，但科研叙事更稳：读者可以清楚知道唯一新增机制是什么、唯一拟合参数是什么、
训练和测试各用了哪些信息。

## 2026-06-19: tau-free shrinkage baseline

新增 `repro/interpretable_shrinkage_origin_audit.py`，专门回答一个关键问题：当前机制是否依赖
固定 ridge 常数 \(\tau=0.05\)。

令

\[
x=M_\mu\phi_{\lambda,\cos},
\qquad
y=M_\mu r_{\cos},
\qquad
\phi=\phi_{\lambda,\cos}.
\]

原 ridge version 是

\[
\hat\kappa=
\frac{\langle x,y\rangle_+}{\|x\|^2+0.05^2}.
\]

新的 tau-free sqrt-retention version 是

\[
\hat\kappa=
\frac{\langle x,y\rangle_+}{\|x\|\|\phi\|}.
\]

解释：numerator 只使用 nuisance projection 后可辨认的 response direction；denominator 同时看
projected norm 和 full feature norm。如果 response feature 主要被 nuisance projection 吃掉，那么
\(\|x\|/\|\phi\|\) 很小，coefficient 自动收缩。不需要固定 \(\tau\)。

### 结果

| role | response | shrinkage | locality | WSD mean | WSD worst | WSD wins | controls |
|---|---|---|---|---:|---:|---:|---:|
| hard interpretable baseline | fixed \(\lambda=20\) | tau-free sqrt-retention | linear | -20.77% | -5.86% | 15/15 | 9/9 non-harm |
| conservative lower bound | fixed \(\lambda=20\) | tau-free full-energy | linear | -3.72% | -1.30% | 15/15 | 9/9 non-harm |
| performance extension | two-observation slow + fast 20 | ridge \(\tau=0.05\) | linear | -29.82% | -5.30% | 15/15 | 9/9 non-harm |
| old high-performance reference | 2.5-observation slow + fast 20 | ridge \(\tau=0.05\) | linear | -32.83% | -5.30% | 15/15 | 9/9 non-harm |

Per-target hard baseline:

| target | mean | worst | wins |
|---|---:|---:|---:|
| WSD sharp | -13.53% | -10.94% | 3/3 |
| WSD linear | -12.19% | -9.21% | 3/3 |
| WSD-con 3e-5 | -54.28% | -48.23% | 3/3 |
| WSD-con 9e-5 | -14.40% | -8.11% | 3/3 |
| WSD-con 18e-5 | -9.46% | -5.86% | 3/3 |

### 当前判断

这个结果很重要：一个 causal LR-drop response 的机制不依赖固定 \(\tau\) 才成立。即使去掉
schedule-geometry tuning 和 fixed ridge，仍然对所有 WSD rows 正优化，并且 controls 不受伤。

因此新的主叙事应分两层：

1. tau-free hard baseline：证明机制可信，解释性最强；
2. ridge performance extension：提高平均收益，但需要额外解释 \(\tau\) 的 protocol 地位。

这比直接把 \(\tau=0.05\) 的高性能版本作为主公式更稳。

## 2026-06-19: nuisance origin audit, DCT vs MPL tangent

新增 `repro/interpretable_nuisance_origin_audit.py`，检查当前 DCT nuisance projection 是否可以替换成
更机制化的 MPL tangent projection。

动机：

\[
r_{\cos}(t)
=
\kappa\phi(t)
+
J_{\mathrm{MPL}}(t)\Delta\theta
+
\epsilon(t).
\]

如果 cosine residual 中的 nuisance drift 主要来自 MPL 参数误差，那么应该把 response feature 和 residual
都投影到 MPL tangent space 的正交补后再估计 \(\kappa\)。这比 generic low-frequency DCT filtering
更可解释。

测试的 nuisance space：

| nuisance | meaning |
|---|---|
| `dct_soft` | current soft DCT low-frequency nuisance, 8 modes, \(\mu=0.01\) |
| `mpl_core3` | tangent of \(L_0,A,\alpha\) |
| `mpl_ld4` | tangent of LR-dependent MPL parameters \(B,C,\beta,\gamma\) |
| `mpl_all7` | all seven local MPL parameter directions |

### 结果

| variant | WSD mean | WSD worst | wins | controls |
|---|---:|---:|---:|---:|
| DCT, tau-free hard baseline | -20.77% | -5.86% | 15/15 | 9/9 non-harm |
| MPL-core tangent, tau-free | +15.19% | +111.37% | 7/15 | 9/9 non-harm |
| MPL-all tangent, tau-free | -4.92% | -1.47% | 15/15 | 9/9 non-harm |
| DCT, ridge performance reference | -32.83% | -5.30% | 15/15 | 9/9 non-harm |
| MPL-LD tangent, ridge | -27.25% | -3.00% | 15/15 | 9/9 non-harm |
| MPL-core tangent, ridge | +54.21% | +202.73% | 4/15 | 9/9 non-harm |

### 判断

1. `mpl_core3` 很差，说明 cosine residual 中会污染 \(\kappa\) 的成分不是简单 backbone
   \(L_0,A,\alpha\) 参数误差。
2. `mpl_all7` 过度投影，保留机制但收益很弱。
3. `mpl_ld4 + ridge` 是一个有价值的机制化替代：它去掉 MPL 的 LR-dependent tangent directions，
   仍然达到 15/15 WSD improvement 和 controls 9/9 non-harm，mean -27.25%。它弱于 DCT performance
   reference，但解释性更接近 MPL 误差分解。

当前写法建议：

- 主线仍保留 DCT tau-free hard baseline，因为它证明 response 机制最稳；
- 在 robustness/ablation 中加入 `mpl_ld4 + ridge`，说明即使用 MPL tangent nuisance 替代 DCT，也能得到全面正优化；
- 不把 `mpl_core3` 写成可行方向，它是负证据。

## 2026-06-19: current interpretable error-curve visualization

新增 `repro/plot_interpretable_error_model.py`，把当前三条可解释路线放到同一组 residual error 图里：

1. tau-free hard baseline；
2. MPL-LD tangent nuisance + ridge；
3. DCT performance extension。

输出目录：

`results/interpretable_error_model/error_comparison/`

### 汇总结果

| group | method | mean | worst | wins/non-harm |
|---|---|---:|---:|---:|
| WSD-family | tau-free hard | -20.77% | -5.86% | 15/15 |
| WSD-family | MPL-LD tangent | -27.25% | -3.00% | 15/15 |
| WSD-family | DCT performance | -32.83% | -5.30% | 15/15 |
| controls | tau-free hard | +0.00% | +0.00% | 9/9 non-harm |
| controls | MPL-LD tangent | +0.00% | +0.00% | 9/9 non-harm |
| controls | DCT performance | +0.00% | +0.00% | 9/9 non-harm |

### 图上的读数

- WSD sharp / linear：MPL-LD tangent 和 DCT performance 明显压低 cooldown 后的正残差；tau-free hard baseline 更保守。
- WSD-con 3e-5：tau-free hard baseline 反而最强，说明单步大 drop 的主要误差可以由非常干净的 fixed-\(\lambda=20\) response 解释。
- WSD-con 9e-5 / 18e-5：三种方法都有正收益，但幅度变小，说明高 final LR 下 cooldown transient 本来较弱。
- Cosine 24k：linear locality 让三种 correction 都为零，误差曲线与 MPL 重合，这是预期的 control behavior。

这组图支撑当前叙事：机制本身不是来自复杂拟合；强性能主要来自 DCT nuisance 和 ridge extension，而可解释替代 MPL-LD tangent 也能保持全面正优化。

## 2026-06-19: interpretability repair, DCT demoted

用户指出当前模型解释性仍然太差。这个批评成立：虽然 DCT tau-free baseline 可以证明某种 response 机制存在，但 DCT 本身只是 generic low-frequency residualizer，不能清楚说明被去掉的 residual 对应 MPL 的哪类误差。因此主线再次收缩。

### 新增审计

1. 更新 `repro/interpretable_nuisance_origin_audit.py`：
   - 加入 `none` nuisance，作为 raw one-dimensional projection failure mode；
   - 加入 no-locality 对照；
   - 明确 `mpl_ld4` 是当前 mechanism-native main candidate；
   - 明确 DCT 只是 performance reference。

2. 新增 `repro/interpretable_scale_stability_audit.py`：
   - 检查同 scale 与跨 scale cosine calibration；
   - 对比 `mpl_ld_tangent`、`dct_performance`、`tau_free_dct`；
   - 输出 `results/interpretable_scale_stability_audit/`。

3. 新增 `results/interpretable_error_model/MODEL_DECISION.md`：
   - 用中文重写当前可解释模型；
   - 直接承认旧模型的问题；
   - 给出 MPL-LD tangent projection 公式、训练/测试流程、结果和局限。

### 关键结果

| variant | WSD mean | WSD worst | wins | note |
|---|---:|---:|---:|---|
| no nuisance raw projection | +672.31% | +2585.94% | 0/15 | 证明 cosine residual 不能直接投影 |
| MPL-LD tangent, no locality | -24.86% | -3.00% | 15/15 | WSD 核心机制不依赖 locality |
| MPL-LD tangent + linear locality | -27.25% | -3.00% | 15/15 | controls 9/9 non-harm |
| DCT performance reference | -32.83% | -5.30% | 15/15 | 数值强，但解释性弱 |

跨 scale:

| method | same-scale WSD | cross-scale WSD | cross-scale worst |
|---|---:|---:|---:|
| MPL-LD tangent | -27.25%, 15/15 | -23.07%, 30/30 | -2.07% |
| DCT performance | -32.83%, 15/15 | -18.98%, 26/30 | +26.68% |
| tau-free DCT | -20.77%, 15/15 | -13.27%, 27/30 | +9.04% |

### 当前决定

主线改为：

\[
r_{\cos}(t)
=
\kappa\phi_{\lambda,\cos}(t)
+
J_{\mathrm{LD}}(t)\Delta\theta_{\mathrm{LD}}
+
\epsilon(t),
\]

\[
x=(I-P_{\mathrm{LD}})\phi_{\lambda,\cos},
\qquad
y=(I-P_{\mathrm{LD}})r_{\cos},
\]

\[
\hat\kappa
=
\frac{\langle x,y\rangle_+}{\|x\|^2+\tau^2}.
\]

其中 \(J_{\mathrm{LD}}\) 是 MPL 的 LR-dependent 参数 \((B,C,\beta,\gamma)\) 的 tangent space。这个版本比 DCT 少一点性能，但可解释性更强，跨 scale 稳定性也更好。DCT、sqrt-locality、gate、channel、正弦展开都不再作为主方法。

## 2026-06-19: observation-bracket MPL-LD, removing fixed tau and old endpoints

上一版 MPL-LD tangent 仍有两个不够干净的 protocol constants：

1. \(\tau=0.05\)；
2. \(\lambda_{\mathrm{slow}}=\lambda_{\mathrm{obs}}/2.5\)，fast endpoint rounded to \(20\)。

新增 `repro/interpretable_observation_bracket_audit.py`，把这两个常数替换为 observation / finite-sample quantities。

### 新 response rate

定义 modal observation interval \(\Delta_{\mathrm{obs}}\)：

\[
\lambda_{\mathrm{obs}}
=
\frac{\log 2}{\eta_{\max}\Delta_{\mathrm{obs}}}.
\]

定义 schedule drop concentration：

\[
q_s=
\frac{\max_t[\eta_{t-1}-\eta_t]_+}
{\sum_t[\eta_{t-1}-\eta_t]_+}.
\]

新规则：

\[
\lambda_s
=
\lambda_{\mathrm{obs}}\frac{1+q_s}{2}.
\]

等价地，response half-life 为

\[
H_s=\frac{2\Delta_{\mathrm{obs}}}{1+q_s}.
\]

解释：diffuse LR decay 使用 two-observation half-life，single sharp drop 使用 one-observation half-life。这个 rule 不使用 target loss，也不再需要 2.5 或 20。

### 新 ridge

MPL-LD tangent projection 后：

\[
x_s=(I-P_{\mathrm{LD}})\phi_{\lambda_s,\cos},
\qquad
y=(I-P_{\mathrm{LD}})r_{\cos}.
\]

估计：

\[
\hat\kappa_s
=
\frac{\langle x_s,y\rangle_+}{\|x_s\|^2+1/N_{\mathrm{cal}}}.
\]

\(N_{\mathrm{cal}}\) 是 cosine calibration suffix 的点数。它替代固定 \(\tau=0.05\)，作为 finite-sample identifiability floor。

### Source-only suffix rule

fit_start 不再只说成手写 8000。对候选 suffix starts，计算

\[
\rho=
\frac{\|(I-P_{\mathrm{LD}})\phi\|^2}{\|\phi\|^2}.
\]

选择最早使所有 source scale 和 observation bracket 两个端点 \(\lambda_{\mathrm{obs}}/2,\lambda_{\mathrm{obs}}\) 上的 \(\rho\le1/N_{\mathrm{cal}}\) 成立的 suffix start。当前实现不枚举 WSD target schedule。额外 dense-grid 审计显示 2、3、5、9、17、33、65、129 个 \(\lambda\) 点都会选出 8000，因此 endpoint-only 规则不是网格分辨率偶然。

结果：

| fit start | lambda points | max retention | floor | passes |
|---:|---:|---:|---:|---:|
| 5000 | 2 | 0.01375 | 0.00191 | 0 |
| 6500 | 2 | 0.00423 | 0.00195 | 0 |
| 8000 | 2 | 0.00148 | 0.00200 | 1 |
| 10000 | 2 | 0.00049 | 0.00207 | 1 |
| 12000 | 2 | 0.00035 | 0.00213 | 1 |

因此自动选择最早通过点 8000。

### 结果

| variant | same-scale WSD | cross-scale WSD | controls |
|---|---:|---:|---:|
| observation-bracket MPL-LD + sample-size ridge | -29.87%, 15/15 | -24.95%, 30/30 | 9/9 non-harm |
| old MPL-LD fixed tau / old lambda | -27.25%, 15/15 | -23.07%, 30/30 | 9/9 non-harm |
| no nuisance raw projection | +602.17%, 0/15 | +572.52%, 0/30 | not viable |
| observation-bracket MPL-LD, no locality | -30.89%, 15/15 | not main | controls worst +56.99% |

Fit-start sensitivity of main variant:

| fit start | WSD mean | worst | wins |
|---:|---:|---:|---:|
| 5000 | +123.32% | +399.52% | 0/15 |
| 6500 | +11.88% | +90.39% | 5/15 |
| 8000 | -29.87% | -4.67% | 15/15 |
| 10000 | -8.53% | -1.37% | 15/15 |
| 12000 | -1.49% | +0.00% | 6/15 wins, 15/15 non-harm |

当前判断：observation-bracket MPL-LD 是比旧 MPL-LD 更好的主候选。它去掉了固定 \(\tau\)、2.5、20，并且 same-scale / cross-scale 都强于旧版本。主要弱点变成：two-observation bracket 和 source-only suffix rule 仍需在写作中谨慎表述为 observation-resolution prior，而不是定理。

## 2026-06-19: 解释性审计后降级 observation-bracket

用户指出当前模型解释性仍然太差。这个批评是对的。虽然 observation-bracket MPL-LD 数值强，但它仍然在 MPL 外部增加了 response basis，并依赖 response-rate prior、MPL-LD tangent projection、fit-start protocol 和 locality boundary。它们都能解释，但组合起来仍像 protocol engineering，不像一个从 MPL 本身推出的主模型。

因此当前结论调整为：

1. observation-bracket MPL-LD 降级为强诊断/消融参考；
2. DCT、sine、gate、channel、sqrt-locality 继续排除出主线；
3. 下一条主线必须直接从 MPL 最后一项 \(B D(t)\) 出发。

新增 `repro/mpl_ld_lag_response_audit.py` 测试最干净的 finite-response 版本：

\[
D_\tau(t_i)
=
\rho_iD_\tau(t_{i-1})
+(1-\rho_i)D(t_i),
\qquad
\rho_i=\exp[-(t_i-t_{i-1})/\tau],
\]

\[
\hat L_\tau(t)
=
L_{\mathrm{MPL}}(t)
+B[D_\tau(t)-D(t)].
\]

固定 \(\tau\)、不从 residual 拟合幅度时：

| \(\tau\) steps | WSD mean | WSD worst | wins | controls worst |
|---:|---:|---:|---:|---:|
| 32 | -0.42% | -0.28% | 15/15 | +0.19% |
| 64 | -3.11% | -2.38% | 15/15 | +1.62% |
| 128 | -9.52% | -5.95% | 15/15 | +6.01% |
| 256 | -15.09% | +18.57% | 14/15 | +15.90% |
| 512 | -6.59% | +92.63% | 10/15 | +36.20% |

这个结果说明 finite-response 机制有真实信号，但还不够安全。小 \(\tau\) 解释性强且全 WSD 正优化，但收益小；\(\tau=128\) 收益中等但 controls harm；更大 \(\tau\) 明显过修正。

负控更重要：如果对同一个 \(B[D_\tau-D]\) feature 从 cosine residual 拟合一个 amplitude，\(\tau=128\) 会得到 same-scale WSD `+565.16%`、cross-scale WSD `+548.27%`、0 wins。说明 cosine residual 里仍有严重全局 MPL drift，不能让 amplitude 自由吸收 residual。

当前最诚实的判断：

- 还没有最终可解释主模型；
- \(B[D_\tau-D]\) 是下一步最值得沿着讲的公式；
- observation-bracket MPL-LD 只能说明“如果做 nuisance control，WSD 数值可以显著改善”，不能作为最终理论。

## 2026-06-19: signed MPL-LD decomposition and adiabatic boundary

继续沿着 MPL 最后一项做更强解释性收缩。将 MPL 的

\[
D(t)=\sum_{v\le t}\Delta\eta_v h_v(t)
\]

按 LR 变化方向拆成

\[
D(t)=D_\uparrow(t)+D_\downarrow(t),
\]

其中 \(D_\uparrow\) 来自 warmup / LR increase，\(D_\downarrow\) 来自 cooldown / LR decrease。

只对 \(D_\downarrow\) 做 finite response：

\[
\hat L_\tau(t)
=
L_{\mathrm{MPL}}(t)
+B[D_{\downarrow,\tau}(t)-D_\downarrow(t)].
\]

这个拆分不是新 basis，也不是 channel route，而是 MPL 自己的 LD sum 中 \(\Delta\eta\) 的符号分解。结果：

| variant, \(\tau=128\) | WSD mean | WSD worst | wins | controls |
|---|---:|---:|---:|---:|
| full \(D_\tau-D\) | -9.52% | -5.95% | 15/15 | worst +6.01% |
| cooldown only | -9.44% | -6.22% | 15/15 | worst +6.53% |

cooldown-only 让 constant controls 精确为 0，但 400M short-cosine 仍有 harm，说明 diffuse full-horizon cosine decay 仍被当成 local cooldown transient。

因此加入 schedule-only adiabatic boundary：

\[
a_s=\left[1-\frac{\ell_\downarrow}{T-W}\right]_+,
\]

\[
\hat L_\tau(t)
=
L_{\mathrm{MPL}}(t)
+a_sB[D_{\downarrow,\tau}(t)-D_\downarrow(t)].
\]

这里 \(\ell_\downarrow\) 是 post-warmup LR-drop support span。这个项不拟合，也不读取 target loss；它表达边界条件：full-horizon diffuse decay 应视为 quasi-adiabatic，不应套用 local cooldown transient。

结果：

| \(\tau\) | WSD mean | WSD worst | wins | controls |
|---:|---:|---:|---:|---:|
| 64 | -2.91% | -1.96% | 15/15 | 9/9 non-harm |
| 128 | -8.73% | -6.22% | 15/15 | 9/9 non-harm |
| 256 | -13.52% | +14.96% | 14/15 | 9/9 non-harm |

继续测试 schedule-only support-bracket 响应时间：

\[
\tau_s
=
\Delta_{\mathrm{obs}}
\left(
1+\min(1,\ell_\downarrow/\Delta_{\mathrm{obs}})
\right).
\]

解释：single-step cooldown 只占很小的 observation window，因此响应接近一个 observation interval；持续超过一个 observation interval 的 cooldown 使用 two-observation response upper bound。它不看 loss，也不按 curve label route。

结果：

| candidate | WSD mean | WSD worst | wins | controls |
|---|---:|---:|---:|---:|
| fixed \(\tau=128\) cooldown adiabatic | -8.73% | -6.22% | 15/15 | 9/9 non-harm |
| support-bracket \(\tau_s\) cooldown adiabatic | -13.77% | -6.29% | 15/15 | 9/9 non-harm |

有效 \(\tau_s\)：WSD sharp/linear 为 256，WSD-con 为 130。当前最干净候选应改为 support-bracket cooldown adiabatic MPL-LD lag。它的优点是没有 residual-fitted 参数，\(\tau_s\) 和 \(a_s\) 都来自 LR schedule / logging resolution，所有 WSD 正优化且 controls non-harm。弱点是收益仍低于 observation-bracket MPL-LD，并且 \(a_s\) 仍是 schedule-level boundary，需要在理论中谨慎表述，不能包装成从 MPL 唯一推出的定理。

新增 rule sensitivity 审计，所有行都不拟合 residual 参数：

| rule | explanation | WSD mean | worst | wins | controls |
|---|---|---:|---:|---:|---:|
| fixed one observation | strong, conservative | -8.73% | -6.22% | 15/15 | 9/9 |
| support linear bracket | strong, recommended | -13.77% | -6.29% | 15/15 | 9/9 |
| support hard two-observation | medium | -13.70% | -6.22% | 15/15 | 9/9 |
| support sqrt bracket | weaker nonlinear prior | -14.15% | -6.79% | 15/15 | 9/9 |
| support log bracket | weaker nonlinear prior | -14.36% | -6.65% | 15/15 | 9/9 |
| fixed two observations | strong but unsafe | -13.52% | +14.96% | 14/15 | 9/9 |

结论：support-linear bracket 不是唯一能工作的点，但它是解释性最硬且安全的折中。sqrt/log 更强但引入非线性 prior，暂时只作为 robustness reference。

新增 amplitude sensitivity 审计。推荐模型使用 MPL 自己的 \(B_s\)，等价于 amplitude scale \(c=1\)。固定 \(c\) 不从 residual 或 target loss 拟合：

| amplitude scale | WSD mean | worst | wins | controls |
|---:|---:|---:|---:|---:|
| 0.25 | -3.93% | -2.09% | 15/15 | 9/9 |
| 0.50 | -7.86% | -4.18% | 15/15 | 9/9 |
| 0.75 | -11.25% | -6.27% | 15/15 | 9/9 |
| 1.00 | -13.77% | -6.29% | 15/15 | 9/9 |
| 1.25 | -15.17% | -5.90% | 15/15 | 9/9 |
| 1.50 | -15.95% | -3.43% | 15/15 | 9/9 |
| 2.00 | -15.61% | +3.76% | 14/15 | 9/9 |

结论：\(c=1\) 不是精确 \(B_s\) 上的孤立巧合，`0.25` 到 `1.50` 都保持 all-win。更大的 scale 平均更强但会过修正，不能作为推荐模型，因为那会重新引入 amplitude selection。
