# 恢复 Observation-Bracket MPL-LD 主线说明

## 1. 为什么不继续零参数 finite-response 版本

零参数 cooldown finite-response 版本是：

\[
\hat L_s(t)
=
L_{\mathrm{MPL},s}(t)
+
a_sB_s[D_{\downarrow,\tau_s,s}(t)-D_{\downarrow,s}(t)].
\]

它的优点是解释性很干净：没有 residual-fitted 参数，只修改 MPL 自己的 cooldown 子项。但实验证据说明它太保守：

| protocol | WSD mean | WSD worst | wins | controls |
|---|---:|---:|---:|---:|
| frozen official MPL | -13.77% | -6.29% | 15/15 | 9/9 non-harm |
| strict cosine-only MPL | -11.44% | -6.40% | 15/15 | 9/9 non-harm |

strict cosine-only 下虽然相对自己 baseline 全部正优化，但修正后仍比 official MPL baseline 平均差 `+37.34%`。这说明它是一个有价值的机制下界，但不够支撑当前工作目标。

因此现在不再把它作为主模型，而是保留为：

- 机制 sanity check；
- 说明 MPL 的 cooldown 项确实存在 finite-response 信号；
- 证明“只改 MPL 最后一项”方向过于保守。

## 2. 恢复的主公式

当前主线恢复为 observation-bracket MPL-LD：

\[
\hat L_s(t)
=
L_{\mathrm{MPL},s}(t)
+
a_s\hat\kappa_s\phi_{\lambda_s,s}(t).
\]

这里真正从 residual 中拟合的只有一个非负标量 \(\hat\kappa_s\)。其它量都来自 LR schedule、logging resolution、source calibration size 或 MPL 公式本身。

## 3. Response feature

定义 positive LR drop：

\[
d_t=[\eta_{t-1}-\eta_t]_+.
\]

causal LR-drop response feature：

\[
\phi_{\lambda,s}(t)
=
\sum_{u\le t}
\exp[-\lambda\eta_u]
\frac{d_u}{\eta_{\max}}.
\]

这个 feature 的约束是：

- causal：只用 \(u\le t\) 的过去 LR；
- drop-only：constant LR 不产生 correction；
- single response：不引入 family label、channel 选择、gate、正弦或 DCT residual basis。

## 4. Observation-bracket response rate

先由 loss logging interval 给出基本响应尺度：

\[
\lambda_{\mathrm{obs}}
=
\frac{\log 2}{\eta_{\max}\Delta_{\mathrm{obs}}},
\qquad
\Delta_{\mathrm{obs}}=128.
\]

对目标 schedule 定义 drop concentration：

\[
q_s=
\frac{\max_t d_t}{\sum_t d_t}.
\]

响应率：

\[
\lambda_s
=
\lambda_{\mathrm{obs}}\frac{1+q_s}{2}.
\]

等价地，响应 half-life 是：

\[
H_s=
\frac{2\Delta_{\mathrm{obs}}}{1+q_s}.
\]

解释：

- diffuse LR decay：\(q_s\approx0\)，使用 two-observation half-life；
- single sharp drop：\(q_s=1\)，使用 one-observation half-life；
- WSD / WSD-con 位于这两者之间。

这个 rule 不使用 WSD loss，也不再使用旧版本里的固定 `2.5` slow endpoint 或 rounded fast endpoint `20`。

## 5. 为什么需要 MPL-LD tangent projection

cosine residual 不是纯误差信号。它至少包含两部分：

\[
r_{\cos}(t)
=
\kappa\phi_{\lambda,\cos}(t)
+
J_{\mathrm{MPL}}(t)\Delta\theta
+
\epsilon(t).
\]

如果直接用 \(r_{\cos}\) 拟合 \(\kappa\)，会把 MPL 参数误差也吸收到 \(\kappa\) 里，迁移到 WSD 后会严重过拟合。无 nuisance 的审计已经显示：

| variant | WSD mean | WSD worst | wins |
|---|---:|---:|---:|
| no nuisance raw projection | +602.17% | +2366.35% | 0/15 |

因此当前只投影掉 MPL 的 LR-dependent tangent：

\[
J_{\mathrm{LD}}
=
\left[
\frac{\partial L_{\mathrm{MPL}}}{\partial\log B},
\frac{\partial L_{\mathrm{MPL}}}{\partial\log C},
\frac{\partial L_{\mathrm{MPL}}}{\partial\log \beta},
\frac{\partial L_{\mathrm{MPL}}}{\partial\log \gamma}
\right].
\]

令 \(P_{\mathrm{LD}}\) 是这个 tangent space 的正交投影。定义：

\[
x_s=(I-P_{\mathrm{LD}})\phi_{\lambda_s,\cos},
\qquad
y=(I-P_{\mathrm{LD}})r_{\cos}.
\]

然后估计：

\[
\hat\kappa_s
=
\frac{\langle x_s,y\rangle_+}{\|x_s\|_2^2+1/N_{\mathrm{cal}}}.
\]

其中 \(N_{\mathrm{cal}}\) 是 cosine calibration suffix 中样本数。这个 \(1/N_{\mathrm{cal}}\) 是 finite-sample identifiability floor，不是用 WSD loss 调出来的固定 ridge。

## 6. Locality boundary

为了避免把 local cooldown transient 转移到 full-horizon cosine decay，使用 schedule-only boundary：

\[
a_s=
\mathbf{1}\{\sum_t d_t>0\}
\left[
1-\frac{\ell_s}{T_s-W}
\right]_+.
\]

这里 \(\ell_s\) 是 warmup 后 LR drop support span。它不是 learned gate，不使用 loss。

无 locality 时 WSD 仍然好，但 controls 会坏：

| setting | WSD | controls |
|---|---:|---:|
| no locality | -30.89%, 15/15 | worst +56.99% |
| linear locality | -29.87%, 15/15 | 9/9 non-harm |

所以 locality 是安全边界，不是核心机制。

## 7. 当前主结果

| method | same-scale WSD | cross-scale WSD | controls |
|---|---:|---:|---:|
| observation-bracket MPL-LD | -29.87%, 15/15 | -24.95%, 30/30 | 9/9 non-harm |
| old MPL-LD fixed tau | -27.25%, 15/15 | -23.07%, 30/30 | 9/9 non-harm |
| zero-param finite-response | -13.77%, 15/15 | not main | 9/9 non-harm |
| DCT performance reference | -32.83%, 15/15 | -18.98%, 26/30 | 9/9 non-harm |

恢复后的判断：

- observation-bracket MPL-LD 是当前主候选；
- zero-param finite-response 是保守机制下界；
- DCT 是数值参考，不是主方法；
- gate/channel/正弦展开不恢复。

## 8. 当前还需要补强什么

这版仍然不是最终完成态。最需要补强的是：

1. 更明确解释 \(a_s\) 的 boundary 地位，避免被看成手工 gate；
2. 进一步验证 suffix rule 不是对当前 WSD target 的间接适配；
3. 检查 strict cosine-only backbone 下 observation-bracket 是否仍保持优势；
4. 最终写 slides 时，把主线讲成“cosine residual contamination -> MPL-LD tangent projection -> observation-bracket response”，而不是简单展示一个拟合公式。
