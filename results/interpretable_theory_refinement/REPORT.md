# Interpretable Theory Refinement Audit

本 audit 不改变核心架构，只检查两个 schedule-only 规则是否可以更理论化：

\[
\widehat L_s(t)=L_{\mathrm{MPL},s}(t)+a_s\widehat\kappa_s\phi_{\lambda_s,s}(t).
\]

其中 \(\widehat\kappa_s\) 仍然只从 cosine residual 经过 MPL-LD tangent projection 后的一维非负投影得到。目标 loss 只用于最后评价。

## 理论修正 1：drop concentration

令 \(d_t=[\eta_{t-1}-\eta_t]_+\)，\(D=\sum_t d_t\)，\(p_t=d_t/D\)。旧公式使用

\[
q_\infty=\|d\|_\infty/\|d\|_1=\max_t p_t.
\]

一个更有统计解释的替代是 Herfindahl concentration：

\[
q_2=\sum_t p_t^2=1/n_{\mathrm{eff}}.
\]

\(q_2\) 可以解释为 drop 分布的 effective atom count：单步 drop 时 \(q_2=1\)，均匀分布在 \(n\) 个 step 上时 \(q_2=1/n\)。
因此 response half-life 可以写成

\[
H_s=(2-q_2)\Delta_{\mathrm{obs}},
\qquad
\lambda_s=\frac{\lambda_{\mathrm{obs}}}{2-q_2}.
\]

这保持 observation bracket：diffuse drop 约为 two-observation half-life，single-step drop 为 one-observation half-life。

## 理论修正 2：locality factor

当前使用

\[
a_s=\mathbf{1}\{D>0\}\left[1-\frac{\ell_s}{T_s-W}\right]_+.
\]

这个项可以从投影解释出来，而不是把它叫作 gate。设 post-warmup horizon 为 \(H=T_s-W\)，
局部 drop support 上的均匀密度为 \(m_t=\mathbf{1}_{t\in\mathrm{supp}(d)}/\ell_s\)，全局 diffuse mode 为 \(u_t=1/H\)。
把局部 forcing 投影到 diffuse mode 的正交补上：

\[
\frac{\|(I-P_u)m\|_2^2}{\|m\|_2^2}=1-\frac{\ell_s}{H}.
\]

所以当前 \(a_s\) 可解释为：去掉 full-horizon adiabatic/diffuse forcing 后，局部 LR-drop forcing 保留下来的能量比例。
这也解释了为什么 full-horizon cosine control 应该被压到 0，而 WSD-con single drop 接近 1。

作为对照，本 audit 还测试了更细的 density projection：

\[
a_s^{\mathrm{density}}=\frac{\|(I-P_u)p\|_2^2}{\|p\|_2^2}.
\]

它更忠实于 drop density，但会把 cosine 的平滑非均匀下降也看成一部分 local signal；实验显示这对 controls 不够保守。

## Schedule Diagnostics

| curve | group | q_inf | q_2 | n_eff | support span | support proj | density proj |
|---|---|---:|---:|---:|---:|---:|---:|
| WSD sharp | core_wsd | 0.000639 | 0.000352 | 2842.3 | 4000 | 0.8168 | 0.8699 |
| WSD linear | core_wsd | 0.000250 | 0.000250 | 3999.0 | 4000 | 0.8168 | 0.8169 |
| WSD-con 3e-5 | core_wsd | 1.000000 | 1.000000 | 1.0 | 2 | 0.9999 | 0.9999 |
| WSD-con 9e-5 | core_wsd | 1.000000 | 1.000000 | 1.0 | 2 | 0.9999 | 0.9999 |
| WSD-con 18e-5 | core_wsd | 1.000000 | 1.000000 | 1.0 | 2 | 0.9999 | 0.9999 |
| Cosine 24k | extra_control | 0.000072 | 0.000056 | 17702.8 | 21840 | 0.0000 | 0.1894 |
| Constant 24k | extra_control | 0.000000 | 0.000000 | 0.0 | 0 | 0.0000 | 0.0000 |
| Constant 72k | extra_control | 0.000000 | 0.000000 | 0.0 | 0 | 0.0000 | 0.0000 |

## Result Summary

| variant | WSD same-scale | WSD cross-scale | controls same-scale | reading |
|---|---:|---:|---:|---|
| current_qinf_support_projection | -29.87% / -4.67% / 15/15 | -24.95% / -3.15% / 30/30 | +0.00% / +0.00% / 0/9 | 当前公式；现在可解释为 support projection。 |
| hhi_q2_support_projection | -29.88% / -4.67% / 15/15 | -24.95% / -3.15% / 30/30 | +0.00% / +0.00% / 0/9 | 更可解释的 q2；结果应与当前接近。 |
| hhi_q2_halflife_support_projection | -29.88% / -4.67% / 15/15 | -24.95% / -3.15% / 30/30 | +0.00% / +0.00% / 0/9 | half-life 线性插值；检查 bracket 解释敏感性。 |
| hhi_q2_density_projection | -30.22% / -4.67% / 15/15 | -24.93% / -3.15% / 30/30 | +1.87% / +8.25% / 0/9 | 更细 density projection；检查是否伤害 controls。 |
| hhi_q2_no_locality | -30.88% / -4.67% / 15/15 | -24.60% / -3.15% / 30/30 | +13.39% / +56.99% / 0/9 | 无边界负控。 |

## Decision

推荐把公式解释更新为 **q2 concentration + half-life bracket + support-projection locality**：

\[
q_s=\sum_t\left(\frac{d_t}{\sum_u d_u}\right)^2,
\qquad
H_s=(2-q_s)\Delta_{\mathrm{obs}},
\qquad
\lambda_s=\frac{\log 2}{\eta_{\max}H_s}
=\frac{\lambda_{\mathrm{obs}}}{2-q_s},
\]

\[
a_s=\mathbf{1}\{D_s>0\}\frac{\|(I-P_u)m_s\|_2^2}{\|m_s\|_2^2}
=\mathbf{1}\{D_s>0\}\left[1-\frac{\ell_s}{T_s-W}\right]_+.
\]

在当前数据上，该版本 WSD same-scale 为 `-29.88% / -4.67% / 15/15`，controls same-scale 为 `+0.00% / +0.00% / 0/9`。
它没有增加 residual-fitted 参数，仍然只有一个 \(\widehat\kappa_s\)。

解释上，\(q_2\) 比 \(q_\infty\) 更像 effective support size；直接插值 half-life 也比插值 rate 更符合 observation-bracket 叙事。
\(a_s\) 不再是经验 gate，
而是 local forcing 去掉 diffuse adiabatic mode 后的能量保留率。density projection 虽然自然，
但会保留 cosine 的平滑非均匀下降，从 controls 看不够保守，因此暂不作为主公式。
