# 可解释误差模型理论说明

本文档给出当前推荐公式的理论地位。目标不是声称已经推出完整优化动力学定理，而是在合理假设下解释每一项为什么存在、如何计算、为什么不引入额外自由度。

当前公式：

\[
\widehat L_s(t)
=
L_{\mathrm{MPL},s}(t)
+
a_s\widehat\kappa_s\phi_{\lambda_s,s}(t).
\]

其中唯一从 loss residual 拟合的量是 \(\widehat\kappa_s\)，并且只从 source cosine residual 拟合。其他量全部来自 LR schedule、观测间隔、MPL 公式或 source suffix 样本数。

## 1. 残差分解假设

对 schedule \(s\)，写

\[
r_s(t)=L_s(t)-L_{\mathrm{MPL},s}(t).
\]

我们假设 residual 中存在三类成分：

\[
r_s(t)
=
\kappa_s\phi_{\lambda_s,s}(t)
+
J_{\mathrm{LD},s}(t)\Delta\theta_{\mathrm{LD}}
+
\epsilon_s(t).
\]

含义：

- \(\kappa_s\phi_{\lambda_s,s}\)：LR drop 诱发的可迁移有限响应；
- \(J_{\mathrm{LD}}\Delta\theta_{\mathrm{LD}}\)：MPL learning-rate-dependent 参数误差的一阶近似；
- \(\epsilon_s\)：未建模噪声和剩余误差。

关键识别问题是：不能直接把 cosine residual 投影到 response feature 上，因为 cosine residual 里混有 MPL-LD drift。这个负控已经被实验确认：no-nuisance raw projection 在 same-scale WSD 上为 \(+602.17\%\)，0/15 wins。

## 2. Causal LR-Drop Response

定义 positive LR drop：

\[
d_{s,t}=[\eta_{s,t-1}-\eta_{s,t}]_+.
\]

令 response state 满足一阶线性方程：

\[
z_t=\exp(-\lambda_s\eta_{s,t})z_{t-1}+d_{s,t}.
\]

归一化后：

\[
\phi_{\lambda_s,s}(t)=z_t/\eta_{\max}.
\]

递推展开为：

\[
\phi_{\lambda_s,s}(t)
=
\frac{1}{\eta_{\max}}
\sum_{u\le t}
d_{s,u}
\exp\left(
-\lambda_s\sum_{v=u+1}^{t}\eta_{s,v}
\right).
\]

这个形式来自一个最小的 linear-response 假设：LR 下降给 residual state 一个 forcing，之后该 state 按 cumulative LR-time 衰减。它只使用过去 LR history，因此是 causal 的；constant LR 区间不产生新的 forcing。

## 3. 为什么用 \(q_2\) 决定响应时间

旧版本用

\[
q_\infty=\max_t d_t/\sum_u d_u.
\]

这个量能区分 single-step drop 和 diffuse drop，但解释性不够强，因为它只看最大一个 step。当前改用 drop mass 的 Herfindahl concentration：

\[
D_s=\sum_t d_{s,t},\qquad
p_{s,t}=d_{s,t}/D_s,
\qquad
q_s=\sum_t p_{s,t}^2.
\]

这等价于

\[
q_s=1/n_{\mathrm{eff},s}.
\]

如果 LR drop 均匀分布在 \(n\) 个 step，\(q_s=1/n\)；如果是单步 drop，\(q_s=1\)。因此 \(q_s\) 可以解释为 LR-drop forcing 的有效原子数，而不是经验分类标签。

## 4. Observation Half-Life Bracket

设 loss curve 的主要 logging interval 为 \(\Delta_{\mathrm{obs}}\)。定义一观测间隔半衰期对应的 rate：

\[
\lambda_{\mathrm{obs}}
=
\frac{\log 2}{\eta_{\max}\Delta_{\mathrm{obs}}}.
\]

当前假设：越集中的 drop，系统越应表现为短时局部响应；越 diffuse 的 drop，越接近慢的平滑响应。因此直接插值 half-life：

\[
H_s=(2-q_s)\Delta_{\mathrm{obs}},
\]

\[
\lambda_s
=
\frac{\log 2}{\eta_{\max}H_s}
=
\frac{\lambda_{\mathrm{obs}}}{2-q_s}.
\]

边界：

- \(q_s\approx0\)：diffuse decay，half-life 约两个 observation intervals；
- \(q_s=1\)：single-step drop，half-life 一个 observation interval。

这比直接插值 \(\lambda\) 更自然，因为假设对象是“可观测响应持续几个 logging interval”，不是 rate 本身。

## 5. MPL-LD Tangent Nuisance Projection

MPL 的 LR-dependent 部分包含参数 \((B,C,\beta,\gamma)\)。若这些参数有小误差，则

\[
\Delta L_{\mathrm{MPL}}(t)
\approx
J_{\mathrm{LD}}(t)\Delta\theta_{\mathrm{LD}},
\]

其中

\[
J_{\mathrm{LD}}(t)=
\left[
\frac{\partial L_{\mathrm{MPL}}}{\partial\log B},
\frac{\partial L_{\mathrm{MPL}}}{\partial\log C},
\frac{\partial L_{\mathrm{MPL}}}{\partial\log\beta},
\frac{\partial L_{\mathrm{MPL}}}{\partial\log\gamma}
\right].
\]

我们只投影 learning-rate-dependent tangent，而不投影 \(L_0,A,\alpha\) 的 backbone tangent。理由是：当前 correction 的目标是 schedule/LR 相关误差；投影过宽会把真正 response 也当成 nuisance。

在 cosine calibration suffix 上，令 \(P_{\mathrm{LD}}\) 为该 tangent space 的正交投影：

\[
x_s=(I-P_{\mathrm{LD}})\phi_{\lambda_s,\cos},
\qquad
y=(I-P_{\mathrm{LD}})r_{\cos}.
\]

然后估计一维非负 amplitude：

\[
\widehat\kappa_s
=
\frac{\langle x_s,y\rangle_+}
{\|x_s\|_2^2+1/N_{\mathrm{cal}}}.
\]

非负约束表达假设：LR-drop finite response 只沿同一方向修正 MPL；若投影后 alignment 为负，则不施加该 response。\(1/N_{\mathrm{cal}}\) 是有限样本可辨识性地板，不是调 target loss 得到的 ridge。

## 6. Support-Projection Locality

如果不加边界，WSD-family 仍然全胜，但 short-cosine controls 会明显受伤。这说明 correction 是局部 cooldown response，不能转移到 full-horizon diffuse cosine decay。

这个边界可以写成投影，而不是 gate。设 post-warmup horizon 为

\[
H_{\mathrm{post}}=T_s-W.
\]

令 \(\ell_s\) 为 positive-drop support span。令 \(m_s\) 是 support 上的均匀 forcing density，\(u_s\) 是整个 post-warmup horizon 上的均匀 diffuse mode。去掉 diffuse mode 后保留的能量比例为：

\[
a_s
=
\mathbf{1}\{D_s>0\}
\frac{\|(I-P_{u_s})m_s\|_2^2}{\|m_s\|_2^2}.
\]

直接计算可得：

\[
a_s
=
\mathbf{1}\{D_s>0\}
\left[1-\frac{\ell_s}{H_{\mathrm{post}}}\right]_+.
\]

因此 \(a_s\) 的含义不是“选择某类 schedule”，而是：从局部 drop forcing 中移除 full-horizon diffuse component，只把局部 transient 能量用于 correction。

## 7. 训练与测试过程

Calibration：

1. 读取 source `cosine_72000.csv`。
2. 计算 \(r_{\cos}=L_{\cos}-L_{\mathrm{MPL},\cos}\)。
3. 使用 source-only suffix rule 选择 \(t\ge8000\)。
4. 对每个目标 schedule，由 LR schedule 计算 \(q_s,H_s,\lambda_s,a_s\)。
5. 在 source cosine 上构造 \(\phi_{\lambda_s,\cos}\)。
6. 通过 MPL-LD tangent projection 得到 \(x_s,y\)。
7. 计算 \(\widehat\kappa_s\)。

Prediction：

1. 由目标 LR schedule 构造 \(\phi_{\lambda_s,s}(t)\)。
2. 输出

\[
\widehat L_s(t)
=
L_{\mathrm{MPL},s}(t)
+
a_s\widehat\kappa_s\phi_{\lambda_s,s}(t).
\]

目标 loss 只用于最后计算 MAE。

## 8. 当前证据

| variant | WSD same-scale | WSD cross-scale | controls |
|---|---:|---:|---:|
| q2 + half-life bracket + support projection | -29.88%, 15/15 | -24.95%, 30/30 | 9/9 non-harm |
| q2 + density projection | -30.22%, 15/15 | -24.93%, 30/30 | worst +8.25% |
| q2 without locality | -30.88%, 15/15 | -24.60%, 30/30 | worst +56.99% |
| no nuisance raw projection | +602.17%, 0/15 | failure | not used |

结论：当前最平衡的解释是

\[
\text{MPL baseline}
\quad+\quad
\text{MPL-LD tangent-cleaned LR-drop finite response}.
\]

其中 \(q_2\) 解释 response time，support projection 解释 locality，MPL-LD tangent projection 解释为什么不能直接从 cosine residual 拟合。

## 9. 局限

- half-life bracket 是 observation-resolution prior，不是完整动力学定理；
- suffix rule 仍是需要冻结的 protocol；
- 当前证据来自固定 loss-curve repository，外部 schedule / 新训练 run 仍缺失；
- 该模型适合说成可解释 residual correction，不能说成普适训练定律。
