# 解释性修正后的模型决定

> 2026-06-19 回滚决定：零参数 cooldown finite-response 版本解释性更干净，但性能太弱，不能支撑当前工作目标。当前主线恢复为 `observation-bracket MPL-LD + MPL-LD tangent projection + sample-size ridge`。它不是 gate/channel/正弦/DCT 拼接模型，真正从 residual 中拟合的仍然只有一个非负标量 \(\hat\kappa_s\)。零参数 finite-response 保留为机制下界和负控，说明“只改 MPL 最后一项”方向有信号但不足以作为主模型。

> 2026-06-19 theory refinement：当前推荐主公式进一步改为 `q2 half-life MPL-LD response`。核心架构不变，仍然是 \(L_{\mathrm{MPL}}+a_s\hat\kappa_s\phi_{\lambda_s}\)；变化只在 schedule-only 规则：用 Herfindahl concentration \(q_s=\sum_t(d_t/\sum_u d_u)^2\) 替代旧的 max-drop \(q_\infty\)，并用 half-life bracket \(\lambda_s=\lambda_{\mathrm{obs}}/(2-q_s)\)。\(a_s\) 解释为 support-projection locality，而不是 learned gate。该版本 same-scale WSD `-29.88%`、15/15 wins，cross-scale WSD `-24.95%`、30/30 wins，controls 9/9 non-harm。

这份文档覆盖此前把 DCT、sqrt-locality、gate、channel、正弦展开等候选混在一起的写法。当前恢复的主线是 observation-bracket MPL-LD：它用 causal LR-drop response 描述 WSD cooldown 后的非瞬时响应，用 MPL-LD tangent projection 去掉 cosine residual 中由 MPL 参数误差造成的污染，并用 sample-size ridge 表达有限样本可辨识性。DCT、gate、channel、正弦展开仍不作为主方法。

2026-06-19 进一步修正：observation-bracket 诊断版本不再使用固定 \(\tau=0.05\)、不再使用 \(\lambda_{\mathrm{obs}}/2.5\) slow endpoint，也不再把 fast endpoint round 到 20。theory refinement 进一步把 response-rate rule 写成 effective-drop-count half-life bracket：

\[
q_s=\sum_t\left(\frac{d_t}{\sum_u d_u}\right)^2,
\qquad
\lambda_s=\frac{\lambda_{\mathrm{obs}}}{2-q_s},
\qquad
\hat\kappa_s=
\frac{\langle x_s,y\rangle_+}{\|x_s\|_2^2+1/N_{\mathrm{cal}}}.
\]

这些量都由观测间隔、LR schedule 和 cosine calibration suffix 的样本数决定，不使用 WSD loss 选择。

## 1. 为什么之前的模型解释性不够

前面的模型有三个问题。

第一，候选项太多。DCT nuisance、sqrt-locality、channel/gate、正弦项都能改善某些图，但它们不像一个统一机制，而像为了贴合 residual shape 做的经验修补。

第二，直接从 cosine residual 拟合 \(\kappa\) 是错误的。最新审计显示，如果不做 nuisance removal，只把 cosine residual 投影到 LR-drop response 上，WSD-family 结果平均恶化约 \(+672.31\%\)，worst 达到 \(+2585.94\%\)，0/15 wins。这说明 cosine residual 里有强的全局 MPL drift；它会被错误吸收到 \(\kappa\) 中。

第三，DCT 虽然数值强，但解释性弱。DCT 说的是“去掉低频”，而不是“去掉 MPL 哪一类误差”。它可以保留为 performance reference，但不应该作为核心理论项。

## 2. Observation-Bracket 诊断公式

对任意目标 schedule \(s\)，预测写成

\[
\hat L_s(t)
=
L_{\mathrm{MPL},s}(t)
+
a_s\hat\kappa_s\phi_{\lambda_s,s}(t).
\]

其中 \(L_{\mathrm{MPL},s}(t)\) 是已有 MPL 预测；\(\phi_{\lambda_s,s}(t)\) 是只由 LR schedule 计算的 causal LR-drop response；\(\hat\kappa_s\) 是唯一从 loss residual 中估计的幅度；\(a_s\) 是 schedule-only support-projection locality，用于防止把局部 cooldown response 转移到 full-horizon cosine decay。\(a_s\) 不是学习参数，也不是按 family 选择的 learned gate。

Positive LR drop 定义为

\[
d_t=[\eta_{t-1}-\eta_t]_+.
\]

响应特征为

\[
\phi_{\lambda,s}(t)
=
\sum_{u\le t}
\exp[-\lambda\eta_u]
\frac{d_u}{\eta_{\max}}.
\]

这个式子有三个约束：

- causal：只使用 \(u\le t\) 的过去 LR history；
- drop-only：constant LR 区间不产生 correction；
- single response：不引入 curve family label、channel 选择或手工 gate。

## 3. Response Rate 如何得到

先由观测间隔给出一个可解释时间尺度。当前 loss logging 的主要间隔是 \(\Delta_{\mathrm{obs}}=128\)，因此

\[
\lambda_{\mathrm{obs}}
=
\frac{\log 2}{\eta_{\max}\Delta_{\mathrm{obs}}}
\approx 18.05.
\]

为了区分 smooth cosine-like decay 与 sharp WSD drop，引入只由 schedule 计算的 effective drop concentration。令 \(D_s=\sum_t d_t\)，\(p_t=d_t/D_s\)，定义

\[
q_s=\sum_t p_t^2=\frac{1}{n_{\mathrm{eff}}}.
\]

单步 drop 时 \(q_s=1\)，均匀分布在 \(n\) 个 step 上时 \(q_s=1/n\)。当前版本不再使用手写 slow endpoint，而是直接插值 half-life：

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

解释：当 LR drop 很 diffuse 时，\(q_s\approx0\)，响应半衰期取两次 observation interval；当 LR drop 是单步 sharp drop 时，\(q_s=1\)，响应半衰期取一次 observation interval。这个 rule 只表达 observation-resolution bracket，不再引入 2.5 或 20 这样的手写端点，也不再依赖最大单步 drop。

## 4. 为什么用 MPL-LD Tangent Nuisance

我们把 cosine residual 写成

\[
r_{\cos}(t)
=
L_{\cos}(t)-L_{\mathrm{MPL},\cos}(t).
\]

如果 MPL 参数有小误差，那么 residual 中会包含

\[
J_{\mathrm{MPL}}(t)\Delta\theta
\]

这一类局部线性误差。当前只投影掉 MPL 中 learning-rate dependent 部分的切空间：

\[
J_{\mathrm{LD}}(t)
=
\left[
\frac{\partial L_{\mathrm{MPL}}}{\partial\log B},
\frac{\partial L_{\mathrm{MPL}}}{\partial\log C},
\frac{\partial L_{\mathrm{MPL}}}{\partial\log \beta},
\frac{\partial L_{\mathrm{MPL}}}{\partial\log \gamma}
\right].
\]

理由是：我们要修正的是 schedule/LR 相关误差，因此只把 MPL 的 LR-dependent 参数误差视为 nuisance；不投影 \(L_0,A,\alpha\) 的 backbone tangent，也不使用泛化的低频 DCT 基作为核心理论。

在 cosine calibration suffix \(t\ge8000\) 上，令 \(P_{\mathrm{LD}}\) 为 \(J_{\mathrm{LD}}\) 的正交投影矩阵。定义

\[
x_s=(I-P_{\mathrm{LD}})\phi_{\lambda_s,\cos},
\qquad
y=(I-P_{\mathrm{LD}})r_{\cos}.
\]

然后估计

\[
\hat\kappa_s
=
\frac{\langle x_s,y\rangle_+}{\|x_s\|_2^2+1/N_{\mathrm{cal}}}.
\]

这里 \(N_{\mathrm{cal}}\) 是 cosine calibration suffix 中的样本数。它替代固定 ridge \(\tau=0.05\)，作用是有限样本 identifiability floor。真正从 residual 学到的仍然只有一个非负标量 \(\hat\kappa_s\)。\(J_{\mathrm{LD}}\) 由已有 MPL 公式和 MPL 参数数值差分得到，不对 WSD loss 拟合；\(\lambda_s\) 和 \(a_s\) 只由 LR schedule 得到。

Calibration suffix 也不再只说成固定 8000。当前 source-only rule 是：在候选 suffix starts 中，在 source cosine 上检查 observation bracket 的两个端点 \(\lambda_{\mathrm{obs}}/2\) 与 \(\lambda_{\mathrm{obs}}\)，选择最早使

\[
\rho=
\frac{\|(I-P_{\mathrm{LD}})\phi\|_2^2}{\|\phi\|_2^2}
\le
\frac{1}{N_{\mathrm{cal}}}
\]

对所有 source scale 和这两个端点成立的起点。这个规则选出 `8000`。它的含义是：从这个位置以后，LR-drop response feature 在 MPL-LD tangent projection 后的可辨认能量已经低于有限样本地板，早期更强的 alignment 更可能是 MPL drift contamination。该规则不枚举当前 WSD target schedule，因此不是对目标集合的间接适配。额外 dense-grid 审计显示用 2、3、5、9、17、33、65、129 个 \(\lambda\) 点都会选出 `8000`，因此 endpoint-only 规则不是分辨率偶然。

## 5. 训练与测试流程

训练 / 校准阶段：

1. 对每个 scale 读取 `cosine_72000.csv`。
2. 用已有 MPL 参数计算 \(L_{\mathrm{MPL},\cos}\) 和 residual \(r_{\cos}\)。
3. 对目标 schedule 由 LR schedule 计算 \(q_s=\sum_t(d_t/\sum_u d_u)^2\) 和 \(\lambda_s=\lambda_{\mathrm{obs}}/(2-q_s)\)。
4. 在 cosine schedule 上构造 \(\phi_{\lambda_s,\cos}\)。
5. 计算 MPL-LD tangent basis \(J_{\mathrm{LD}}\)，得到 \(x_s,y\)。
6. 用上面的一维公式估计 \(\hat\kappa_s\)。

测试 / 转移阶段：

1. 不使用目标 curve 的 loss 拟合任何参数。
2. 由目标 LR schedule 构造 \(\phi_{\lambda_s,s}\) 和 \(a_s\)。
3. 输出

\[
\hat L_s(t)=L_{\mathrm{MPL},s}(t)+a_s\hat\kappa_s\phi_{\lambda_s,s}(t).
\]

## 6. 关键结果

同 scale cosine-to-WSD：

| variant | WSD mean | WSD worst | wins | controls |
|---|---:|---:|---:|---:|
| q2 half-life MPL-LD + support projection | -29.88% | -4.67% | 15/15 | 9/9 non-harm |
| no nuisance raw projection | +602.17% | +2366.35% | 0/15 | 9/9 non-harm with locality |
| observation-bracket MPL-LD, no locality | -30.89% | -4.67% | 15/15 | controls worst +56.99% |
| observation-bracket MPL-LD + linear locality | -29.87% | -4.67% | 15/15 | 9/9 non-harm |
| old MPL-LD fixed tau / old lambda | -27.25% | -3.00% | 15/15 | 9/9 non-harm |
| DCT performance reference | -32.83% | -5.30% | 15/15 | 9/9 non-harm |

跨 scale 稳定性：

| method | same-scale WSD | cross-scale WSD | cross-scale worst | reading |
|---|---:|---:|---:|---|
| q2 half-life MPL-LD | -29.88%, 15/15 | -24.95%, 30/30 | -3.15% | 当前推荐公式 |
| observation-bracket MPL-LD | -29.87%, 15/15 | -24.95%, 30/30 | -3.15% | 强诊断参考，不是最终主模型 |
| old MPL-LD fixed tau | -27.25%, 15/15 | -23.07%, 30/30 | -2.07% | 被新版本替代 |
| DCT performance | -32.83%, 15/15 | -18.98%, 26/30 | +26.68% | 均值强，但有跨 scale 坏例 |
| tau-free DCT | -20.77%, 15/15 | -13.27%, 27/30 | +9.04% | 较保守，但仍依赖 DCT |

这些结果支持当前工作结论：MPL-LD tangent nuisance control 后，LR-drop response 可以显著改善 WSD-family；把 response time 写成 \(q_2\) half-life bracket 后，公式解释性更强且不损失效果。DCT 仍只作为数值上限和消融参考。

## 7. 当前局限

- q2 half-life bracket 中“两次 observation interval”仍是结构性 prior。它比 \(q_\infty\)、2.5/20 更干净，但不能夸大成物理定律。
- suffix rule 目前在 `8000` 处取得最佳强收益；`10000` 之后仍 15/15 正优化但收益变小，说明 calibration window 仍是需要谨慎讨论的 protocol。
- \(a_s\) 是 support-projection boundary。它解决 full-horizon cosine controls，但仍是边界条件，不是 optimizer dynamics 的完整定理。
- 当前数据仍是固定 loss-curve repository；外部新 schedule 或新 training run 才能真正验证泛化。

## 8. 写作建议

如果暂时写作，核心贡献应保守地这样讲：

1. MPL 对 LR-drop 后的 transient response 有系统性遗漏。
2. 直接用 cosine residual 估计 response 会失败，因为 residual 混入了 MPL 的全局漂移。
3. 用 MPL-LD tangent projection 去掉 LR-dependent MPL 参数误差方向后，只需一个非负 response amplitude 就能稳定改善 WSD-family。
4. response rate 和 ridge 都可以由 observation resolution / finite-sample identifiability 给出，不需要固定 \(\tau=0.05\)、2.5 或 20。
5. 这个机制在跨 scale calibration 中仍然全胜，因此比 DCT 低频滤波更适合作为诊断参考；但最终主线应优先尝试 \(B[D_\tau-D]\) finite-response。

不要把 DCT、sqrt-locality、gate、channel、正弦展开写成主方法。它们最多是负证据、性能参考或探索记录。
