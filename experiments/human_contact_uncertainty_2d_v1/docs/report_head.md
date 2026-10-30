# 报告 / Report — human_contact_uncertainty_2d_v1

平面 PoC：阻抗-意图型人类不确定性、风险敏感选择与 Ec + BH 账户。
Planar PoC: impedance-informed human uncertainty, risk-sensitive selection, and Ec + BH accounts.

**仅为研究仿真；不构成硬件执行或部署的授权或证据。**
*Research simulation only; this is neither authorisation for nor evidence about hardware execution or deployment.*

Folder: `/Users/carl/code/contactPassivity/experiments/human_contact_uncertainty_2d_v1/`. Base commit:
`5e792cf7157449a9806c9c3aeb6e3b5464780b3a`. Mathematics: `formulation.md`. Sources and adaptations:
`literature_mapping.md`.

## 中文执行摘要

**做了什么。** 在一个隔离文件夹中实现了一个平面刚体工具，`q = (x, y, θ)`，沿柔性墙擦拭：
* 偏心工具尖端；
* 法向力目标 10 N；
* 切向进给 8 cm/s；
* 姿态调节。

模拟人类在机器人上一个已知接触片（顶边）的**未知坐标 s** 处施力。人力由隐藏的阻抗模型产生，
包括潜在意图轨迹、速度相关残差和 OU 生理噪声。实现了以下部件：
* 完整的 Ec + 净 BH 账户与 `BH ≥ S` 被动后备储备；
* P1 式结构化"阻抗-潜在意图"高斯过程，在固定增益与已条件化轨迹下是精确的联合高斯条件化，
  联合推断意图、位置、残差权重、力传感器偏置与 OU 状态；
* Gauss–Markov 与朴素 GP 基线；
* 闭环信念传播：Monte Carlo（每次 rollout 抽取一个一致的函数样本）与一阶传播（含反馈雅可比与跨时协方差）；
* 风险中性/熵风险（β>0 为风险规避，β = −θ/2 与 P3/P4 对应）的有限候选策略规划；
* 按"单拍预测功"施加约束的 1 kHz 伺服能量滤波；
* 独立的真值审计器。

所有闭环结论来自冻结代码与独立种子（≥100）：
* 主比较 4000 回合；
* 不可行初始化 150 回合；
* 消融、位置扫描与步长细化 3040 回合；
* 分叉真值校准 200 回合（每个预测器 4244 次分叉）；
* 事后（post-hoc）鲁棒界 800 回合（新种子）。

**主要结论（数字见正文）。**

1. **能量层本身有效（tier C 证据，tier A 构造）。**
   * 无保护的 C0 在所有人机交互场景中 100% 违反真账户，每回合机器人对人的净功 1.1–2.1 J（BH0 = 0.15 J）。
   * 带滤波的 C1–C4 净功 0.003–0.12 J；在全部 7050 个带滤波的独立回合中，Ec 从未违规。
2. **采样数据实现很关键。**
   * 朴素的"瞬时功率"条件在 1 ms 零阶保持内泄漏能量，最坏真 BH −80 mJ。
   * 改为约束"单拍预测功"后，以**真值账户**驱动的参考滤波（ORACLE）在 50 回合中最坏只有 −21 µJ 的数值亏欠，约为 BH0 的 1e-4。这是数值验证，不是采样间证明。
3. **最受证据支持的升级是"与位置相关的历史功误差信念"。**
   * 分布内 5 个场景合计，无裕度的 C1 有 7/500 回合违反 `h_H = BH − S`（最坏 −7 mJ）。带协方差裕度的 C2/C3/C4 为 0/500（单侧 95% 上界 0.60%）。
   * 在"相同名义力、不同力臂 × 位置不确定性"扫描中：C3（力 + 位置不确定性）0/270，仅考虑力不确定性 12/270，C1 20/270。
   * 记账误差随位置不确定性从 1.7 增至 10.4 mJ。
4. **结构化 GP：离线预测明显更好，闭环却没有改善。**
   * 离线，分布内：0.3 s 力 RMSE 0.83–1.0 N，基线为 1.5–1.9 N；功预测已校准（z 标准差 1.0–1.2）。
   * 对候选运动的反事实响应斜率为 0.84–0.95，朴素 GP 约 0.05，Gauss–Markov 为 0。
   * 闭环中，与运动无关的 Gauss–Markov 预测器（C2）在所有交互场景都更好：
     * 对人正功少 0.01–0.03 J；
     * 法向力 RMSE 更低；
     * 后备裕度更大；
     * 伺服滤波介入少约 2.5 倍。
   * 原因：C3 预测更"尖锐"，规划更贴近 `h_H` 边界，而目标函数并不奖励保留 BH。
   * 分叉真值校准表明，闭环 0.3 s 裕度预测对 C3 过度自信（分布内 `h_H` z 标准差 2.25，C2 为 1.39），离线校准不能直接迁移到闭环。
5. **失败。**
   * 未见过的振荡意图（OOD）下，结构化 GP 严重过度自信（离线 NLPD 8.7）。闭环完成率降为 93–95/100，法向力 RMSE 约 2.4 N。
   * 力/位置传感器偏置超出先验（0.8 N、6 cm）时，所有滤波控制器都失败：`h_H` 违规 75–98/100，C3 真 BH 违规 26/100。记账 BH 乐观约 80 mJ，即偏置 × 人引起的接触点位移，力数据本身无法辨识。
   * 事后（post-hoc）在新种子上加入区间鲁棒界（假设 |b_f| ≤ 1 N、位置误差 ≤ 8 cm，**覆盖**了真实偏置）后，偏置场景的 `h_H` 违规从 33/50（C2）、48/50（C3）降为 0/50。代价是分布内任务性能下降，例如块释放覆盖率 0.98→0.94、法向力 RMSE 约翻倍。界本身错误时会怎样，未被检验（§6.7）。
6. **风险敏感目标作用很小且方向与 P3/P4 一致。** β 增大时选择的刚度略升（1–2%）、速度略降，即"更刚而非更柔顺"。β 较大时指数权重的有效样本数降到 1–2/48，估计不可靠。
7. **`BH ≥ S` 很保守。** 低 BH（0.02 J）时账户保持，但几乎无法完成任务（1–2/100）。不可行初始化（`h_H(0) < 0`）时，滤波在 0.1 s 内卸载接触回到集合内，此后任务不可行。

**建议。**
1. 采用"单拍预测功"的采样数据滤波。
2. 采用"与位置/偏置相关、跨时保留"的历史功误差信念，并配以**经过校准的**偏置区间界。
3. **不建议**以当前形式把结构化 GP 规划或熵风险目标作为主线升级。结构化 GP 可作为预测与监测模块；在用于闭环之前需要：
   * 在线校准，即新息监测或方差膨胀；
   * 包含预测模型误差的协方差；
   * 一个真正奖励保留人类预算的目标。

**理论缺口**见 §9。所有统计结果属于 tier C（经验），高斯收紧属于 tier B（近似），都**不是**确定性安全证明。

## Executive summary (English)

| question | answer (held-out seeds; tier) |
|---|---|
| Does the Ec + BH tick-work filter bound robot→human energy? | Yes in simulation (C). C0: 100% account violations, 1.1-2.1 J net work. Filtered: ≤ 0.12 J, no Ec violation in 7050 filtered episodes. The ORACLE reference shows µJ-level sampled-data deficits; the instantaneous filter leaks up to 80 mJ. |
| Is location uncertainty non-degenerate and does modelling it matter? | Yes (C). The past-work error grows 1.7 → 10.4 mJ with sensor uncertainty. h_H violations: C1 20/270, force-only 12/270, C3 0/270. |
| Does the structured (P1-style) GP predict better? | Offline, in distribution, yes: 2× lower force RMSE, calibrated work, correct response to motion. Out of distribution or with gain mismatch it is over-confident. |
| Does it improve closed-loop outcomes? | **No.** C2 (Gauss–Markov) is better on interaction, force tracking and margins. C3 is over-confident in forked-truth calibration and regresses on OOD intent. |
| Does risk-sensitivity (β>0) produce compliance? | **No.** It produces slightly *stiffer*/slower selections (1-2%), consistent with P3/P4; practically negligible. |
| Biggest failure | Persistent sensor bias outside the prior: the ledger is optimistic by ≈ 80 mJ, with 75-98% h_H violations. A post-hoc interval bound that covers the true bias removes them on fresh seeds (33/50 and 48/50 → 0/50), at a task cost (§6.7). |
| Recommendation | Adopt the tick-work filter and the location/bias-correlated past-work error belief with a **calibrated bound**. Do not adopt the structured-GP planner or the entropic objective without online calibration and an objective that values the human budget. |
