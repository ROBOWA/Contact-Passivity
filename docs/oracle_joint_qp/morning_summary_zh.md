# 早晨摘要：PARTIAL

已实现已知前臂作用点、完整状态 MuJoCo oracle、off/shadow/active、七变量关节力矩
QP、双账户、非线性候选复核、有限重试和显式失败快照。基线保持不变。

主案例 active 完整执行，净对人输出由 0.241623266 J 降至
0.119998983 J（减少 50.34%），
账户在控制边界的最小裕量 1.86e-07 J。
但法向力最大偏差 53.104 N，离开桌面
1.1498 s；保接触/任务质量不合格。
held-out 紧预算与部分其他 active 案例显式失败，不能宣称整体完成或普遍可行。
子步最小裕量 -1.11e-06 J；端点约束不是连续时间保证。

建议先读：`docs/oracle_joint_qp/report.md`、`acceptance.json`、`reviews.md`，
再看 `active_comparison.png`；完整日志位于 `results/oracle/final/`。

```
.venv/bin/python -m pytest -q
.venv/bin/python -m kuka_table_sim.oracle_experiments case --name resistance --mode active --budget .12 --task-weight 1 --out results/oracle/reproduce
.venv/bin/python -m kuka_table_sim.oracle_experiments matrix --out results/oracle/final
```

失败试验退出码为 2，日志仍保留；不要把这个退出码当成未保存结果。
下一步需要明确决定：是否把制动能量储备/可行性维护与更强的任务约束纳入新阶段。
没有改动摩擦、接触参数、步长、名义增益或 10 N 目标来掩盖问题。
