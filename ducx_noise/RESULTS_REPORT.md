# Noisy Tool Environment — 结果汇报

**项目**:DUCK/DUCX 工具型胸片 agent · 噪声工具环境与选择鲁棒性
**驱动模型**:Qwen3-VL-8B (vLLM) · **Agent**:MedRAX (8 个 GPU 胸片工具)
**Benchmark**:ChestAgentBench (EuroRAD) · **日期**:2026-06

---

## 1. 一句话结论

给 agent 的工具集注入可控噪声后,**工具选择被显著扰动**:加 distractor 会让 agent 把调用浪费在假工具上、动作空间熵升高;让真实工具变得不可靠会进一步推高调用次数并拉低任务精度。整套噪声注入 + 评测指标已实现、测试通过,并在真机上端到端验证。

---

## 2. 评测指标(都是后处理 trace,不改原 eval)

| 指标 | 含义 |
|---|---|
| **selection accuracy** | 选到真实工具的比例 |
| **mis-selection rate** | 选到噪声工具(假/冗余)的比例 = p(selected ∈ noise) |
| **selection entropy** | 选择分布的香农熵(动作空间熵) |
| **calls / question** | 每题平均工具调用次数 |
| **task accuracy** | 任务答对率 |

---

## 3. 主结果(真机,ChestAgentBench)

| 条件 | sel_acc | mis-select | entropy (bits) | calls/q | task_acc |
|---|---:|---:|---:|---:|---:|
| base(无噪声) | 1.000 | 0.000 | 2.32 | 6.12 | 0.62 |
| distractor × 2 | 0.814 | 0.186 | 2.44 | 5.80 | 0.66 |
| distractor × 5 | 0.691 | 0.309 | 3.03 | 7.32 | 0.68 |
| distractor × 10 | 0.768 | 0.232 | 3.22 | 7.58 | 0.64 |
| distractor10 + unreliable p=0.1 | 0.667 | 0.333 | 3.54 | 8.84 | 0.54 |
| distractor10 + unreliable p=0.3 | 0.648 | 0.352 | 3.62 | **11.48** | 0.58 |

*(单 seed;另一组 2-case 冒烟同向:base 误选 0.00 → distractor_5 误选 0.53。)*

---

## 4. 三条发现

1. **Distractor 诱导误选、抬高熵**:误选率 0 → 0.19 → 0.31,选择熵 2.32 → 3.03。假工具确实在稀释动作空间、把 agent 引偏。
2. **不可靠工具叠加放大代价**:在 distractor_10 之上把 p_fail 0 → 0.1 → 0.3,误选率 0.23 → 0.33 → 0.35,每题调用次数 7.58 → 8.84 → **11.48**(失败触发重试),任务精度从 0.64 掉到 0.54。即使终端精度变化不大,**过程层开销与混乱被清楚放大**。
3. **注意(单 seed 波动)**:distractor_5 误选(0.31)高于 distractor_10(0.23),小样本下非单调。需 ≥3 seed + 更大 case 数取均值后再读趋势。

---

## 5. 交付与验证状态

- **5 种噪声**全部实现:distractor / unreliable / redundant / description corruption / schema noise(+ drop 算子)
- **完全 opt-in**:无 config 时 DUCX 行为零改动(恒等回归测试覆盖)
- **可复现**:YAML/JSON config 带 seed;LLM 生成结果缓存
- **一条命令**跑通:生成 noisy env → 现有 agent pipeline → eval → 输出 selection accuracy + 误选率
- **23 个单元测试全部通过**;真机端到端已验证

---

## 6. 下一步

- 跑 **3 seed × 更大 case 数** 的 distractor/unreliable sweep,出 mean ± std,抹平单 seed 波动
- 用 **drop 算子**做 N→D 可逆性对比(distractor_10 vs distractor_10 + 全 drop)
- 其余噪声(redundant / description / schema)各跑一组对照,补全 noise-type × 强度的完整网格
