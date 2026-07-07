# 任务提纲 × 跑出的结果

Qwen3-VL-8B-Instruct，vLLM，fp16，greedy，seed 0，Killarney L40。每条任务下写实测结果。

---

## 配置

**改成 greedy decoding，固定 seed 保证可复现**
> 已实现（`--decoding greedy` → temperature=0/top_p=1；`--seed` 透传 vLLM+torch）。
> **结果**：同 seed+greedy 重复跑同 3 例，`raw_output` **逐字节相同** → 可复现验收通过。

**VLM 从 bf16 改成 fp16 加载**
> 已实现（vLLM serve `--dtype float16`，每条记录记 `model_dtype` 溯源）。
> **结果**：fp16 加载 16.65 GiB、**无 NaN / 无精度退化**；模型正常作答。

---

## 定义（先敲定，阻塞后面）

**三个概念边界：Distractor（输入层）/ Failure（结果层）/ Unreliable（行为层）**
> 已实现为集中式 taxonomy（`ducx_noise/taxonomy.py`，枚举+判定函数，阈值占位）。
> **结果**：Distractor（输入轴）与 Failure/Unreliable（结果/行为轴）可并存（单测验证）。
> 实测里 **outcome 实际只出现 SUCCESS/FAILURE**——因为 (a) 模型**从不误选** distractor
> （见下，0/500 及 225/225），(b) `malformed_tool_calls`/`tool_errors` 这两个 Unreliable
> 信号 runner 尚未从 trace 填（默认 0）。所以 **Unreliable 这一档在本轮数据里未被触发**。

**两条路径样本怎么分：tool useless vs tool useful + distractor**
> `tool_useless` = ChestAgentBench MCQ（临床文本可答，工具无用）；
> `tool_useful` = 探针题「哪个病理概率最高」（答案=真实分类器 argmax，看图猜≈1/6，工具必需）。
> **结果（工具的价值符号随路径翻转，都显著）**：
> - tool_useless：no_tool **0.52** → 加工具 **0.30**（**−0.19，p<1e-3**，n=191 配对；
>   最早 n=50 时是 −0.04，放大后成为稳健的负效应）。
> - tool_useful：no_tool **0.22-0.27** → 工具 **0.65-0.78**（**+0.55，p<1e-3**）。

---

## Distractor 机制

**Tool sequence 打乱 order，避免 distractor 固定放太后**
> 已实现 `tool_order: fixed|shuffle|controlled`；shuffle 整体打乱、位置与角色解耦。
> **结果**：6-seed 测试确认 shuffle 下 distractor **落点分散、不再锁末尾**（单测通过）。

**检查 description 与真实 tool 是否 align（明显假 / 以假乱真两档）**
> 已实现 `similarity: obvious|aligned`（aligned = LLM 改写真实工具描述，缓存）。
> **结果（n=500）**：**"以假乱真更毒"假设被推翻**——aligned 不系统性比 obvious 差
> （c2/c3 aligned 反而更高：0.488>0.414、0.668>0.452；c5 才低）。两档纠缠、无稳定差异。

**Distractor 的 append 位置规则（head / tail / 随机 / 指定位置）**
> 已实现 `position: head|tail|random|index`（配合 controlled 做位置扫描）。
> **结果（位置消融，n=500，count=5 aligned）——唯一干净的效应**：
> | head | index=0 | random | index=3 | tail |
> |---|---|---|---|---|
> | 0.472 | 0.478 | 0.512 | 0.532 | 0.532 |
> **有序首因效应**：越靠前越伤（head→tail 单调升，head-vs-tail ~1.9σ），且都低于 clean 0.654。

---

## 实验

**按两条路径各跑：tool useless / tool useful + distractor**
> 完成。n=50 首批（450 条）+ n=500 扫描（count 12 条件×500 + position 5 条件×500）。
> **结果（数量剂量曲线 n=500，position=random，clean=0.654）**：
> | count | obvious | aligned |
> |---|---|---|
> | 1 | 0.756 | 0.796 |
> | 2 | 0.414 | 0.488 |
> | 5 | 0.522 | 0.466 |
> | 10 | 0.444 | 0.540 |
> | 20 | 0.752 | 0.760 |
> **非单调 U 形**：count=1 和 20 ≈ clean，中间 2-10 才塌。"干扰越多越差"即使 n=500 也不成立。

**每条路径都跑：加 tool vs 不加 tool 两种情况**
> 完成（no_tool / tool_only / … 条件齐全）。
> **结果（gating 混合集，n=379 配对）**：never 0.359 < always 0.536 < **gated(oracle) 0.633**；
> **gated 显著胜 always +0.10（p<1e-3）、胜 never +0.27（p<1e-3）**。
> 机制：文本可答题**有工具时模型 99% 忍不住去调**，一调 acc 0.50→0.30（主动被工具带偏）。

**推理时就把 response 的 likelihood 存下来，不能事后补**
> 已实现（`--capture-logprobs`，agent 走 trace、no-tool 走 direct call，推理时落盘）。
> **结果（n=50）**：真实记录 `resp_logprob_mean` 296 token、量级正常；
> **有工具反而更不自信**（with-tool −0.084 vs no-tool −0.054）。

---

## 分析

**Tool contribution：单个工具/distractor 的边际贡献**
> **结果**：工具本身边际贡献符号随任务翻转（工具必需 +0.55 / 文本可答 −0.19）。
> 单个 distractor 的边际：1 个多余**真实**工具 **无害**（多工具 all_real 0.787 ≈ oracle 0.782，
> 配对 p=1.0）；假 distractor 要**堆到 ~20 个**才显著（见下）。

**Tool combination：工具组合的效果**
> **结果**：
> - 数量×similarity 网格：非单调 U 形（见上），无单调剂量律、无 aligned>obvious。
> - 多工具组合（探针 CLS，配对）：oracle(1工具) 0.782；+1 无关真实工具 0.787（**无差异**）；
>   **+20 干扰 0.640（−0.14，p=4e-5 显著）**——且模型 **225/225 仍正确调 classifier、0 次调干扰**，
>   伤害纯粹来自 context 占据。剪枝回原工具 = oracle → **回收 +0.14（p=4e-5）**。
> - 调用集合：模型调对工具时 acc 高，被迫多调/调错工具的少数样本 acc 低。
>
> **⚠️ 误选是任务相关的（重要修正）**：上面"0 次调干扰"**只对探针任务成立**（calls/q=1.00，
> 模型精准打一次）。在**开放式 ChestAgentBench**上（上周实验，全套真工具+5假，500题）模型
> 探索式狂调（calls/q 5.67→7.49），**误选率 misselect=0.376（38% 调用落到假工具）**、
> sel_acc 1.00→0.62。但那里 **task_acc 几乎不变（0.648→0.658）**——文本可答,调错也能救回。
> 所以污染有**两条伤害通道**：被动 context 占据（探针,伤 task_acc）+ 主动误选（开放任务,
> 高误选但不伤 task_acc）。calls/q 是中介变量,由任务是否"工具必需且答案唯一"决定。

**Response likelihood 对比：加 tool vs 不加 tool，再按 correct/incorrect 拆开**
> **结果（n=50）**：
> | 组 | mean logprob |
> |---|---|
> | with_tool | −0.084 |
> | no_tool | −0.054 |
> | with_tool · correct | −0.070 |
> | with_tool · incorrect | −0.101 |
> | no_tool · correct | −0.018 |
> | no_tool · incorrect | −0.081 |
> 两点：①有工具整体更不自信；②**置信度追踪对错**（correct 的 logprob 恒高于 incorrect）。

---

## 一页结论

- **工具价值符号随"任务是否非工具不可"翻转**：文本可答 −0.19、工具必需 +0.55（都 p<1e-3）。
- **最大杠杆是 gating（要不要给工具），不是选哪个工具**：gated 显著胜 always/never。
- **误选是任务相关的**：**探针任务**（calls/q=1）模型精准打一次、**0 误选**（路由 top1=1.0）；
  **开放式 ChestAgentBench**（calls/q 5-7）**误选 38%**。伤害两条通道：被动 context 占据
  （探针,伤 task_acc）、主动误选（开放任务,不太伤 task_acc）。
- **distractor 数量非单调**，1-2 个无害，~20 个才显著；**位置有首因效应**；**aligned 不更毒**。

图/表：`analysis/multitool/{gating,pollution_recovery,dissociation,bounds}.png`、
`analysis/toolbias_n500/sweep_n500.png`；完整叙事见 `analysis/TOOLSPACE_SUMMARY.md`。
