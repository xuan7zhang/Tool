# VLM Tool-Use / Distractor 实验报告

Qwen3-VL-8B 在 tool-use 场景下、被 distractor 工具干扰时的行为变化。
从确定性解码配置锁定，到 taxonomy / distractor 机制 / Stage-5 schema / runner / 分析，
再到 Killarney L40 上的真实实验（n=50 → n=500）。所有代码在 `ducx_noise/`、`analysis/`，
模型为 Qwen3-VL-8B-Instruct（vLLM，fp16，greedy，seed 0）。

---

## 0. 决策点（已确认）

1. **两条路径的样本**：`tool_useless` = ChestAgentBench MCQ（临床文本即可作答，工具无用）；
   `tool_useful` = `probe_task.py` 造的「哪个病理概率最高」题——答案由真实 DenseNet 分类器的
   argmax 决定，看图猜 ≈ 1/6，**工具必需**。
2. **correctness**：`extract.py` 的 temp-0 LLM 抽取器（Qwen2.5-7B，非自评）抽最终字母后与标准答案
   exact-match。判定标准始终是抽取字母 == answer，不是语义 judge。
3. **aligned distractor 构造**：LLM 改写真实工具描述成以假乱真的近似版（缓存；离线模板兜底）。
4. **taxonomy**：三个 label 按草案实现成占位常量 + 判定函数，阈值留空待定。

---

## 1. 配置锁定（Stage 1，可复现前置）

- **greedy decoding**：`--decoding greedy`（默认）强制 temperature=0、top_p=1，关闭采样；`--decoding sample`
  保留原采样。仅作用于主生成，答案抽取始终 temp 0。
- **fp16**：vLLM serve 加 `--dtype float16`（`--model-dtype fp16|bf16` 记进每条记录做溯源）。
- **seed**：`--seed`（默认 0）透传到 vLLM per-request seed，并 seed Python/NumPy/torch。
- **验收（真实硬件）**：同 seed + greedy，重复跑同 3 例 `raw_output` **逐字节相同**；fp16 加载 16.65 GiB、
  无 NaN。
- 顺带修了一个潜在 bug：seed/logprobs 原会污染 `openai_kwargs`，导致 aligned 生成用的 raw
  `openai.OpenAI` client 构造失败；拆成独立 `chat_extra` 只给 ChatOpenAI。

## 2. Label Taxonomy（Stage 2，`ducx_noise/taxonomy.py`）

- `Outcome{SUCCESS, FAILURE, UNRELIABLE}`（互斥）+ `ToolRole{REAL, DISTRACTOR_OBVIOUS, DISTRACTOR_ALIGNED}`。
- **Distractor 是独立的输入轴**（tool list 里存在不该调的工具），与 outcome 并存（一条记录可同时
  distractor-present 且 FAILURE）。
- 所有判定阈值/优先级抽成文件顶部 **PLACEHOLDER 常量**（`DISTRACTOR_MIN_COUNT`、`UNRELIABLE_ON_*`、
  `OUTCOME_PRECEDENCE=FAILURE>UNRELIABLE`），docstring 带正反例——等最终定义。
- 单测覆盖互斥性 + 并存性。

## 3. Distractor 机制（Stage 3，`config.py`/`transforms.py`/`generation.py`）

- `DistractorConfig`：`similarity(obvious|aligned)`、`position(head|tail|random|index)`、`index`。
- `NoiseConfig.tool_order(fixed|shuffle|controlled)`：**shuffle 整体打乱、distractor 不再锁在末尾**
  （位置与角色解耦）；controlled 精确按 index 放（位置消融用）。
- aligned 走 `generate_aligned_distractors`（LLM 改写真实描述，缓存）。
- `python -m ducx_noise.preview --config X` 可离线打印实际 tool 顺序核对。

## 4-5. Runner + 日志 Schema（Stage 4-5）

- `--path`、`--condition` 记进每条记录；condition ∈ `{no_tool, tool_only, tool_useless, tool_useful,
  tool_useful_distractor}`。
- **Stage-5 record**（`ducx_noise/schema.py`）：sample_id / query / image_ref / path / condition /
  tool_set / tool_roles / n_distractors / distractor_positions / distractor_similarity / raw_output /
  called_tools / final_answer / is_correct / **outcome_label** / **resp_logprob_sum/mean** /
  per_token_logprob / seed / model_dtype / decoding / timestamp / run_id。
- likelihood 在推理时就存（agent 走 trace logprobs，no-tool 走 direct call logprobs），非事后补。
- **验收（真实记录）**：`validate_record` 零缺失字段；called_tools 从 trace 正确解析；
  per_token_logprob 296 token，resp_logprob_mean ≈ -0.18，量级正常。

## 6. 分析（Stage 6，`analysis/toolbias_analysis.py`）

读日志出 6.1 tool contribution / 6.2 combination / 6.3 likelihood，CSV + 图，不改推理代码。

---

## 实验设置（真实运行）

- **平台**：Killarney，kn064/kn065（L40s×4/节点），vLLM serve + langchain agent client。
- **拓扑**：每条流 1 卡 vLLM + 1 卡 agent tools；两条流并行铺满 4 卡。
- **tool set**：ImageVisualizer + DicomProcessor + **ChestXRayClassifier**（探针任务的必需工具）。
- **规模**：先 n=50 摸底（450 条），后 n=500 压噪声（clean/no_tool 各 500 + count 12 条件 ×500 +
  position 5 条件 ×500）。

---

## 结果（n=500，每点 ±0.04 CI）

### 基线
| condition | acc |
|---|---|
| no_tool（探针，看图猜）| 0.268（chance≈0.167）|
| tool_useful（clean，真实工具）| **0.654** |

工具在「工具必需」任务上 **+0.386**。探针设计成立。

### 数量剂量曲线（position=random）
| count | obvious | aligned |
|---|---|---|
| 1 | 0.756 | 0.796 |
| 2 | 0.414 | 0.488 |
| 3 | 0.452 | 0.668 |
| 5 | 0.522 | 0.466 |
| 10 | 0.444 | 0.540 |
| 20 | 0.752 | 0.760 |

**非单调、U 形**：count=1 和 count=20 都 ≈ clean（0.75-0.80），中间 2-10 塌到 0.41-0.67。
即使 n=500 CI 收紧，"干扰越多越差"依然不成立。

### aligned vs obvious
**无"以假乱真更毒"**：aligned 在 c2/c3 反而更高，c5 才低，两档纠缠、无系统差异。假设推翻。

### 位置消融（count=5, aligned）—— 唯一干净的效应
| head | index=0 | random | index=3 | tail |
|---|---|---|---|---|
| 0.472 | 0.478 | 0.512 | 0.532 | 0.532 |

**有序首因效应**：干扰项越靠前越伤（head 0.472 单调升到 tail 0.532，head-vs-tail ~1.9σ），
且所有位置都低于 clean 0.654。

图：`analysis/toolbias_n500/sweep_n500.png`。

---

## 结论

1. **工具在必需任务上真有用**：no_tool 0.268 → clean 0.654（+0.39），紧。
2. **干扰项确实能掉分**（2-10 个时 0.65→0.41-0.52），但**非单调 U 形，不是剂量律**。
3. **位置 > 数量**：位置有干净的首因效应（前面比后面毒）；数量和 similarity 都提不出稳健规律。
4. **模型全程 0/500 从不调用 distractor 工具**——伤害纯粹来自 context 占据 tool-list，不是选错工具。
   这是本实验最稳、最有意思的行为学结论。
5. likelihood（n=50 批）：有工具时回答反而更不自信（with-tool -0.084 vs no-tool -0.054），
   且置信度追踪对错（correct 比 incorrect logprob 更高）。

## 混淆与局限（必须点明）

- **count 与 distractor 内容混淆**：每个 count 用不同的生成干扰集（同 seed 确定，但内容随 count 变），
  所以"数量"效应和"具体哪些干扰项"分不开。U 形很可能是这个混淆 + 固定 500 题的特异交互，
  而非真的数量律。干净测数量需**固定干扰池、逐个增量加** + 跨多 seed。
- **单 seed（seed 0）**：位置首因效应 ~1.9σ 是单 seed 结果，多 seed 才能定死显著性。
- `malformed_tool_calls`/`tool_errors`（UNRELIABLE 信号）schema 已留但 runner 未从 trace 填（默认 0），
  故本轮 outcome 只有 SUCCESS/FAILURE。
- taxonomy 阈值仍是占位常量。

## 运维教训

- 两条重型流（vLLM+agent+MemorySaver+capture-logprobs=20）在 n=500 撑爆 job 的 **160G cgroup** →
  双双 OOM（exit 137）。修复：大 N 准确率扫描 **关掉 --capture-logprobs**（host RAM 降到 ~59G）。
- OOM 杀掉的 vLLM 会留孤儿进程占显存 → 下一个 vLLM "Engine core init failed"。
- kn064 中途被别的 ray/verl 训练占了（24GB/卡）；跑前先 `nvidia-smi --query-compute-apps` 确认，
  别假设 interactive job 的卡是空的。已改用空闲的 kn065。

## 复现

```bash
# 配置锁定的单条命令（serve fp16 + greedy + seed，跑 ChestAgentBench）
MODEL_DTYPE=fp16 DECODING=greedy SEED=0 bash run_chexbench_repro.sh

# 两条路径 × 条件
bash ducx_noise/run_toolbias_experiment.sh          # tool_useless + tool_useful_distractor

# n=500 剂量/位置扫描（logprobs 关闭以避免 OOM）
MODE=count    CAPTURE_LOGPROBS=0 MAX_CASES=500 bash ducx_noise/run_toolbias_sweep.sh
MODE=position CAPTURE_LOGPROBS=0 MAX_CASES=500 SKIP_BASE=1 bash ducx_noise/run_toolbias_sweep.sh

# 分析
python -m analysis.toolbias_analysis --logs-glob 'logs/[ts][bw]_*/*.json' --out-dir analysis/toolbias_full
```

## 产出

- 代码：`ducx_noise/{taxonomy,schema,preview}.py`、`config.py`、`transforms.py`、`generation.py`、
  `run_toolbias_experiment.sh`、`run_toolbias_sweep.sh`；`analysis/toolbias_analysis.py`。
- 表：`analysis/toolbias_full/{contribution,combination,likelihood}.csv`。
- 图：`analysis/toolbias_n500/sweep_n500.png`。
- 真实日志：`logs/tb_*`（n=50，带 logprob）、`logs/sw_*_n500`（n=500 准确率）。
