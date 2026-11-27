# V2 DQN 联调接口：尚未批准用于正式训练

入口 `rl.v2_live_env.V2LiveEnv`。二号位可用它联调新的动作与状态宽度；不能把 V1 五动作权重直接加载到本接口。

```python
from rl.v2_live_env import V2LiveEnv, ACTION_ORDER, FEATURE_NAMES

env = V2LiveEnv(corpus, public_train_tasks, sections, backend, evaluator)
state, info = env.reset(seed=0)
action_mask = info['action_mask']
state, reward, terminated, truncated, info = env.step(action_index)
```

动作固定为 `retrieve_more, rewrite_query, read_1, read_2, read_3, write`。三个读取动作绑定当前可见候选列表中的具体章节；每次决策前必须重新读取候选和掩码。候选按题面/当前查询与章节元信息的重合度排序，不读取尚未打开的正文内容。可见长度来自段落长度元数据。

数值状态宽度取 `len(FEATURE_NAMES)`，输出宽度取 `len(ACTION_ORDER)`。状态包含剩余成本与步数、资料数量、题目公开措辞、查询变化、最近阅读是否支持问题的模型反馈，以及三个候选的可用性、匹配度、长度。阅读反馈是执行模型的诊断，可能出错，不是答案标签。题目中的 checklist、required_paper_ids 和 source_section_ids 在进入执行环境前被去除。

每步返回真实 API Token 成本。暂定奖励为 `-0.1 * step_tokens / token_budget`；成功终稿加外部 evaluator 返回的 `[0,1]` 质量；预算或步数耗尽减 1。这只是可替换的联调奖励，不是校准后的正式奖励。`evaluator` 必须由调用方显式传入，生产实验应使用独立验证过的评分器。模型传输或结构错误返回 `eligible_for_training=False`，二号位必须丢弃整条失败 rollout，不能把接口故障当作动作质量。

## 每项能力应当产生的区别

| 能力 | 状态变化 | 应当有用的情形 | 应当付出的代价 |
|---|---|---|---|
| 继续检索 | 新论文摘要进入材料 | 还缺相关论文 | 步数，后续模型输入成本 |
| 改写查询 | 下一次检索排序改变 | 现有检索方向偏离缺口 | 模型调用成本，且可能改坏 |
| 读候选 1/2/3 | 只有被选段落的正文进入材料 | 摘要没有实验数值、条件或限制 | 实际读取调用及后续输入成本 |
| 写作停止 | 生成终稿并结束 | 证据足够或追加证据不值得 | 写作成本及漏答风险 |

## 当前交付状态

已有 8 篇训练关联论文、24 个 PDF 定位片段、12 道训练侧题目；源 PDF 位于本地 `outputs/v2-benchmark/sources/`。这是初步选材，仍需逐段视觉复核与证据充分性审查。`experiments/v2_benchmark.py` 提供直接写、固定多读、按需规则三条路线。

第一批运行因为 PDF 排版字符与引文格式校验失败而中止，其结果不能作为可训性结论。已经修复 PDF 空白/连字匹配和模型已写明 ID 的方括号格式；完整重复对照、通用漏答评分校准、实际成本下的预算选择仍待完成。4500 Token 与读取前预留 1800 Token 是暂定值，不是由实验标定得出的预算。

在跨题重复对照出现稳定的质量/成本差异之前，只能称本接口可联调，不能称 DQN 已有有效训练空间。独立测试题和 V1 冻结轨迹不可混入这批训练试验。
