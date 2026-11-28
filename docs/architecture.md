# 技术架构与接口：从一次调度到一份终稿评分

在这个系统中，**策略选择下一步，Runner 执行并记账，Judge 在整批执行结束后评终稿**。替换调度器不需要改检索器、模型调用或评分代码。以下从三个接口说明如何接入策略、读取轨迹和关联终稿评分。实验结果与复现命令见[项目首页](../README_zh.md)。

## 1. 如何接入一个调度策略？

策略实现 [`BaseScheduler`](../core/schema.py) 的两个方法：`reset(seed=None)` 清除上一题的策略状态；`select_action(state, action_mask)` 接收当前快照与按 `ACTIONS` 排列的布尔列表，返回一个**合法动作的字符串 ID**，不是下标。Mask 只表达可执行性：无论文时不能写，有论文时即使覆盖度很低也允许写。是否值得继续检索，由策略决定。

以下示例从仓库根目录运行，只读 pilot 样例，使用默认 Mock 后端，不需要 API Key 或 PyTorch。这个策略先检索、有论文就写，用于演示接口，不是正式实验中的 Rule 或 DQN。

```pycon
>>> from pathlib import Path
>>> import json
>>> from core.runner import Runner
>>> from core.schema import ACTIONS
>>> class RetrieveThenWrite:
...     def reset(self, seed=None):
...         pass
...     def select_action(self, state, action_mask):
...         action = 'write_fast' if state['retrieved_paper_count'] else 'retrieve_more'
...         assert action_mask[ACTIONS.index(action)]
...         return action
...
>>> corpus = json.loads(Path('data/samples/agent_skill/agent_skill_papers.json').read_text(encoding='utf-8'))
>>> task = json.loads(Path('examples/runner_task.json').read_text(encoding='utf-8'))
>>> episode = Runner(corpus).run_episode(RetrieveThenWrite(), task, seed=0)
>>> [step['action_id'] for step in episode['transitions']]
['retrieve_more', 'write_fast']
>>> episode['execution_kind'], episode['terminal_reason']
('mock', 'final_written')

```

[`Runner.run_episode`](../core/runner.py) 每题重置 Runner 和策略，随后循环调用 `select_action(runner.observe(), runner.valid_actions())`，并把返回值交给 `runner.step(action_id)`，直到终止。[规则策略](../core/rule_scheduler.py)直接使用快照；[冻结 DQN](../rl/scheduler.py) 用[同一状态编码器](../rl/state_encoder.py)生成 45 维输入，在合法动作中选 Q 值最大的一个。示例可用 `python -m doctest docs/architecture.md` 离线验证。

## 2. 如何读一条轨迹？

| 数据结构 | 何时取得 | 应该看什么 |
|---|---|---|
| `WorkspaceSnapshot` | `runner.observe()`：动作选择前 | 读取当前材料版本和已检索篇数；由 `step_count` 与 `max_steps` 计算剩余步数。`verifier` 是对材料或总结的过程检查，不是终稿标签。 |
| `StepTransition` | `runner.step(action_id)`：一次动作执行后 | `state_before`、当时的 `action_mask`、`action_id` 和 `state_after` 还原一次决策。`success` 是这一步是否成功；`done` 表示整题是否终止。`calls` 记录该步的实际调用，可能包含自动校验与修复。 |
| `EpisodeResult` | `run_episode()` 返回；也可在终止后调用 `runner.result()` | `transitions` 是有序步骤。用 `terminal_reason` 和整局 `success` 判断是否交稿；`final_text` 是成功终稿，`total_workflow_tokens` 是整条轨迹的成本。 |

**评分和计费有两处容易误读：**

- **无效分不等于零分。** 检索或总结执行成功后会自动调用 Verifier。只有 `checked_material_revision` 对应当前材料、且 `valid_scores` 标记为真的分项才有效。未总结时只有覆盖度可评。前后有效维度不同，`verifier_delta` 为 `None`；写作没有后置 Verifier，因此也没有写后增量。判定逻辑见 [`comparable_delta`](../core/schema.py)。
- **预算不是服务端扣费上限。** Runner 在调用前检查余额，返回后按真实 usage 结算，最后一次调用可能使总用量超过 48,000 token。达到或超出预算时以 `token_budget` 终止；用量未知时以 `usage_unknown` 终止，不当作零。两者都优先于该步的候选终稿：即使写出了文本，也不算成功交付。

## 3. 如何把终稿评分对应到正确的轨迹？

1. **先封存，不打分。** [终评执行入口](../experiments/final/run_final.py)为同一题的 Rule、v1、v2 各建独立轨迹。整批 10 题 × 3 策略达到终态后才写封存索引，包含轨迹文件和终稿的哈希。失败轨迹也属于整批，不能省略。
2. **从封存批次准备盲评。** [三策略盲评入口](../experiments/final/judge_final.py)核对批次、权重与成功稿集合；[`prepare_sealed_test_reviews`](../evaluation/judge.py) 核验公开任务、冻结语料及封存轨迹，只为成功稿返回评审材料。`build_judge_request(prepared)` 只构造请求，不发送。Judge 能看到题面、终稿、引文证据和书目，看不到策略名或本地轨迹绑定。先收齐再评，确保评分不会影响后续调度，也不会漏掉失败轨迹。
3. **按稿绑定，不按题号覆盖。** `finalize_judge_result(prepared, receipt)` 校验回执中的本地绑定、冻结请求、终稿原文哈希及模型标识。分析时用 `trajectory_id` 找轨迹、用 `final_text_sha256` 核对具体稿件；不能只按 `review_id` 合并，否则会混淆同题的三份稿。失败或无有效评分留空，不补零；Judge token 与工作流成本分账。

[公开逐轨结果表](../results/final/report/per_task.csv)可用于核对路径、分数和成本。它不是原始封存批次，不能当作盲评入口的输入。
