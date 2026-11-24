# 一号位交接：基础 Agent 与统一 Runner

契约核查结果、已修复问题和剩余验收条件见 [ROLE1_AUDIT.md](ROLE1_AUDIT.md)。共享执行器仅接受 DeepSeekBackend 或显式 MockBackend，避免把离线替身标成真实实验。

**数据状态：三号位正式数据尚未交付。** 五篇 AgentSkill 论文已归入 `data/samples/agent_skill/`，仅供功能测试。样例任务须使用 `split=pilot`；Runner 拒绝将 `purpose=pilot_only` 的语料用于 `train/validation/test`，日志保留 `data_purpose`。正式数据预留位置与旧产物分类见 `data/README.md`。

适用契约：`CONTRACT.md`，`config_version=1`。旧 `python -m litreview` 仍保留；二号应接入下面的 `core.Runner` 模块接口，不使用旧固定流水线采样。

## 二号立即接入

无需安装第三方依赖，Python 3.10+。在仓库根目录运行：

```powershell
python -m core --backend mock --corpus data/samples/agent_skill/agent_skill_papers.json --task examples/runner_task.json --output outputs/pilot/mock-handoff
```

单步接口（Mock 不联网、无密钥）：

```python
from pathlib import Path
from core.runner import Runner
from core.schema import ACTIONS, ReviewTask, RunnerConfig
from litreview.pipeline import load_corpus

corpus = load_corpus(Path("data/samples/agent_skill/agent_skill_papers.json"))
runner = Runner(corpus, config=RunnerConfig(), output_dir="outputs/pilot/sampling")
state = runner.reset(ReviewTask(
    review_id="R001",
    survey_instruction="仅依据摘要比较各论文的技能更新和验证机制。",
    split="pilot",
    difficulty="medium",
), seed=0)
mask = runner.valid_actions()  # 初始 [True, False, False, False, False]
transition = runner.step(ACTIONS[0])
next_state = transition["state_after"]
done = transition["done"]
```

真实执行只替换后端：

```python
from litreview.clients import DeepSeekBackend
runner = Runner(corpus, backend=DeepSeekBackend.from_env(),
                config=RunnerConfig(), output_dir="outputs/pilot/sampling")
```

完整调度使用 `runner.run_episode(policy, task, seed=0)`；`policy` 只需实现：

```python
class MyScheduler:
    def reset(self, seed=None):
        pass

    def select_action(self, state, action_mask):
        # 用二号自己的 StateEncoder 和 Q 网络决策。
        # 返回 ACTIONS 中的字符串，不能直接返回整数索引。
        return ACTIONS[next(i for i, valid in enumerate(action_mask) if valid)]
```

上面的 MyScheduler 只演示接口，不是可用的 DQN 策略。可用固定策略见 `core/rule_scheduler.py` 的 `RuleScheduler`。二号负责 StateEncoder 的拼接顺序与归一化；本交付不擅自定义另一套编码器或奖励。

## 公共数据与行为

- 公共类型集中在 `core/schema.py`。任务和配置为 dataclass；`observe/step/result/run_episode` 返回可直接 JSON 序列化的 dict，调用者修改返回对象不会污染 Runner。
- 动作索引固定为：`retrieve_more / summarize_fast / summarize_deep / write_fast / write_deep`。非法动作抛 `ValueError` 且不消耗步骤；`action_unavailable_reasons()` 返回原因。
- 五动作执行均由 Runner 管理。检索为本地词项匹配，每次补充默认 3 篇未取过的论文，同分按编号排序；语料耗尽后屏蔽检索。大规模语料检索效果尚未评测。
- 有论文即可执行总结或终稿，即使没有总结也允许基于原摘要撰写。质量分低不会屏蔽动作。补充论文会清空旧总结，避免把旧覆盖范围误认为当前总结。
- 成功检索、总结后自动运行同一 Verifier。检索后无总结时仅 coverage 有效；总结后四维有效。材料变更后若校验失败，旧分数版本落后，`verifier_value` 返回 `None`。
- `V` 暂用有效维度等权平均；只有前后有效维度集合完全相同、且各自评分版本有效时才计算 ΔV。首次有效评分、维度变化、失败或写作步骤的 ΔV 均为 `None`。
- `action_history` 始终包含五个动作。`state_before` 是动作前快照；动作次数包含失败尝试，但内部修复不算新动作。低收益阈值来自配置；不可比较增量不会被当成低收益。
- `step.success` 表示动作及必要校验全部成功，低分本身不是执行失败。`done` 表示 Episode 终止；`EpisodeResult.success` 仅在生成合格结构终稿时为真，不保证独立 Judge 质量。
- `write_*` 成功即结束，不再次运行 Verifier；历史 `last_verifier_delta=None`。Runner 永不调用 Judge，训练标签和测试金标均不传入状态。
- `seed` 记录并传给调度器；本地检索确定性排序。它不保证远端模型完全可复现。

## 调用、预算与日志

真实模式限制官方 DeepSeek 地址和 `deepseek-flash`；fast/Verifier 使用 `thinking.enabled + reasoning_effort=low`，deep 使用 `high`。实现依据：[DeepSeek 思考模式文档](https://api-docs.deepseek.com/guides/thinking_mode/)。日志分别记录请求模型与响应返回模型；后者未返回时保留 `null`。

`config.json` 为标准库可直接读取的统一配置，等价于契约建议的 `config.yaml`，不增加 YAML 依赖。默认 8 步、48000 工作流 Token；这些是交付初始值，不是已完成训练标定的最优值。

- 输出 JSON/证据/引用格式失败最多修复一次；网络错误不自动重试，以免无用重发。每次尝试都记录 Prompt、输入、响应、用量、耗时、档位、错误和缓存命中。
- 缓存默认关闭。启用时只在当前 Episode 中复用相同模型/档位/Prompt/输入的成功响应；跨 Episode 清空，避免两策略执行顺序影响成本。缓存命中新增计费为零。
- `input_tokens/output_tokens` 仅含动作调用，`verifier_tokens` 包含校验输入与输出；三者合计为当前步骤 `total_workflow_tokens`。校验成本归触发动作，重试成本全计入。
- Mock 的 0 Token 是明确标记的合成成本；不能用于标定或实验结论。真实返回缺失用量时为 `null`，未终止的 Episode 以 `usage_unknown` 停止，Token 占比也为 `null`，禁止当成零编码。建议二号将此类失败轨迹隔离，不用于成本标定。
- 预算在调用之间检查，不能预知 API 实际输入及思考 Token，因此不是精确的计费硬上限，最后一个调用可能超过预算。预算耗尽时不再修复或调用 Verifier；终稿已成功写出时按成功结束。
- 输出目录下每次创建唯一 `episode-*` 子目录；包括 `episode.json`、`workspace.json`、每步 `before/after` 与完整调用日志，以及成功时的 `report.md`。不指定 `output_dir` 时只返回内存记录，引用字段为 `None`。

## 固定规则与三号交接

固定规则：无材料先检索；有材料无总结先 fast 总结；评分有效且综合分达到 0.75 时 fast 写作；覆盖不足且还有论文时补检索，否则 deep 修订；最多调用 3 次总结；剩最后一步时优先写作。参数保存在配置中。正式测试前需完成训练集标定并冻结，不能看测试结果再调。

三号可直接读取 `EpisodeResult.final_text/source_papers/review_id/split` 离线评测。`examples/runner_task.json` 与当前五篇语料仅为 pilot 联调，不冒充三号尚待交付的独立训练/测试集。

Verifier 四维 Prompt 已位于 `agents/executor.py`，等权综合分位于 `core/schema.py`。三方仍需在正式标定前确认具体评分尺度、阈值和 StateEncoder 编码；本次实现不代表完成了标定。

## 验收命令

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
& ./tests/verify-agents.ps1
git diff --check
```

真实联调命令（会发送所选论文标题、摘要、任务及中间产物到官方 API）：

```powershell
python -m core --backend deepseek --corpus data/samples/agent_skill/agent_skill_papers.json --task examples/runner_task.json --output outputs/pilot/role1-delivery
```

当前交付验证范围：自动测试与 Mock 联调；真实 API 联调曾被本机网络权限/自动审批阻断，不能声明真实五动作已验收。获得具体语料发送授权后，继续运行真实固定策略及两个深度动作的冒烟验证。
