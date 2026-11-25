# CONTRACT.md — DQN 多智能体文献综述系统精简开发契约

> **适用范围**：仅 1、2、3 号位；目标周期一周。**Day 1 冻结核心接口**，其他内部实现充分自主。
>
> **统一版本**：使用单一 `config_version`（初始 `1`）标记本轮公共动作、State Schema、执行器/Verifier 配置、评分规则和实验设置。若跨模块语义发生不兼容改变，更新该版本并标明需要重采样或重训练的结果；不另设六套独立版本或正式审批链。

## 1. 固定架构与动作空间

### 1.1 五个冻结动作（Action IDs）

按以下顺序编码，DQN 输出维度固定为 **5**：

| index | action_id | 含义 | 固定模型约束 |
|---:|---|---|---|
| 0 | `retrieve_more` | 首次/补充检索 | 1 号位自主实现检索 |
| 1 | `summarize_fast` | 轻量总结或修订 | `deepseek-flash` + `reasoning_effort=low` |
| 2 | `summarize_deep` | 深度总结或修订 | `deepseek-flash` + `reasoning_effort=high` |
| 3 | `write_fast` | 轻量终稿撰写 | `deepseek-flash` + `reasoning_effort=low` |
| 4 | `write_deep` | 深度终稿撰写 | `deepseek-flash` + `reasoning_effort=high` |

- **动作语义与索引 Day 1 冻结**。Fast/Deep 是同一模型的不同执行配置，不强制对应五个独立 Agent 实例。
- 四类角色能力必须覆盖：检索、总结、**Verifier 校验**、撰写。Verifier 为统一 Runner 的**自动工作流组件**，不属于五动作之一。
- 所有工作流 LLM 统一走 **DeepSeek 官方 API**，当前 `deepseek-flash` 对应 **DeepSeek-V4.1-Flash-1017**；请求及响应需记录实际模型标识、`reasoning_effort`、时间和 Token。其他 Prompt、检索器、上下文、重试实现由 1 号位自主决定。
- **Judge 不是工作流 Agent**。3 号位自主选择强模型评终稿；训练集真实终稿评分可作 DQN Terminal Reward；**测试时不参与推理和工作流，待 Rule/DQN 测试轨迹全部结束后才离线评分**。Judge 结果不进入 Workspace/State，其 Token 不计入工作流成本。官方文档：<https://api-docs.deepseek.com/quick_start/pricing/>、<https://api-docs.deepseek.com/guides/thinking_mode/>。

### 1.2 统一闭环与 Verifier

- Runner 执行 `retrieve_more`、`summarize_fast`、`summarize_deep` 后，若执行成功，**自动且按相同规则**调用 Verifier；Verifier 输出四维 `[0,1]` 评分：`coverage`、`factuality`、`citation_support`、`consistency`。评分写入 Workspace，含对应材料版本与有效性标记。
- **没有可评价总结时**，Verifier 仍返回四个数值字段，但必须通过 `valid_scores` 指明哪些分量有效；未评项可用 `0.0` 作为占位值，**不得将占位 0 分解释为真实差评**。四项评分的 Prompt、标定方法和综合分权重由 1、3 号位拟定，三方在训练标定前冻结。
- Verifier 提供 `issues/feedback` 等可选文本，但**不得决定 `next_action`**。两种调度策略只通过公共 State 读取其评价。
- `write_*` 成功生成终稿即结束 Episode；**不会再次产生 Verifier ΔV**。终稿交给 Judge 离线评分。达到预算/步数上限或无法继续执行时按统一失败终止规则结束。
- 固定规则与 DQN 共用 Runner、Workspace、Verifier、执行器、重试缓存规则、预算上限和 Action Mask。

## 2. 公共代码与接口（Day 1 冻结）

建议仓库结构，内部文件可自由增补：

```text
project/
├── CONTRACT.md
├── config.yaml                  # 全局 config_version、阈值与预算
├── .env.example                 # 只包含密钥占位符
├── core/
│   ├── schema.py               # 三方唯一公共数据结构
│   ├── runner.py               # 1：Workspace/Step/Episode/重试缓存
│   └── rule_scheduler.py       # 1：固定规则 Baseline
├── agents/                      # 1：执行器/Verifier/Prompt/检索工具
├── rl/                          # 2：StateEncoder、标定 Gym、DQN
├── experiments/                 # 2：真实采样、训练、两组消融、测试脚本
├── data/                        # 3：arXiv 库、10/≥5 综述任务
├── evaluation/                  # 3：终稿强模型 Judge、统计与错配指标
├── tests/                       # 三方：接口测试
└── outputs/                     # 标定、真实轨迹、Judge、曲线、checkpoint
```

**推荐函数签名**（具体使用 dataclass / Pydantic / dict 由 1 号位决定，但输入输出字段语义与类型须一致）：

```python
ACTIONS = (
    "retrieve_more", "summarize_fast", "summarize_deep",
    "write_fast", "write_deep"
)

class Runner:
    def reset(self, review_task, seed=None): ...
    def observe(self): ...                # WorkspaceSnapshot，只含当前可观测信息
    def valid_actions(self): ...          # 长度 5 的物理 Action Mask
    def step(self, action_id): ...        # -> StepTransition，内部处理 API / Verifier / 缓存
    def run_episode(self, policy, review_task, seed=None): ...

class BaseScheduler:
    def select_action(self, state, action_mask): ...  # -> 上述 action_id 之一
    def reset(self, seed=None): ...

class FinalJudge:
    def evaluate(self, review_task, final_text, source_papers): ...
    # 仅供真实训练轨迹的终稿奖励标注、或测试执行完成后的离线评测使用；
    # 禁止在 Runner.step / Scheduler.select_action 中调用。
```

- **1 号位**保证每个动作的实际执行、异常处理、重试与缓存全部在 Runner/执行层完成；2 号位仅通过统一 Runner **组织**采样，禁止另起一套 DeepSeek API 封装。
- `select_action` 是唯一公共策略接口；`RuleScheduler` 和 `DQNScheduler` 只替换此实现，不修改 Runner 业务流程。
- API Key 只通过环境变量/`.env` 传入，不能提交或写入轨迹。

## 3. 公共 Schema（最少必需语义）

本节字段应在 `core/schema.py` 中集中定义，**字段名、数值含义、Action 顺序和可观测时序 Day 1 冻结**；内部可以增加不破坏兼容性的字段。

### 3.1 ReviewTask（3 → 1、2）

```json
{
  "config_version": 1,
  "review_id": "R001",
  "split": "train",
  "survey_instruction": "对给定主题完成研究综述",
  "difficulty": "easy",
  "corpus_scope": "local_arxiv_ai"
}
```

- `difficulty` 为 `easy / medium / hard`，可映射为 DQN 数值特征；难度标签由 3 号位按训练集规则生成。
- 3 号位持有独立的参考/标注论文集合；**不把测试集的 `gold_paper_ids` 或 Judge 分数暴露给 Runner 和 Scheduler**。
- 初始数据 **10 份独立训练综述、至少 5 份独立测试综述**；若预算允许，测试可扩展至 10 份。严格按 `review_id` 隔离。

### 3.2 VerifierResult（1 → Workspace）

```json
{
  "scores": {
    "coverage": 0.78,
    "factuality": 0.86,
    "citation_support": 0.76,
    "consistency": 0.90
  },
  "valid_scores": {
    "coverage": true,
    "factuality": true,
    "citation_support": true,
    "consistency": true
  },
  "checked_material_revision": 3,
  "issues": []
}
```

- 四分数均在 `[0,1]`，越高越好；`valid_scores` 和材料版本用于识别未评估/过期结果。
- 1 号位与 3 号位拟定评分 Prompt/归一化和综合分 `V` 的计算方式；三方在正式标定前固定。不同有效维度的前后评分不得直接相减。

### 3.3 WorkspaceSnapshot（1 → 2；双策略可读）

最少包含：

```json
{
  "config_version": 1,
  "review_id": "R001",
  "difficulty": "easy",
  "retrieved_paper_count": 6,
  "summary_available": true,
  "verifier": {"scores": {}, "valid_scores": {}, "checked_material_revision": 3},
  "step_count": 3,
  "max_steps": 8,
  "workflow_tokens_used": 8000,
  "workflow_tokens_budget": 24000,
  "action_history": {},
  "terminal": false,
  "terminal_reason": null
}
```

`max_steps` 和 Token 预算是**可配置示例**，不要求统一采用上述数值。

`action_history` 必须按**五个动作分别**记录当前 Episode 内的四项动态特征：

1. `call_count`：该动作已调用次数（任务内负载）；
2. `last_verifier_delta`：最近一次该动作执行后、在可比较有效维度上的 Verifier 综合分增量（若不可用则 `null`）；
3. `consecutive_low_gain_or_failures`：连续低收益或失败次数；
4. `token_share`：该动作在当前 Episode 中累计 Token / Episode 总工作流 Token（分母为零时取 `0`）。

**统计口径**：Verifier 自动调用的 Token 归属到触发它的 `retrieve_more` 或 `summarize_*` 动作；`write_*` 的 `last_verifier_delta=null`，不得伪造增量。统计截至**当前动作之前**，不允许提前读取本次执行结果。

**理论口径**：跨综述任务的静态能力差异由 Q 网络权重隐式学习；State 显式编码当前综述 Episode 内的动态历史。冷启动采用统一默认值并保留 `delta_valid`/有效性掩码；不使用未来动作反馈。

### 3.4 StepTransition / EpisodeResult（1 记录，2 管理，3 评测）

每个真实步骤至少记录：

```json
{
  "config_version": 1,
  "review_id": "R001",
  "trajectory_id": "R001_trial02",
  "step_index": 3,
  "state_before_ref": "outputs/traces/.../before.json",
  "action_id": "summarize_deep",
  "action_mask": [true, true, true, true, true],
  "state_after_ref": "outputs/traces/.../after.json",
  "verifier_before": 0.71,
  "verifier_after": 0.77,
  "verifier_delta": 0.06,
  "input_tokens": 1200,
  "output_tokens": 2200,
  "verifier_tokens": 400,
  "total_workflow_tokens": 3800,
  "success": true,
  "done": false,
  "terminal_reason": null
}
```

- **字段示例不是实测数据**。实际日志还应存 Prompt/思考档位、API 错误与重试、输出引用、耗时，以及是否命中缓存。
- `input_tokens + output_tokens + verifier_tokens` 在该示例中组成 `total_workflow_tokens`；真实记录不得重复累加 Verifier Token。
- 一条轨迹的 `EpisodeResult` 至少含 `trajectory_id`、最终终稿引用（可为空）、成功/失败、总 Token、执行步数、动作使用统计；**只有真实完成终稿**才调用独立 Judge。
- `JudgeResult` 独立保存 `overall_score`（归一化 `[0,1]`）、分项、模型/Prompt、评分依据、`split`；不能回写 Workspace。失败轨迹按预定失败奖励处理，不伪造 Judge 分数。

### 3.5 StateEncoder（2 号位实现、三方共用）

状态至少包含：

- **任务/进度**：难度、已检索论文数、当前是否有总结、步骤数/比例、已用与剩余 Token 预算；
- **Verifier**：四项评分、每项可用/过期标记；
- **当前 Episode 的动作历史**：五个动作 × `[call_count, last_verifier_delta, consecutive_low_gain_or_failures, token_share]`，以及增量可用性标记。

1 号位与 2 号位 Day 1 确认拼接顺序、缺失值编码和归一化，并对真实 Runner 与 Gym 仿真使用**同一个编码器**。不人为强制状态必须为固定 32 维，但一轮训练中维度和顺序不可变；配置随 `config_version` 存档。

## 4. Action Mask 与任务错配（两件不同的事）

### 4.1 物理 Action Mask

- Mask 长度固定为 **5**，只屏蔽**技术上不可执行**的动作，例如没有任何论文时不能撰写终稿、必要输入缺失、工具/API 不可用。
- 预算耗尽、步数上限或无法恢复的运行错误触发**统一终止**，而不是因为质量低、能力预测差就屏蔽动作。
- 允许在低质量或高质量状态做出次优但技术可执行的选择，以便真实评估策略差异。
- 标定 Gym 若缺少某状态-动作的采样支持，须**另记 `simulation_support`** 并声明回退/低置信度处理；不能把数据缺失冒充真实世界的物理非法动作。

### 4.2 任务错配自动规则（3 号位实现，三方冻结）

使用统一的 Verifier 综合分 `V`、可比较的 `ΔV`、动作前状态与真实执行结果：

| 类别 | 候选触发条件 | 重要边界 |
|---|---|---|
| 低效/失败 `ineffective` | API/执行失败；或 `V_before < 0.75` 且 `ΔV ≤ 0.02` 的无效返工 | 仅适用于有有效、可比 Verifier 分数的动作；失败单独计数 |
| 冗余/过度配置 `redundant` | 对应目标已达标仍重复无关检索/修订；或 `easy` 状态下 `summarize_deep` 的 `ΔV ≤ 0.02` | 不能只因 `coverage ≥ 0.85` 就把修订事实错误判为冗余；应检查相关缺口与实际增量 |
| 过早终结 `premature_write` | 预算尚充裕、关键 Verifier 指标有效但未达标时选择 `write_*` | 这是质量风险代理标签，须与最终 Judge 分数交叉检查，不是终稿质量的绝对真值 |

- 阈值 `0.75 / 0.02 / 0.85` 为**标定前候选**，由三方在训练标定后、正式测试前冻结。若无法比较两次有效分数，则记 `unknown`，不硬判。
- `write_deep` **不产生 Verifier 的后验 ΔV**，因此不能用 `ΔV ≤ 0.02` 给它打“过度配置”标签。只有同一输入 Workspace 上存在可靠可比的 `write_fast` 对照时，可额外判断深度撰写是否存在显著冗余；否则记为未知。
- 同一步可能命中多类错配。输出 `mismatch_any` 用于**去重后的错配步骤频次**，并单独输出各类计数，避免相加重复。
- 以上标签是**可复现的操作性判定**，不等同于真实反事实最优动作。独立 Judge 最终分数用于检验这些标签与终稿质量的关联。

### 4.3 负载指标

**每个真实 Episode 与每次仿真评估均记录**：五动作调用次数、Token 占比、在可执行机会中的选择比例，以及可选的归一化熵/Gini。检索/总结/撰写职责不同，**不以调用次数趋于五等分作为合理目标**。

主实验和两个消融实验都必须展示**错配频次、类别计数、动作负载分布**；不得只展示训练 Reward 曲线。

## 5. 训练、标定仿真与独立真实测试

### 5.1 三阶段流程

**阶段 A：并行研发与真实标定**

1. Day 1 冻结五动作和 `core/schema.py`，1 号位提供可模拟 Runner，2 号位并行完成 Mock Gym/DQN 基础训练环；**3 号位须在 Day 2 晚前交付 10 份训练任务及所需语料，并冻结 Judge 模型与终稿 Rubric/Prompt、提供可调用评分接口**。
2. 1 号位尽早交付真实 `step()` 及单篇 `run_episode()`；2 号位使用**统一 Runner** 在训练集采集不同动作、难度、Verifier 区间的实际 Token、延迟、分数变化、完成状态和真实轨迹。
3. Day 3 起，先从已交付的 10 份训练任务中选 1–2 份试跑估算费用，再按成本与覆盖情况继续多轨迹采样；完整真实训练终稿由 3 号位使用 **Day 2 已冻结的 Judge** 标注 Terminal Reward，供仿真质量代理校准。2 号位负责组织采样，1 号位不承担重复手工采样。

**阶段 B：标定 Gym 环境与 DQN 消融**

1. 2 号位对真实结果按可观察状态（难度、阶段、评分区间等）和动作分组，构建**明确标注的标定仿真环境**：采样有依据的 Token、Verifier 变化及失败/终止分布。
2. 需记录采样数量、支持覆盖、平滑/回退假设；若某分支数据极少，优先补采样或标记低可信度，不得伪造“真实观察到的增益”。
3. 仿真中生成的终局质量是**代理估计**，如由训练集真实终稿 Judge 结果校准，则保存 `estimated_final_quality`、估计方法及校准误差；**不能命名为真实 `judge_score`**。若样本不足，使用明确披露的 Verifier 代理质量奖励，并在真实测试中检验与 Judge 的一致性。
4. DQN 在 Gym 中训练，保存 Reward/收敛曲线及 Replay Buffer 大小、Target-Net 更新频率两组消融。可选使用已有真实轨迹进行离线微调，但需记录具体方法和数据支持边界。

**阶段 C：真实工作流端到端评测**

1. 固定规则与冻结 DQN 模型在**相同的独立测试综述**上分别真实执行，建议至少 5 份；预算允许可增加到 10 份，新增样本必须事先封存且从未用于训练/调参。
2. **先执行、后评价**：Rule 与冻结权重的 DQN 在运行测试轨迹时**完全不调用 Judge**。所有测试轨迹与终稿保存后，3 号位才使用相同、冻结的终稿 Judge Rubric 进行离线评分，并尽可能隐藏输出的策略来源。Judge 不写入 Workspace/State，评估 Token 不计工作流成本。不得根据测试 Judge 分数再训练 DQN、调整奖励/阈值/Prompt，或选择性重跑结果。
3. 对每份测试综述报告配对 Judge 总分与分项、工作流 Token、任务错配、负载分布、完成率、步骤/返工、延迟。样本少时报告每份结果，不夸大统计显著性。

### 5.2 Reward 与真正优化目标

目标是提高真实终稿 Judge 质量，并控制工作流成本。可采用：

\[
R_T = \alpha Q_{\mathrm{final}} - \beta C_{\mathrm{workflow}} - \gamma F_{\mathrm{failure}}
\]

- **真实完整训练轨迹**：`Q_final` 为 3 号位真实 Judge 评分（Terminal Reward）；Judge 只在该训练轨迹的终稿完成后、**工作流之外**调用。评分作为离线训练标签，不能回流为后续运行时 Workspace 状态。
- **模拟轨迹**：`Q_final` 必须是**校准质量代理值**或明确的 Verifier 代理值，不能称为真实 Judge 打分；应检验其与训练集真实 Judge 的相关性及误差。
- `C_workflow` 包含所有 DeepSeek 动作、Verifier 自动校验、失败重试的输入/输出 Token；**Judge Token 不包含在内**。若逐步扣除 Token 成本，不可在终局重复扣罚同一成本。
- `F_failure` 为失败或无终稿终止的事先定义惩罚。权重及归一化尺度经训练集标定后冻结，禁止用测试集反调。
- 错配指标必须报告，但是否加入 Reward 由 2 号位根据训练集实验决定；若加入，应说明它与 Verifier 评分同源、可能产生指标投机，避免同时用它作为唯一的效果证明。

### 5.3 Baseline 与消融

- **Baseline**：1 号位设计合理、可复现的**固定规则**（固定流水线/默认档位/有限次重试，可读取 Workspace 和四维评分）；在测试前冻结，不利用测试集做阈值优化。它与 DQN 共用 Runner、Verifier、Agent 配置、数据和物理 Action Mask。
- **DQN**：行动选择依据训练过的 Q 网络；真实测试时冻结参数、关闭探索，不得读取 Judge 在线评分、测试标签或未来状态。测试后离线 Judge 只用于独立评价，不得用于调参。
- **两组必做消融**：Replay Buffer 容量与 Target-Net 更新频率，使用相同的标定环境、其他超参及种子；每项至少三个有意义的候选档位，具体数值由 2 号位结合训练步数自主确定。
- **收敛速度**：提前定义可计算的收敛判据（如固定仿真验证场景上滑动平均奖励达到稳定），并同步报告错配、负载、Token 和完成率；平滑曲线要保留原始数值和滑动窗口参数。

## 6. 一周交接、变更与验收

| 节点 | 交接与验收 |
|---|---|
| Day 1 | 三方确认五动作、`core/schema.py`、`config_version`、任务/评分/轨迹字段和 Runner 签名，运行 Schema 单测 |
| Day 2 | 1 号位交付真实单步/单篇最小 Runner；2 号位 Mock DQN 可训练；**3 号位晚前交付 10 份训练任务和论文语料，冻结 Judge 模型、Rubric/Prompt 及评分口径，提供可调用接口** |
| Day 3 | 2 号位使用已交付训练集调用 Runner 形成可信标定日志；3 号位用冻结 Judge 给真实训练终稿评分；三方检查 Token、Verifier 增量与评分标签口径 |
| Day 4–5 | 2 号位完成标定仿真、训练及两项消融；1 号位冻结 Baseline；**3 号位封存至少 5 份独立测试任务**（预算允许可增至 10 份），继续按既定 Judge 评分训练终稿；不得使用测试集校准仿真或选择参数 |
| Day 6 | 真实测试集 Rule / 冻结 DQN 分别运行，执行过程中不调用 Judge；全部轨迹封存后统一离线评分 |
| Day 7 | 三方核验代码、模型、标定来源、原始日志、错配/负载指标和可复现图表 |

**简化变更规则**：只使用一个全局 `config_version`。Day 1 之后若公共字段/动作语义/评分口径必须改变，提出一条变更记录（修改内容、原因、受影响文件、是否需要重采样/重训练），三方确认后更新版本；模块内部实现不需审批。

**最低通过标准**：五动作及 Verifier 真正可运行；真实标定与仿真数据明确区分；DQN/Rule 可以在相同 Runner 内无缝切换；Replay Buffer 与 Target-Net 消融具备收敛和错配/负载统计；至少五份独立测试综述上完成真实端到端对比；所有结果如实报告，**不预设 DQN 全面优于 Baseline**。
