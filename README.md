# 多 Agent 文献综述与 DQN 调度实验

本项目在同一套 Agent、语料和预算下，比较固定规则（Rule）与 DQN 调度的综述质量、工作流 Token 消耗、任务错配和动作负载。两种策略共用五动作 Runner，只替换 `select_action`。

**当前进度：共享 Runner 已完成真实 pilot；DQN 已能离线训练、导出并接回同一 Runner。尚未用正式数据完成标定或独立测试，不能据此判断哪种策略更好。**

## 项目状态与交接

| 负责人 | 已有交付 | 下一步与入口 |
|---|---|---|
| 1 号位：执行层与 Rule | 五动作 Runner、Workspace、四维 Verifier、固定规则；两次真实 pilot 合计覆盖五动作 | [Runner 接入和单步示例](ROLE1_HANDOFF.md) · [真实 pilot 验收范围](ROLE1_ACCEPTANCE.md)。正式采样前共同确认 [Verifier 评分口径](SCORING_PROTOCOL.md)。 |
| 2 号位：DQN 与实验 | 状态编码、Masked Double-DQN、轨迹采集与对账、标定仿真、两项单因素消融、冻结推理均已通过离线演练 | [训练与交接手册](ROLE2_HANDOFF.md)。收到正式任务和 Judge 样例后，先复核 **T4 采集 → T5 标定**的数据和标签，再用 1–2 个训练任务估算费用、检查动作覆盖。 |
| 3 号位：数据与评价 | 仓库里的 AgentSkill 论文和任务仅供 pilot；[正式数据目录](data/formal/README.md) 目前只有交付说明 | 待交付正式语料、10 个训练任务、冻结的 Judge 接口及评分样例、统一错配分类器；另行封存至少 5 个独立测试任务。 |

跨岗字段、动作顺序和评测边界以 [CONTRACT.md](CONTRACT.md) 为准。正式测试时先在相同任务上跑完 Rule 与冻结 DQN，封存两者的全部轨迹，再由 3 号位对成功终稿离线盲评。Judge 不参与调度，其 Token 不计入工作流成本。

## DQN 训练与复现

从仓库根目录运行。共享 Runner 和冻结推理只需 Python 3.10+ 标准库；**只有训练和消融需要 PyTorch**。本项目已在 Python 3.12.13、CPU 上验证，不要求 GPU 或 API Key。

**1. 确认共享 Runner 能离线运行。** 此命令使用 Mock 后端和 pilot 样例，运行的是正式比较将复用的 `RuleScheduler`，不是旧四角色流水线：

```powershell
python -m core --backend mock --corpus data/samples/agent_skill/agent_skill_papers.json --task examples/runner_task.json --output outputs/pilot/mock-handoff
```

**2. 在训练环境跑一次最小演练。** 若已有安装过[训练依赖](requirements-rl.txt)的环境，直接激活即可。新机器可用 Python 3.12 的 `python -m venv .venv` 创建环境，激活后执行 `python -m pip install -r requirements-rl.txt`。PowerShell 用 `.\.venv\Scripts\Activate.ps1` 激活，macOS/Linux 用 `source .venv/bin/activate`。

```powershell
python -m experiments.ablate --size smoke --fixture-calibration --config base --seeds 7 --output outputs/dqn-smoke-001
python -m experiments.frozen --checkpoint outputs/dqn-smoke-001/base-seed7/policy.json --output outputs/dqn-pairs-001
```

第一条命令用合成任务采集 Mock 轨迹、拟合仿真环境并训练一个策略；第二条加载该 `policy.json`，让 Rule 与冻结 DQN 在同一 Mock Runner 上执行。**两条命令的输出目录都必须是新目录**，重复运行时更换编号，并同步修改 checkpoint 路径。

| 查看什么 | 输出文件 |
|---|---|
| 使用了什么数据和配置、各组结果如何 | `outputs/dqn-smoke-001/` 下的 `experiment.json`、`comparison.json` |
| 原始训练回报、验证指标和冻结权重 | `outputs/dqn-smoke-001/base-seed7/` 下的 `train.json`、`result.json`、`policy.json` |
| 两个策略的轨迹是否全部保存 | `outputs/dqn-pairs-001/execution_complete.json`，含每条轨迹的路径和哈希 |

要复现**两组消融演练**，从第一条训练命令去掉 `--config base --seeds 7` 并换一个输出目录。Replay Buffer 容量与 Target-Net 更新间隔各三档，共用中档对照，共 5 个配置 × 3 个种子。`--size full` 只增加训练步数，不会把 Mock 轨迹变成正式数据。正式训练包、留出验证、代理质量和错配统计的交接见 [ROLE2_HANDOFF.md](ROLE2_HANDOFF.md)。

## 旧版四角色流水线（保留演示）

以下 `python -m litreview` 命令保留给早期流水线使用；其调用次数、输出格式和 token 说明不适用于上面的五动作 Runner。不要用这条旧路径采集 DQN 训练数据或生成正式 Rule 对照。

### 先运行离线演示

在项目目录的 PowerShell 中运行：

```powershell
python -m litreview --backend demo --corpus examples/demo_papers.json --query "reinforcement learning scheduling" --top-k 1 --output outputs/demo-001
```

输出目录必须是新目录，重复运行时更换编号。会生成：

- `report.md`：带论文编号的综述。
- `result.json`：摘要、证据、正文结构和校验结果。
- `trace.json`：每一步的执行角色、模型、耗时、输出、成功状态和 token 用量；失败时也保存已完成的调用轨迹。

演示数据为虚构示例，不是 arXiv 论文。离线后端只进行简单词项匹配和摘要原文拼接，校验响应也是演示用固定值。
因此离线结果只能用于检查流程，不能用于证明生成质量、错配率或 DQN 效果。离线 token 用量和质量分数均为 `null`。

### 接入 deepseek-flash

默认使用官方地址 `https://api.deepseek.com`，模型为 `deepseek-flash`。
推荐在项目根目录使用 `.env` 文件配置。从模板复制（已有 `.env` 时不要覆盖）：

```powershell
Copy-Item .env.example .env
```

编辑 `.env`，将 `DEEPSEEK_API_KEY=` 后面填入你的真实密钥：

```dotenv
DEEPSEEK_API_KEY=你的真实API密钥
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-flash
```

`.env` 已被 Git 忽略，`.env.example` 只保存不含密钥的模板。
每次启动 DeepSeek 后端时自动读取项目根目录 `.env`，修改后重新运行命令即可，不需要设置系统环境变量或重启终端。
已有的进程环境变量优先于 `.env`，然后才使用默认地址和模型；曾设置同名环境变量时应注意这一优先级。
配置支持单行 `KEY=value`、可选的成对单/双引号、注释和 `export` 前缀；不执行变量替换、反斜杠转义或多行值。
如果使用第三方网关，在 `.env` 中调整地址及模型名；接口须兼容所用请求参数。

```powershell
python -m litreview --backend deepseek --corpus examples/demo_papers.json --query "Compare scheduling and evidence checking in literature review systems" --top-k 2 --output outputs/deepseek-001
```

一次正常流程调用四次 API，会产生服务商计费。摘要阶段如遇输出校验错误，会带着错误反馈修正一次；最多五次调用，全部尝试的用量都会计入轨迹。
网络及 HTTP 错误不自动重试，其他阶段不自动修正。单次最大输出 4096 token，超时 120 秒；仍不合格的输出会中止流程。
token 统计直接采用接口返回值，不使用字符数冒充 token 数；缺失用量保留为 `null`。

接口参考：[首次调用](https://api-docs.deepseek.com/)、[Chat Completions](https://api-docs.deepseek.com/api/create-chat-completion/)、[JSON 输出](https://api-docs.deepseek.com/guides/json_mode/)。

### 换成真实论文

按示例 JSON 格式准备文件，将 `dataset_kind` 改为 `arxiv_abstracts`：

```json
{
  "dataset_kind": "arxiv_abstracts",
  "papers": [
    {"id": "实际arXiv编号", "title": "真实论文标题", "abstract": "真实摘要原文"}
  ]
}
```

程序检查字段、重复编号和引用范围，但不会联网核实论文真实性。真实数据应保留来源与采集记录，切勿将演示数据标记为真实论文。
当前检索会把整个给定集合发送给模型，适合先用少量论文验证流程；大规模语料的索引、分批处理和上下文预算属于后续工作。

现有五篇 AgentSkill 真实论文和五个任务仅供功能测试，不是正式实验数据；三号位尚未交付正式数据。分类与运行方法见 [data/README.md](data/README.md)。可直接运行：

```powershell
python -m litreview --backend deepseek --corpus data/samples/agent_skill/agent_skill_papers.json --task-file data/samples/agent_skill/review_tasks.json --task-id review-001 --output outputs/pilot/agent-skill-001
```

也可通过 `python -m litreview.import_arxiv` 从官方元数据 API 导入新的论文列表，保存可核验的原始 Atom 返回。

固定流程为：检索角色筛选论文 → 总结角色为每篇生成摘要和原文证据 → 撰写角色生成正文 → 校验角色对照原摘要审查摘要和正文。
四个角色共享同一底层模型，通过不同提示词承担不同工作。检索无结果时中止。
本地检查保证论文编号有效、所选论文被覆盖、证据确实出现在摘要中；语义上的事实支持由模型校验，仍需人工抽检。
同模型自查不是独立质量评分，因此 `quality_score` 暂留空。校验未通过时保留问题和中间输出，不生成正式报告。

## 验证

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
& ./tests/verify-agents.ps1
```

请在训练环境运行 Python 测试，以覆盖依赖 PyTorch 的用例；第二条命令是 PowerShell 的项目约定检查。测试覆盖旧流水线、共享 Runner 的动作与计费边界、DQN 的 Mask 与冻结推理、采集对账、按任务留出验证及消融演练。自动测试不调用付费 API，不能替代真实工作流联调或独立终稿评测。
