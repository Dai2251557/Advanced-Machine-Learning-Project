# 多 Agent 文献综述与 DQN 调度实验

目标：在相同 Agent 和数据条件下，比较固定规则调度与 DQN 调度的综述质量、token 消耗和任务错配情况。

当前实现是**第一阶段的固定规则对照组**，使用 Python 3.10+ 标准库，无额外依赖。
已提供四个角色、DeepSeek 接口、输出验证、调用轨迹和离线演示。尚未实现 DQN 或完成正式实验。

## 先运行离线演示

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

## 接入 deepseek-flash

默认使用官方地址 `https://api.deepseek.com`，模型为 `deepseek-flash`。
在本地设置 `DEEPSEEK_API_KEY` 后，重新打开终端使环境变量生效。不要把密钥写入代码或提交到 Git。
程序读取环境变量，不自动加载 `.env` 文件。
如果使用第三方网关，需要自行设置 `DEEPSEEK_BASE_URL` 和必要时的 `DEEPSEEK_MODEL`；接口须兼容所用请求参数。

```powershell
# 可选：显式指定模型和地址，不包含密钥。
$env:DEEPSEEK_MODEL = "deepseek-flash"
$env:DEEPSEEK_BASE_URL = "https://api.deepseek.com"
python -m litreview --backend deepseek --corpus examples/demo_papers.json --query "Compare scheduling and evidence checking in literature review systems" --top-k 2 --output outputs/deepseek-001
```

一次正常流程调用四次 API，会产生服务商计费。请求不自动重试，单次最大输出 4096 token，超时 120 秒；截断、空响应、格式错误或校验拒绝会中止流程。
token 统计直接采用接口返回值，不使用字符数冒充 token 数；缺失用量保留为 `null`。

接口参考：[首次调用](https://api-docs.deepseek.com/)、[Chat Completions](https://api-docs.deepseek.com/api/create-chat-completion/)、[JSON 输出](https://api-docs.deepseek.com/guides/json_mode/)。

## 换成真实论文

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

固定流程为：检索角色筛选论文 → 总结角色为每篇生成摘要和原文证据 → 撰写角色生成正文 → 校验角色对照原摘要审查摘要和正文。
四个角色共享同一底层模型，通过不同提示词承担不同工作。检索无结果时中止。
本地检查保证论文编号有效、所选论文被覆盖、证据确实出现在摘要中；语义上的事实支持由模型校验，仍需人工抽检。
同模型自查不是独立质量评分，因此 `quality_score` 暂留空。校验未通过时保留问题和中间输出，不生成正式报告。

## 接下来按这三步推进

1. **真实基线**：选定综述主题，收集一小批真实摘要和任务，跑通 DeepSeek，建立人工质量评分规则与错配定义。
2. **DQN 调度**：为同类任务增加多个能力/成本不同的 Agent 配置，定义状态、候选动作及质量与 token 联合奖励；加入训练/验证/测试划分、经验回放和目标网络。
3. **实验交付**：相同条件下比较固定规则与 DQN，改变回放池大小和目标网络更新频率，多随机种子评估，生成训练曲线、对比表、报告和 PPT。

仅有四种各司其职的固定角色时，路由选择空间不足；同角色的多个候选配置是后续调度实验的重要前提。
当前调用轨迹提供基础观测，不等同于已经构造好的强化学习训练集。

## 验证

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
& ./tests/verify-agents.ps1
```

自动测试不调用付费 API，覆盖四角色流程、数据与证据校验、未知引用、校验拒绝、截断和异常响应、token 统计、接口请求格式及 CLI 成功/失败输出。
真实 API 联调需要本地密钥和网络，不能用自动测试通过代替。
