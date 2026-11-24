# 第一批真实论文与综述任务

这批材料来自用户提供的五个 arXiv PDF 链接，面向 Agent 技能优化、联邦技能演化及语义知识共享。
仅用于建立真实流程的初始基线，五篇论文和五个任务尚不足以支持 DQN 的泛化效果结论。

## 文件

- `agent_skill_papers.json`：可直接读取的论文集合，包含原始标题、摘要、作者、版本、发布日期、来源和内容哈希。
- `agent_skill_papers.atom.xml`：官方元数据 API 的原始返回，供离线核对和重建。
- `review_tasks.json`：人工编写的五个任务，包含检索覆盖预期和人工检查要点。它们是检查指南，不是模型质量评分或完整标准答案。

摘要来自 arXiv 官方 Atom API，只合并排版换行及空白，不翻译或改写。版本以此次导入结果为准，链接固定到对应版本。
完整原始返回与标准化摘要分别保留 SHA-256。`retrieved_at` 采用 UTC。

| arXiv 编号 | 论文 | 本批综述中的关注点 |
|---|---|---|
| 2605.23904 | SkillOpt: Executive Strategy for Self-Evolving Agent Skills | 文本空间技能优化与验证门控 |
| 2606.03143 | FederatedSkill: Federated Learning for Agentic Skill Evolution | 个性化技能演化与语义技能差异 |
| 2604.16778 | Federation over Text: Insight Sharing for Multi-Agent Reasoning | 推理轨迹汇聚与共享洞察库 |
| 2610.21348 | FedSkillGit: Federated Skill Evolution via Version Control | 规则补丁、隐私筛选与客户端验证 |
| 2607.21361 | FedAgentKE: Federated Semantic Knowledge Evolution for Heterogeneous Agents | 异构框架之间的语义知识演化 |

主题来源见各论文的 `source_url`。这些论文用作系统的输入材料；不能仅因主题相关就声称它们实现了本项目要求的 DQN 调度。
数据中的摘要、标题和作者署名归原作者，来源及版本链接随记录保留。

## 运行一个任务

在项目根目录执行（输出目录须未存在）：

```powershell
python -m litreview --backend deepseek --corpus data/agent_skill_papers.json --task-file data/review_tasks.json --task-id review-001 --output outputs/real-review-001
```

可将任务编号替换为 `review-002` 至 `review-005`，并相应更换输出目录。
`top_k` 默认使用任务内的设置，仍可通过命令行显式覆盖。只有任务的 `query` 会交给 Agent，人工检查要点与预期论文编号不作为提示词提供。
默认流程正常完成需要四次付费 API 调用；摘要校验失败时最多修正一次，并保留失败尝试和全部用量。先检查一个任务的结果，再决定是否执行剩余任务。

## 人工检查

参照每个任务的 `evaluation_points`：核对引用、原文证据、指标单位、适用范围和摘要未提供的信息。
`expected_paper_ids` 是本批任务的人工覆盖预期，程序验证编号有效，但不会把检索匹配率直接视为事实准确率。
任务涉及相互重叠的论文与内容，本批不标记为相互独立的训练集、验证集或测试集。
后续扩大语料和任务时，再按论文组或主题控制训练与评测之间的信息泄漏。

## 重新采集

程序仅支持现代 arXiv 编号和官方 abs/PDF 链接。每次最多 50 篇，合并为一次 API 请求，无自动循环抓取或重试。
选择新的输出文件保留当前快照，例如：

```powershell
python -m litreview.import_arxiv 2605.23904v2 2606.03143v1 2604.16778v3 2610.21348v1 2607.21361v1 --output outputs/reimport/papers.json
```

导入会验证返回编号集合、版本、标题、摘要和作者，并保存同名 `.atom.xml`。失败时不会用示例摘要填补缺失数据。
接口说明：[arXiv API User's Manual](https://info.arxiv.org/help/api/user-manual.html)。
