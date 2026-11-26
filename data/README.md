# 数据分类与交付状态

冻结数据包 `arxiv600-review40-v1` 已随主分支交付：`arxiv_corpus.json` 含 600 篇论文摘要；`train_tasks.json` 含 30 道训练题；`test_tasks.json` 含 10 道独立公开测试题。训练开发不读取测试题做调参。私有测试证据不在 Git 中，公开完整性核验不覆盖它们。

使用 `utils.data_release.load_corpus()` 投影为 Runner 格式，使用 `load_manifest('train')` 获取二号采样需要的预先冻结清单。不要用 `litreview.pipeline.load_corpus()` 直接加载原始列表，也不要自行改写冻结数据。完整说明见 [三号交接](../ROLE3_HANDOFF.md)。

| 位置 | 分类 | 用途 |
|---|---|---|
| `samples/agent_skill/` | 五篇真实 AgentSkill 论文及五个 pilot 任务 | 功能测试、接口联调、流程演示 |
| `../examples/demo_papers.json` | 虚构演示语料 | 离线回归测试 |
| `arxiv_corpus.json`、`train_tasks.json`、`test_tasks.json`、`release.json` | 三号冻结数据包 | 已用于 [R005、R001 两条真实训练侧 Rule 探针](../handoff/role1_train_probes/README.md)；不代表已完成批量标定或独立测试 |
| `formal/` | 原预留目录，目前只有说明 | 保留迁移提示，数据实际位置见上一行 |

这里的“功能测试样例”不是机器学习的 `split=test` 独立测试集。AgentSkill 样例的语料标记为 `purpose=pilot_only`，任务使用 `split=pilot`；Runner 拒绝将该语料用于 `train/validation/test` 任务，并将用途写入输出日志。不要仅修改标签把样例变成正式数据。

样例来源、原始元数据及运行方法见 [AgentSkill 样例说明](samples/agent_skill/README.md)。原来的 `data/agent_skill_papers.json`、`data/review_tasks.json` 已迁移到该目录，请更新本地命令。

样例新运行产物放在 `outputs/pilot/`。此前基于这五篇论文生成的产物，无论目录名包含 `real` 还是 `delivery`，仍然只是联调结果，不属于正式实验；旧产物保留原样，不改写历史记录。
