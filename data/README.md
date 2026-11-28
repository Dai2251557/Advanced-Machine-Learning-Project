# 实验数据与联调样例

冻结数据包 `arxiv600-review40-v1` 包含以下文件：`arxiv_corpus.json` 含 600 篇论文摘要；`train_tasks.json` 含 30 道训练题；`test_tasks.json` 含 10 道独立公开测试题。训练开发不读取测试题做调参。私有测试证据不在 Git 中，公开完整性核验不覆盖它们。

使用 `utils.data_release.load_corpus()` 投影为 Runner 格式，使用 `load_manifest('train')` 获取训练采样所需的冻结清单。不要用 `litreview.pipeline.load_corpus()` 直接加载原始列表，也不要自行改写冻结数据。题目来源与切分见[数据审查报告](../curation/REPORT.md)。

| 位置 | 分类 | 用途 |
|---|---|---|
| `samples/agent_skill/` | 五篇真实 AgentSkill 论文及五个 pilot 任务 | 功能测试、接口联调、流程演示 |
| `../examples/demo_papers.json` | 虚构演示语料 | 离线回归测试 |
| `arxiv_corpus.json`、`train_tasks.json`、`test_tasks.json`、`release.json` | 正式冻结数据包 | 训练题用于采样和标定；10 道独立测试题只在策略冻结后执行。结果见[独立十题三策略终评](../results/final/独立十题终评.md)。 |
| `formal/` | 原预留目录，目前只有说明 | 保留迁移提示，数据实际位置见上一行 |

这里的“功能测试样例”不是机器学习的 `split=test` 独立测试集。AgentSkill 样例的语料标记为 `purpose=pilot_only`，任务使用 `split=pilot`；Runner 拒绝将该语料用于 `train/validation/test` 任务，并将用途写入输出日志。不要仅修改标签把样例变成正式数据。

样例来源、原始元数据及运行方法见 [AgentSkill 样例说明](samples/agent_skill/README.md)。基于这五篇论文的运行输出放在 `outputs/pilot/`，不作为正式实验结果。
