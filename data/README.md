# 数据分类与交付状态

**三号位正式数据尚未交付。当前没有已验收的正式训练、验证或测试数据集。**

| 位置 | 分类 | 用途 |
|---|---|---|
| `samples/agent_skill/` | 五篇真实 AgentSkill 论文及五个 pilot 任务 | 功能测试、接口联调、流程演示 |
| `../examples/demo_papers.json` | 虚构演示语料 | 离线回归测试 |
| `formal/` | 三号位正式数据交付预留目录，目前只有说明 | 接收后用于训练、标定及独立测试 |

这里的“功能测试样例”不是机器学习的 `split=test` 独立测试集。AgentSkill 样例的语料标记为 `purpose=pilot_only`，任务使用 `split=pilot`；Runner 拒绝将该语料用于 `train/validation/test` 任务，并将用途写入输出日志。不要仅修改标签把样例变成正式数据。

样例来源、原始元数据及运行方法见 [AgentSkill 样例说明](samples/agent_skill/README.md)。原来的 `data/agent_skill_papers.json`、`data/review_tasks.json` 已迁移到该目录，请更新本地命令。

样例新运行产物放在 `outputs/pilot/`。此前基于这五篇论文生成的产物，无论目录名包含 `real` 还是 `delivery`，仍然只是联调结果，不属于正式实验；旧产物保留原样，不改写历史记录。
