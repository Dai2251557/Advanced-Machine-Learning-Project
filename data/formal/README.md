# 正式数据位置说明

**状态更新**：三号已在 data-fix 分支交付，现已集成本地。为保持冻结路径与哈希不变，没有迁移到本目录；实际文件在 `data/arxiv_corpus.json`、`data/train_tasks.json`、`data/test_tasks.json` 和 `data/release.json`。使用 `utils.data_release` 加载。以下为原先的交付计划。

此目录目前只有交付说明，没有正式语料或任务。禁止用 AgentSkill 样例填充占位。

三号位交付后，应保留来源与版本记录，并检查：

- 语料、ReviewTask 字段符合 `CONTRACT.md`，训练与测试按 `review_id` 隔离。
- 正式训练任务与独立测试任务明确分开；测试标签及 Judge 结果独立存放，不交给 Runner 或调度器。
- 核验实际任务数量与契约要求，不把现有五个 pilot 任务计入正式交付。

具体文件名和目录细分待实际数据到达后确定；当前不虚构正式数据或完成状态。
