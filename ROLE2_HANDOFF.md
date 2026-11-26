# 2 号位交接：DQN 训练与冻结评测

**当前进度：采集入口已有真实训练轨迹可验；正式标定还没有足够的动作覆盖和终稿标签。** 训练、消融、导出和冻结推理已能离线演练。仓库没有可供独立比较的正式 DQN checkpoint，也没有 Rule/DQN 独立测试结论。

| 已交付 | 对训练意味着什么 |
|---|---|
| [冻结数据与 Judge 协议](ROLE3_HANDOFF.md)：600 篇摘要、30 道训练题（清单推荐首批 10 题）、10 道独立测试题的公开题面 | 使用 `utils.data_release.load_corpus()` 和 `load_manifest('train')`；测试题及私有证据不参与标定、调参或选权重。 |
| [R001、R005 的完整 collection](handoff/role1_train_probes/README.md)：两条都成功写稿，工作流分别消耗 15,290 和 11,215 token | 两条都走 `retrieve_more → summarize_fast → write_fast`。`summarize_deep` 和 `write_deep` 各有 4 次可选机会、0 次实际选择。可验证格式、用量和快速路径，不能估计快慢动作差异。 |
| [离线 Judge 接口](evaluation/judge.py) | 协议已冻结不等于探针已打分。两条终稿均未随附绑定轨迹的独立 Judge 标签；Verifier、Mock 分数和旧版 R001 样例都不能代替。 |

## 从现有探针到正式训练包

1. **核验已有输入。** 在仓库根目录运行 `python -m utils.data_release`，再运行 `python -m unittest discover -s tests -p test_role_integration.py`。后者用冻结训练清单验证两份 collection，并核对文件哈希、成功终止、动作序列和真实用量；两项检查都不发起付费调用。采集 → 标定的数据契约见 [experiments/collect.py](experiments/collect.py)：任务和论文用规范化 JSON 哈希，终稿用原始 UTF-8 文本哈希。
2. **补齐动作覆盖，而不是重复同一条 Rule 路径。** 先与执行层确认 [Verifier 口径](SCORING_PROTOCOL.md)、实际后端、配置和调用预算。经授权后仅在训练题上用配置好的共享 Runner 调用 `collect_episode(runner, task, policy, mode='formal', manifest=manifest)`，保存返回值（含 `next_mask`、来源与采样计划），不要用 Runner 的中间文件替代。用 `support_report` 分开看每个动作的可选机会、实际选择次数和独立题数；失败与用量未知的轨迹也要保留。不同前缀上的快慢选择不是严格反事实。
3. **为成功的训练终稿绑定独立标签。** 对已通过 `validate_episode` 的轨迹，评价侧用 `prepare_train_review` 和 `build_judge_request` 构造盲评请求。仓库不会自动调用 Judge。实际评分须单独授权，评分方留存同次调用的 `{binding, request, response}`，用 `finalize_judge_result` 核对回执，再用 `save_judge_result` 为每条轨迹单独保存标签。将这些标签原样放入训练包的 `labels`，不要手拼分数或只按 `review_id` 关联。`validate_collection` 检查轨迹、终稿哈希和评分归档格式，并不认证服务端调用。缺评不是零分；Judge 不进入 Workspace、Mask 或工作流成本。
4. **按题留出，再拟合。** 正式训练包为 `{episodes, labels, manifest, holdout_review_ids}`。在 30 道训练题内预先登记留出 ID，同题所有轨迹必须在同一侧。`fit_calibration` 只用拟合集；保持检索规格一致，不混用不同检索版本的样本。`evaluate_holdout` 对照 Rule 的实际与模拟完成率、成本、可比增量及分母，并报告终稿代理 MAE 和训练均值基线 MAE。仅用这两条探针时，留出任一题后都只剩一道拟合题；程序可以运行，但两种留出方式都没有可比较的独立标签（`compared=0, mae=null`），不是零误差。

**如何解读仿真。** `model.manifest()` 给出各桶的样本数和独立题数。仿真按剩余预算逐次回放调用：已发起的失败调用计费，未获准的修复既不计费也不采用其输出；只回放观测到的 `cache_hit`。完全无样本的合法分支按用量未知的截尾终止处理，这是建模假设，不是测得的失败率。

有新鲜 Verifier 分数时，终稿代理可用同阶段、同写作档位、至少两道训练题的残差校正，不足时回退至有效 Verifier 均值。无可用分数时，优先取同阶段、同档位至少两道已标注无分终稿的题均值，否则使用拟合集标签的跨题均值（`pooled_mean_low_support`）。两者都不是该终稿的 Judge 真分。若存在可成功写稿的无分状态却完全没有训练标签，模型会在 `unscorable_writes` 列出受影响的写作动作，默认质量环境拒绝启动；可补标，或显式设 `RewardConfig(quality_weight=0)` **仅诊断成本**。回退率高、留出误差大或动作覆盖窄时，应先补样，不据此选择正式策略。

## 消融、冻结和独立测试

- **训练与选权重。** 使用 `python -m experiments.ablate --size full --training-bundle <训练包> --threshold <预先登记值> --output <新目录>`。默认 5 个配置 × 3 个种子（`7,17,27`），每次 20,000 环境步、CPU 单线程。Replay Buffer 容量 `512/2048/8192`，Target-Net 间隔 `50/200/800` 梯度步，共用 `2048/200` 对照。每 500 步在固定仿真场景贪心验证；连续三次达到门槛才记收敛步数，否则为 `null`。只用训练侧信息选 checkpoint。
- **成对执行。** 正式 checkpoint 绑定训练时的检索规格。冻结推理只需 `DQNScheduler.load(path)`，无探索或梯度更新。获得真实执行授权后，把相同 Runner、对应公开测试题和正式 checkpoint 交给 `experiments.frozen.execute_pairs(..., mode='formal', test_manifest=...)`。使用预先冻结的测试清单（本次交付 `load_manifest('test')` 对应 10 题），题目集合必须与清单完全匹配；不要由候选输入临时生成清单。入口在执行前检查任务、语料和配置；交错执行顺序，失败也保存。
- **封存与盲评。** 只有整批 Rule/DQN 执行结束后才生成带文件哈希的 `execution_complete.json`。之后由评价侧对成功终稿离线盲评，不向评分者提供策略名。Judge token 单列。`mismatch_status=awaiting_frozen_role3_classifier` 表示错配分类仍待统一；不能用低收益次数冒充 `mismatch_any`。

## 仅复现离线演练

在已安装[训练依赖](requirements-rl.txt)的环境中，从仓库根目录执行；输出必须使用新路径。采样和冻结推理只需标准库，训练和消融才需 PyTorch。

```bash
python -m experiments.collect --policy all --output outputs/role2-fixtures-001.json
python -m experiments.ablate --size smoke --fixture-calibration --config base --seeds 7 --output outputs/role2-smoke-001
python -m experiments.frozen --checkpoint outputs/role2-smoke-001/base-seed7/policy.json --output outputs/role2-pairs-001
```

第一条独立演示 Mock 采集。第二条自行采集 Mock 轨迹、按题留出、拟合仿真并训练；它不读取第一条命令的输出。第三条仅验证冻结策略能接回 Mock Runner。省略 `--fixture-calibration` 且不传训练包时，使用手工定义的解析环境；`--size full` 只改变实验规模，不改变数据来源。两种演练都不能作为正式训练或独立测试证据。

| 查什么 | 看哪里 |
|---|---|
| 数据来源、标定假设、留出报告、跨种子对比 | 输出根目录的 `experiment.json`、`comparison.json` |
| 原始回报和损失、逐次验证指标、含 Mask 的轨迹 | 每个配置/种子目录的 `train.json`、`result.json`、`eval_steps.jsonl` |
| 冻结权重及来源 | 同目录的 `policy.json`；用于推理，不含优化器或 Replay Buffer，不能用来恢复训练。 |
