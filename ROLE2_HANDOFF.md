# 2 号位交接：训练工作流

目前可以验证接口、数值训练和数据管道。**合成环境、Mock Runner 轨迹、用 Mock 轨迹拟合的经验环境，都不是正式训练或独立测试证据。** 2 号位只使用 `core/runner.py`；不修改 1 号位的执行器或 Rule，不调用 Judge。

## 复现离线演练

本机用 Python 3.12.13。训练环境 `.venv/` 的直接依赖见 `requirements-rl.txt`；不需要 NumPy 桥接、Gymnasium 或 GPU。冻结策略 `rl.scheduler.DQNScheduler`、采样和标定工具只用标准库。以下命令从仓库根执行，输出路径用**不存在的新路径**，不覆盖旧结果：

```bash
.venv/bin/python -B -m unittest discover -s tests
.venv/bin/python -B -m experiments.collect --policy all --output /tmp/role2-new-fixtures.json
.venv/bin/python -B -m experiments.ablate --size smoke --fixture-calibration --output outputs/my-role2-smoke
.venv/bin/python -B -S -m experiments.frozen --checkpoint outputs/my-role2-smoke/base-seed7/policy.json --output outputs/my-role2-pairs
```

`--fixture-calibration` 显式走 T4 → T5 → 训练：合成公开任务经共享 Mock Runner 采集、按任务留出、用训练部分拟合仿真，再训练和评估。这里 Mock 成本为零、有效分恒定，结果只说明能接通。若省略该标志，使用 T2 手工定义的、有快慢动作差异和非零成本的解析环境。两者都不能用于正式结论。`--size full` 只改变训练量，**不会将模拟来源变为真实来源**。

## 正式数据交接时做什么

1. **先核验来源，再采样。** 3 号位交付正式语料、10 个训练任务、冻结 Judge 标识及一条脱敏评分样例。独立的可信 manifest 与 `experiments/collect.py` 文件头的字段对照：公共任务和整库论文摘要用规范化 JSON 哈希；终稿用原始 UTF-8 文本哈希。`source_id` 应标明交付版本。当前适配器要求 `dataset_kind=arxiv_abstracts`、`purpose=formal_experiment`。这些是 2 号位的摄取格式，不是要求 3 号位修改公共 Schema；若实际命名不同，先确认适配关系，不应重标 pilot。不要把测试标签/金标交给训练入口。
2. **限范围复审 T4→T5。** 正式任务和 Judge 样例都收到后，再用 `/agy-coplan --read` 复审，而不是在 T3 提交后触发。优先范围：`CONTRACT.md`、`SCORING_PROTOCOL.md`、`experiments/collect.py`、`rl/{adapter,calibration,synthetic_env,reward}.py`、对应测试，以及经授权可外发的脱敏训练任务/评分样例（实际传参逐个列路径）。不选整个数据目录、测试集、`.env*`、密钥、Judge 私有配置、原始调用 payload 或未获准外发的语料。重点核对字段映射、缺失用量、轨迹/终稿关联、代理与真分区分、按题留出、零支持分支假设。无法安全分离时先停，不以整目录代替。
3. **正式探针。** 确认费用、外发范围、Verifier 口径和 Runner 配置后，先在训练集用 Mock 低成本核查检索；再显式提供配置好的共享 Runner，调用 `collect_episode(runner, task, policy, mode='formal', manifest=...)`。先 1–2 个代表任务，核对价格、延迟、调用修复、评分和状态覆盖后再扩采。默认 CLI 不会建真实后端。保存 collect 函数的**返回值**：它比原 Runner 的中间产物多 `next_mask`、来源与采样计划。`support_report` 同时报每动作可选机会、被选次数、独立任务数。不同前缀不是严格反事实。
4. **Judge 分开交付。** 只对成功的真实训练终稿打分；外部 sidecar 每条含 `review_id, trajectory_id, split, final_text_sha256, source_kind=independent_judge, model, prompt_id, overall_score, subscores, rationale`。由 `validate_collection` 与轨迹关联，不改变 Workspace、工作流成本或物理 Mask。失败、漏评和真实零分不同。记录 workflow 代码/Prompt 和实际后端版本；版本不明不猜测，不混用不兼容数据。
5. **拟合与留出。** 训练 bundle 为 JSON 对象 `{episodes, labels, manifest, holdout_review_ids}`。先登记训练内的留出综述 ID；同一任务的重复轨迹一起划分。`evaluate_holdout` 输出 Rule 的实际/模拟完成率、成本、可比增量及分母，另给终稿代理 MAE 和训练均值基线 MAE。无 Rule 留出或未标注时相应项为空，不是零误差。`fit_calibration` 只用拟合部分，`model.manifest()` 列各桶的样本和独立综述数。至少两个不同训练题支持时，代理用同阶段/写作档位的 Verifier 残差均值；否则只用有效 Verifier，并标明来源。无有效质量依据的成功终局不伪造质量奖励：可用 `model.make_env(reward_config=RewardConfig(quality_weight=0))` 诊断成本，正式质量训练应先补样本。
6. **消融与冻结。** 使用 `experiments.ablate --size full --training-bundle <训练包> --threshold <预先登记值> --output <新目录>`。默认种子 `7,17,27`；容量 `512/2048/8192`，Target-Net 间隔 `50/200/800`（梯度步），共享 `2048/200` 对照，共 15 次、每次 20,000 环境步，CPU 单线程。每 500 步使用固定仿真场景贪心验证；连续三次达到门槛，才记达到时的步数，否则为 `null`。训练内决定 checkpoint，不能使用测试结果选权重。完全无观测的合法分支在当前仿真中**假设**为用量未知的截尾终止；它不是测得的失败概率，会影响策略偏好。若回退率高、代理误差大或几乎只采到规则路径，先补样/检验敏感性，不应把该策略称为已验证最优。
7. **独立测试。** 至少 5 个封存 `review_id`。由获得真实执行授权的操作者将相同 Runner、公开测试任务和已选正式 checkpoint 传给 `experiments.frozen.execute_pairs(..., mode='formal', test_manifest=sealed_delivery)`。`sealed_delivery` 由数据负责人提前独立封存：含 `source_id`、`config_version`、`dataset_kind`、`data_purpose`、整库 `papers`（ID → 公开论文哈希）及仅含本批全部测试题的 `tasks`（ID → `{split: 'test', task_sha256: ...}`），哈希算法同第 1 步。不要用待执行的候选输入临时生成清单。入口在创建输出目录或发起调用之前核对整批任务及语料；完成索引记录交付编号和清单摘要，不向策略提供清单或测试标签。交错策略顺序、失败也保存。`execution_complete.json` 只在整批 Rule/DQN 都执行结束后生成，含文件哈希；没有这个索引的目录不是完成的批次。随后由 3 号位另行盲评；不要把带策略名的文件直接交给盲评者。当前 CLI 是 Mock 演练，不能替代正式 ≥5 题比较。

## 产物和边界

- `experiment.json`：配置网格、种子、Python/PyTorch 版本、数据来源、标定假设和留出报告。`comparison.json`：跨种子的回报摘要、收敛步数及各自最终验证指标。
- 每个配置/种子目录：`train.json`（未平滑原始步级回报与损失）、`result.json`（逐次独立评估、成本和五动作负载）、`eval_steps.jsonl`（含完整状态、有效位、动作、机会 Mask 的模拟验证轨迹）、`policy.json`（权重、编码顺序、来源和训练配置，不含 Judge）。
- `mismatch_status=awaiting_frozen_role3_classifier` 是有意保留的边界。所有判定输入均可从评估轨迹读取；3 号位冻结分类器后，对同一份轨迹复算 `mismatch_any` 和分类明细。不能用低收益历史次数替代错配次数，写作无后置 Verifier 增量。
- 冻结推理仅需 `DQNScheduler.load(path)`，再调用 `runner.run_episode(scheduler, task, seed)`。不含探索、重放或梯度更新。训练 checkpoint 用于推理，不是恢复优化器/Replay 继续训练的断点。
