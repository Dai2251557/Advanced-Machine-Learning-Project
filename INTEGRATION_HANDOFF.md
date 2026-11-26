# 三方集成交接

DQN 调度器已在 [PR #3](https://github.com/Dai2251557/Advanced-Machine-Learning-Project/pull/3) 合入 `main`；冻结数据、Judge 协议、检索适配和训练侧 Rule 探针已在 [PR #5](https://github.com/Dai2251557/Advanced-Machine-Learning-Project/pull/5) 合入。这里区分三件事：接口能否接通、已有多少真实轨迹、能否据此比较策略。

## 离线核验：冻结数据 → 共享 Runner

在仓库根目录运行 `python -m utils.data_release` 可核验公开交付：600 篇摘要、30 道训练题和 10 道测试题的公开题面。以下示例只加载训练清单和训练题，使用默认 Mock 后端；可用 `python -m doctest INTEGRATION_HANDOFF.md` 直接测试，无需 API Key 或 PyTorch。

```pycon
>>> from core.rule_scheduler import RuleScheduler
>>> from core.runner import Runner
>>> from core.schema import ReviewTask
>>> from utils.data_release import ROOT, load_corpus, load_manifest, read_json
>>> corpus, manifest = load_corpus(), load_manifest('train')
>>> len(corpus['papers']), len(manifest['tasks'])
(600, 30)
>>> task = ReviewTask.from_dict(read_json(ROOT / 'data/train_tasks.json')[0])
>>> runner = Runner(corpus)
>>> result = runner.run_episode(RuleScheduler(runner.config), task)
>>> result['execution_kind'], result['success'], result['total_workflow_tokens']
('mock', True, 0)

```

Mock 验证的是数据能进入 Runner 并走到终稿，不是正式采样。真实采样必须使用配置好的后端和训练清单，保存 `collect_episode(..., mode='formal', manifest=manifest)` 的返回值，而非中间的 Runner 输出。

## 已有证据与下一步

| 交接 | `main` 上可核验的证据 | 仍待完成 |
|---|---|---|
| 执行层 → 训练 | [训练侧检索诊断](RETRIEVAL_REVIEW.md)；[R001、R005 两条 Rule 轨迹](handoff/role1_train_probes/README.md)包含真实用量，均为 `retrieve_more → summarize_fast → write_fast` | 确认 Verifier 口径与扩采预算，补足深档动作和不同训练题的覆盖。 |
| 评价 → 标定 | [冻结数据和 Judge 协议](ROLE3_HANDOFF.md)可用；工作流 Verifier 的分数不等于终稿 Judge 分数 | 由评价侧为成功的训练终稿生成与轨迹绑定的独立标签；按 `review_id` 留出，检验仿真的完成率、成本和质量代理误差。 |
| 训练 → 评测 | [二号训练与冻结入口](ROLE2_HANDOFF.md)已能用合成或 Mock 数据演练 | 在正式训练包上完成标定、消融并选定 checkpoint；然后在同批独立题上执行 Rule/DQN、封存、盲评。尚无可报告的独立比较结果。 |
