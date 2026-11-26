# 三方本地集成

集成分支：`codex/role1-data-integration`。基于团队 main `68582e2`，合入三号 `data-fix` 的 `e1a6c06`；同一时刻 `dwb` 的 `47efebe` 文件树相同。没有合并或推送远程 main，也没有覆盖原一号分支。

README 保留系统与二号说明，三号原 README 完整保存在 `ROLE3_HANDOFF.md`。忽略规则合并，移除误入版本控制的两个 Python 字节码文件；原始冻结数据、清单、Judge 协议与旧标签归档保持不变。

## 使用入口

```python
from core.runner import Runner
from core.schema import ReviewTask
from core.rule_scheduler import RuleScheduler
from utils.data_release import ROOT, load_corpus, load_manifest, read_json

corpus = load_corpus()
manifest = load_manifest('train')
task = ReviewTask.from_dict(read_json(ROOT / 'data/train_tasks.json')[0])
runner = Runner(corpus)  # 默认 Mock，只验证接口，不能作为正式轨迹。
result = runner.run_episode(RuleScheduler(runner.config), task)
```

正式采样需使用已配置的 DeepSeekBackend，并调用二号 `experiments.collect.collect_episode(..., mode='formal', manifest=manifest)`，保存函数返回的完整结果。此次没有发送新语料给外部服务，没有运行正式训练或测试。

## 当前核验与剩余事项

- `python -m utils.data_release`：600 篇、30 训练题、10 公开测试题，清单/摘要/任务哈希验证通过；未读取私有测试证据。
- 新增跨岗测试验证三号清单与二号哈希算法一致、公开训练题能进入 Runner 和 45 维编码器、Mock 不会被正式采样入口误收。
- 当前 Windows Python 未安装 PyTorch，训练相关测试被显式跳过，不能声称本机 DQN 训练已验收。
- 新数据格式问题已由三号适配器解决，不需要重命名原文件字段。
- 中文题面与英文摘要检索现已增加术语扩展和 BM25 排序，训练侧诊断见 `RETRIEVAL_REVIEW.md`；这不等于独立测试质量验收。R005、R001 两条真实训练侧 Rule 探针已完成，完整交接见 `handoff/role1_train_probes/README.md`。
- Judge 冻结协议已交付；工作流 Verifier 的口径、训练标定及两者区别仍需三方确认。
