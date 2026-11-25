# 一号位真实 pilot 验收（2026-11-24）

输入为用户已授权发送的五篇 AgentSkill 功能测试论文、`examples/runner_task.json` 和运行中间产物；目的地为 DeepSeek 官方 API。全部运行均标记 `pilot / pilot_only / real`，没有调用 Judge，不属于正式训练、标定或独立测试。

## 固定规则完整流程

轨迹：`outputs/pilot/role1-acceptance/episode-1ed59c78a4324f66a2d2d808a975dc88/episode.json`。

| 动作 | 成功 | 实际工作流 Token |
|---|---|---:|
| retrieve_more（本地检索及真实 Verifier） | 是 | 2415 |
| summarize_fast（总结及真实 Verifier） | 是 | 8090 |
| write_fast | 是 | 4062 |
| 合计 | final_written | 14567 |

共 3 个动作、4 次 API 调用，无输出修复。响应模型标识为 `deepseek-flash`（服务商未返回更具体底层版本，不推断）。最终四维诊断为 1.0 / 0.9 / 0.85 / 1.0，V=0.9375，固定规则据此撰写终稿。这是过程诊断，不是独立终稿质量评分。

终稿在同目录 `report.md`，包含 PILOT 标记、摘要范围说明和引用。人工查看发现检索带入一篇非核心相关论文，终稿以范围说明单独处理；检索质量仍需正式数据验证。

## 深度动作独立冒烟

命令：`python -m core.acceptance --backend deepseek`。该入口显式执行检索、快速总结、深度总结、深度写作，用于接口验收，不作为固定规则或 DQN 策略。

轨迹：`outputs/pilot/action-acceptance/episode-4e89ba692e9d4752992278395f954074/episode.json`。

| 动作 | 成功 | 实际工作流 Token |
|---|---|---:|
| retrieve_more | 是 | 2242 |
| summarize_fast | 是 | 9383 |
| summarize_deep | 是 | 10382 |
| write_deep | 是 | 4903 |
| 合计 | final_written | 26910 |

共 4 个动作、6 次 API 调用。两次 pilot 合计 41477 Token，覆盖全部五动作和自动 Verifier。深度动作请求档位为 high，快速动作与 Verifier 为 low。日志已核对逐调用、逐步骤和 Episode Token 一致，且没有包含本地 API Key。

本次全量 67 项自动测试通过，AGENTS 验证及 git diff 空白检查通过。真实验收仅有两个 pilot 任务执行实例，不代表统计意义上的模型质量或调度收益证明。

## 预算和评分状态

预算耗尽或用量未知现在优先于终稿成功；边界测试包含恰好达到、超过、低于预算以及用量未知，失败时不发布正式报告，保留原始候选响应。在预算内于最后一步生成终稿仍允许成功。真实调用仍可能使实际账单超出阈值，这不是服务商计费硬限额。

评分定义、等权 V 和候选阈值已整理至 `SCORING_PROTOCOL.md`，仍待三号确认及正式训练集标定。未声称完成三方冻结、实际 DQN 联调或正式实验。
