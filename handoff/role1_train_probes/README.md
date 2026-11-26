# 一号位真实训练探针交接（2026-11-26）

[R005-collection.json](R005-collection.json) 和 [R001-collection.json](R001-collection.json) 是三号冻结语料上的**真实训练侧 Rule 轨迹**，由共享 Runner、`DeepSeekBackend` 和二号 `collect_episode(..., mode='formal')` 生成，并由 `validate_episode` 校验。语料版本为 `arxiv600-review40-v1`，代码基线为 `390c572`，检索方法为 `bilingual_bm25_v1`，Runner 使用仓库 `config.json`。仅用于训练标定与接口核验；没有 Judge 标签，不属于独立测试，也不证明终稿质量。

| 任务 | 动作 | 成功调用 | 工作流 Token | Verifier 最后四维评分 | 仓库交付文件 SHA-256（LF） |
|---|---|---:|---:|---|---|
| R005 | retrieve_more → summarize_fast → write_fast | 4 | 11,215 | 1.00 / 0.90 / 0.80 / 1.00 | `556591b7172d582b7a4e2afd0f5746bd79a7f871308db27cfc7cc85cda4e9023` |
| R001 | retrieve_more → summarize_fast → write_fast | 4 | 15,290 | 0.85 / 0.90 / 0.65 / 0.90 | `5c37a2be0a242c625203e00903b06f0ffd4fa316d362495d54b888a6799792bb` |

仓库发布的两份 JSON 使用 LF，完整性测试直接校验文件字节。原 Windows 采样文件只在换行（CRLF）上不同，其 SHA-256 分别为 R005 `8cef4d21e3ab62c17a2c97097769fb2d7f6ae4124df0a64bd44e6d7751c73421`、R001 `ac4ea4e4fddc146525a74758a83defbfdeefda6aa62b3ee892645bafa01b781f`。

四维顺序为 coverage / factuality / citation_support / consistency。两条轨迹均为三步完成，合计 26,505 个已报告工作流 Token、8 次成功 API 调用。服务端响应的模型标识为 `deepseek-flash`；不能由此推断更细的底层版本。所有调用都报告了输入和输出 Token，没有输出修复。Judge 未调用，以上四维是工作流过程诊断，不能当作终稿 Judge 分。

另一次 R005 试跑在首次 Verifier 请求发生 `connection_error`，用量未知，按 `usage_unknown` 终止且未继续 R001。这次失败不计入上表，两份交付 JSON 也不含该次记录；不能据此断言失败请求的服务端费用为零。

观察到的质量风险：R005 检索的第三篇被 Verifier 指为主题无关，但写作契约要求覆盖全部选文；R001 的 citation_support 为 0.65，等权综合分 0.825 仍使固定规则选择 `write_fast`。这些是训练侧观察，不据此单方修改检索、门槛或五动作。请在正式标定前由一、二、三号位确认 Verifier 尺度、平均分门槛与低维度处理，再冻结配置和基线。

二号位可直接读取两个 `*-collection.json` 校验采样与状态支持，勿当作 Judge 已标注数据。三号位若评训练终稿，应使用冻结 Judge 对两份成功轨迹分别生成外部标签，保持 `review_id + trajectory_id + final_text_sha256` 绑定，不把标签回写轨迹。交接文件保留完整状态、调用日志及材料；交付时已核查没有 API Key 或 Authorization 头。原始 Runner 输出目录不随仓库交付。

交付时的验证记录：冻结数据完整性检查通过；历史环境使用 PyTorch `2.14.1+cpu`，133 项测试、`tests/verify-agents.ps1` 与 `git diff --check` 均通过。这不是对仓库训练依赖 `requirements-rl.txt`（固定 `torch==2.15.0`）的安装验证，也不是正式 DQN 训练结果。
