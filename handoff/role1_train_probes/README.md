# 一号位真实训练探针交接（2026-11-26）

这两份 `collection.json` 是三号冻结语料上的**真实训练侧 Rule 轨迹**，由共享 Runner、`DeepSeekBackend` 和二号 `collect_episode(..., mode='formal')` 生成，并由 `validate_episode` 校验。语料版本为 `arxiv600-review40-v1`，代码基线为 `390c572`，检索方法为 `bilingual_bm25_v1`，Runner 使用仓库 `config.json`。仅用于训练标定与接口核验；没有 Judge 标签，不属于独立测试，也不证明终稿质量。

| 任务 | 动作 | 成功调用 | 工作流 Token | Verifier 最后四维评分 | 完整采样文件 SHA-256 |
|---|---|---:|---:|---|---|
| R005 | retrieve_more → summarize_fast → write_fast | 4 | 11,215 | 1.00 / 0.90 / 0.80 / 1.00 | `8cef4d21e3ab62c17a2c97097769fb2d7f6ae4124df0a64bd44e6d7751c73421` |
| R001 | retrieve_more → summarize_fast → write_fast | 4 | 15,290 | 0.85 / 0.90 / 0.65 / 0.90 | `ac4ea4e4fddc146525a74758a83defbfdeefda6aa62b3ee892645bafa01b781f` |

四维顺序为 coverage / factuality / citation_support / consistency。两条轨迹均为三步完成，合计 26,505 个已报告工作流 Token、8 次成功 API 调用。服务端响应的模型标识为 `deepseek-flash`；不能由此推断更细的底层版本。所有调用都报告了输入和输出 Token，没有输出修复。Judge 未调用，以上四维是工作流过程诊断，不能当作终稿 Judge 分。

第一次在受限网络中试跑 R005 时，首次 Verifier 请求发生 `connection_error`，用量未知，因此按照 Runner 规则以 `usage_unknown` 终止，未继续 R001。失败记录保留在本机 `outputs/train-probes-bilingual-v1-20261126/`，**不计入上表成功样本**；也不能断言该失败请求的服务端费用为零。放开网络后使用新输出目录完成上表两条任务，原失败记录未改写。

观察到的质量风险：R005 检索的第三篇被 Verifier 指为主题无关，但写作契约要求覆盖全部选文；R001 的 citation_support 为 0.65，等权综合分 0.825 仍使固定规则选择 `write_fast`。这些是训练侧观察，不据此单方修改检索、门槛或五动作。请在正式标定前由一、二、三号位确认 Verifier 尺度、平均分门槛与低维度处理，再冻结配置和基线。

二号位可直接读取两个 `*-collection.json` 校验采样与状态支持，勿当作 Judge 已标注数据。三号位若评训练终稿，应使用冻结 Judge 对两份成功轨迹分别生成外部标签，保持 `review_id + trajectory_id + final_text_sha256` 绑定，不把标签回写轨迹。交接文件保留完整状态、调用日志及材料；已核查没有 API Key 或 Authorization 头。原始 Runner 磁盘轨迹在本机 `outputs/train-probes-bilingual-v1-20261126-network/traces/`，不重复提交。

交付前验证：冻结数据完整性检查通过；`.venv` 使用 PyTorch `2.14.1+cpu` 跑完 133 项测试，全部通过；`tests/verify-agents.ps1` 与 `git diff --check` 通过。仓库 `requirements-rl.txt` 当前写的是 `torch==2.15.0`，但 2026-11-26 官方 PyPI 未提供该版本，二号位需修正依赖清单后才能按文档复现；本次没有改动其训练依赖文件或算法。
