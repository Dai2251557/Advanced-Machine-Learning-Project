# 固定 arXiv 摘要库上的综述调度实验：数据与终稿评测

主分支已收录冻结数据包 `arxiv600-review40-v1`：**600 篇历史 arXiv 摘要、30 道训练题、10 道独立测试题**，用于比较 Rule 与 DQN 在相同材料和预算下如何调度；不用于衡量开放网络检索或最新研究进展。语料保留原有带版本的 arXiv ID，发表年份为 2018（132 篇）、2019（33 篇）、2020（435 篇）；`data/corpus_chunk/` 的六个文件是主语料的顺序分片，不是额外 600 篇。题面以中文表述，证据边界是库内摘要，不把摘要未给出的数值、因果关系或跨论文胜负作为必答事实。

## 哪些文件交给谁

| 材料 | 位置 | 用途 |
|---|---|---|
| 固定语料 | `data/arxiv_corpus.json` | 唯一主语料；`utils.data_release.load_corpus()` 无损投影出 Runner 所需的 `id/title/abstract` 与数据用途。 |
| 公开任务 | `data/train_tasks.json`、`data/test_tasks.json` | 只含公开 `ReviewTask` 六字段。训练程序只读训练集；测试执行阶段才分发公开测试题。公开指策略可见，不是把测试题用于模型开发的授权。 |
| 数据交付清单 | `data/release.json` | 预先冻结的语料、任务及质检材料摘要、Judge 标识和推荐的首批 10 个训练题 ID。`load_manifest('train')` / `load_manifest('test')` 返回相应切分的清单。它能检测与此版交付不一致，不能自行证明来源身份；正式运行前由负责人独立保留可信副本。 |
| 原题逐题审查与训练题来源 | `curation/train_audit.json`、`curation/train_sources.json` | 前者逐题列出原题、原参考、处置及原因；后者是新题的出题种子和难度理由。两者都不传给策略或作为必引书单。 |
| 主题要点与证据 | `evaluation/train_criteria.json` | 数据方验题／人工复核用。`aspect` 须能从题面推出，`evidence` 是一组可查的摘要原文，不是唯一答案；Judge 不靠清单命中计分。 |
| 本地测试私有材料 | `.local-data/sealed/test_sources.json`、`.local-data/sealed/test_criteria.json` | 仅限数据负责人；独立于公开任务、训练和模型请求，已被 Git 忽略。**不会随 clone / push 交付**；负责人应另行受控保管，不能靠公开摘要哈希还原。 |

逐题审查和切分结果、论文覆盖的统计口径与首批标定理由见 [数据审查说明](curation/REPORT.md)。旧任务和原始文件可以从数据基线 `0e07995` 复查。旧采集脚本不是这份冻结快照的重建入口；本次没有重新抓取或修改论文摘要。

## 离线核验

从仓库根目录运行，不需要模型密钥，不调用网络：

```bash
python3 -B -m utils.data_release
python3 -B -m unittest discover -s tests
# 仅本地持有完整私有交付的数据负责人运行；只输出汇总，不显示测试金标。
python3 -B -m utils.data_release --with-private
```

训练端可以读取 `load_corpus()`、公开训练任务和 `load_manifest('train')`，无需导入 Judge 或给 Runner 发私有参考。测试时对完整公开测试批次使用**同一份预先封存**的 `load_manifest('test')` 输出；不要把待测输入现场重新计算成“可信清单”。`config_version=1` 对齐当前公共任务/Runner 格式；本批数据和 Judge 分别由 `source_id` 和 `judge.prompt_id` 绑定。不要将同名旧任务、旧 Prompt 的标签或 pilot/fixture 混入正式结果。

## 终稿 Judge 的边界

- **覆盖度是答没答全，不是引没引齐。** `coverage` 看对题面公开要求的回应；`factuality` 看事实主张是否有摘要证据、是否夸大；`citation_support` 看引用是否真的支撑对应论断；`consistency` 看全文是否自洽。题目之外的隐藏结论和预选名单召回率都不计分。证据不足时，承认不可比较优于编造结论。
- 训练与测试共用同一冻结 Rubric。只给真实成功终稿评分，按 `review_id + trajectory_id + final_text_sha256` 追溯，不按 `review_id` 覆盖同题不同终稿；失败留空。示例与真实标签分开。旧 `R001_judge.json` 已原样归档在 `outputs/examples/legacy/`，它不是交通题正式标签，不能作奖励。
- 训练时真实终稿分可作为终局标签。测试时 Rule 与冻结的 DQN 必须先完成整批并保存，再离线盲评；Judge 分数、测试参考和 Judge Token 都不进入工作流。已交付的 [R005、R001 训练探针](handoff/role1_train_probes/README.md)有真实工作流调用，但都没有独立 Judge 标签。仓库中的离线检查验证的是字段、绑定及证据读取，不是 Judge 实评的一致性。

错配分类与终稿评分分别验收：`experiments.ablate.metrics` 已统计五动作调用次数、合法机会选择率和 Token 占比，但 `mismatch_status` 仍是 `awaiting_frozen_role3_classifier`。这表示尚无冻结的错配分类结果，不表示零错配。

**调用迁移**：旧的 `judge_final_review(..., api_key=...)` 不再是正式入口。新版不加载密钥，没有自动网络传输。对已通过 `experiments.collect.validate_episode` 的真实训练轨迹，使用 `evaluation.judge.prepare_train_review(episode=..., frozen_papers=load_corpus()['papers'], manifest=load_manifest('train'))`。对正式测试，使用 `prepare_sealed_test_reviews(execution_dir=..., public_tasks=..., frozen_papers=..., manifest=load_manifest('test'))`；它检查 `execution_complete.json`、整批 Rule/DQN 配对和轨迹字节哈希，不接受边跑边评。无成功终稿则无待评材料。

训练和测试都用 `build_judge_request(prepared)` 构造完整的 Chat Completions 请求体，不手工拼接。第一条 `system` 消息是固定 Rubric；第二条 `user` 消息是规范化 JSON（题面、终稿、正文引文证据和文末书目）。请求不含 split、策略名、出题参考或本地 `binding`。正式配置为 `model=deepseek-v4-pro`、`thinking.type=disabled`、`temperature=0`、`max_tokens=2048`、`stream=false`、`response_format.type=json_object`，不附加其他选项。`judge.prompt_id` 的 v3 标识覆盖整个模板与参数；每条 `binding.request_sha256` 另绑定该终稿的具体请求体。题目与 Rubric 文字未变，但旧 v2 回执不是 v3 回执；不要给旧标签改标。

此函数只构造本地数据。实际打分须另获模型、外发范围与预算授权；服务商是否支持并遵守这些参数尚未经在线验证，不允许静默改参或切换供应商。获批后，评分方须在请求发出时留存实际请求体，与同次原始响应成对归档。将 `receipt={"binding": 本地绑定, "request": 实际请求体, "response": 原始响应}` 交给 `finalize_judge_result(prepared, receipt)`，校验参数、消息、终稿、四分项、等权均分和依据，再用 `save_judge_result(label, out_dir)` 排他创建轨迹标签。协议一致性检查不证明服务端实际执行行为。`python3 -B -m utils.run_judge_example` 只向 stdout 演示一个 `synthetic_fixture`，不可作训练奖励。
