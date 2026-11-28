"""全部使用自造字符串、内存轨迹与伪回执；不读数据集，不调用模型，不写标签。"""

import copy
import hashlib
import json
import unittest
from pathlib import Path
from unittest.mock import mock_open, patch

from evaluation._sealed import sealed_episodes
from evaluation.judge import (AGGREGATION, FIXTURE_MODEL, JUDGE_MODEL, PROMPT_ID, RUBRIC_PROMPT, SCORE_KEYS,
                              build_judge_request, finalize_judge_result, prepare_fixture_review, prepare_sealed_test_reviews,
                              prepare_train_review, save_judge_result)


PAPERS = [{"id": "toy/1v2", "title": "颜色实验", "abstract": "盒子输出输入色块的颜色；仅测试过红色。"},
          {"id": "toy-B", "title": "重量实验", "abstract": "天平仅测量物体重量。"},
          {"id": "unused", "title": "旁支实验", "abstract": "不应作为隐含背景的无关摘要。"}]
INSTRUCTION = "说明盒子的输入输出，并讨论适用范围。"
REPORT = "# 文献综述\n\n范围：仅依据提供的论文摘要。\n\n## 方法\n\n盒子返回输入色块的颜色。 [toy/1v2]\n\n## 参考文献\n\n- [toy/1v2] 颜色实验\n"
SCORES = dict(coverage=0.62, factuality=0.62, citation_support=0.63, consistency=0.63, overall=0.63,
              rationale="离线数值协议用例，非实际评分。")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def digest(obj):
    return sha(json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())


def fixture(text=REPORT, tid="T", split="train"):
    return prepare_fixture_review(survey_instruction=INSTRUCTION, final_text=text, frozen_papers=PAPERS,
                                  review_id="same-task", trajectory_id=tid, split=split)


def receipt(prepared, content=None):
    return {"binding": dict(prepared["binding"]), "request": build_judge_request(prepared),
            "response": {"model": prepared["binding"]["model"], "choices": [
        {"finish_reason": "stop", "message": {"content": content if content is not None else json.dumps(SCORES)}}]}}


def task(rid, split):
    return {"config_version": 1, "review_id": rid, "split": split, "survey_instruction": INSTRUCTION,
            "difficulty": "easy", "corpus_scope": "local_arxiv_ai"}


def delivery(tasks):
    return {"config_version": 1, "source_id": "test-program-only", "dataset_kind": "arxiv_abstracts",
            "data_purpose": "formal_experiment", "papers": {p["id"]: digest(p) for p in PAPERS},
            "tasks": {t["review_id"]: {"split": t["split"], "task_sha256": digest(t)} for t in tasks},
            "judge": {"model": JUDGE_MODEL, "prompt_id": PROMPT_ID}}


def episode(t, tid, success=True):
    # 只仿造对接所需字段，不把这些对象当作 Runner 实测来源。
    return {"trajectory_id": tid, "review_id": t["review_id"], "split": t["split"], "execution_kind": "real",
            "dataset_kind": "arxiv_abstracts", "data_purpose": "formal_experiment", "success": success,
            "terminal_reason": "final_written" if success else "max_steps", "final_text": REPORT if success else None}


def train_inputs(success=True):
    t = task("program-train", "train")
    m = delivery([t])
    ep = dict(episode(t, "hidden-policy-id", success), collection_kind="formal", public_task=t,
              source={"manifest_sha256": digest(m)}, final_text_sha256=sha(REPORT.encode()) if success else None,
              policy="hidden-policy", source_papers=[{"id": "toy/1v2", "abstract": "不能从轨迹替代冻结原文"}])
    return dict(episode=ep, frozen_papers=PAPERS, manifest=m)


def batch_inputs(count=5, policies=("rule", "dqn")):
    tasks = [task(f"program-test-{i}", "test") for i in range(count)]
    m = delivery(tasks)
    files = {}
    rows = []
    for i, t in enumerate(tasks):
        for name in policies:
            success = (i, name) != (0, "rule")
            tid = f"hidden-{name}-{i}"
            ep = episode(t, tid, success)
            reason = ep['terminal_reason']
            ep['transitions'] = [{'action_id': 'write_fast' if success else 'retrieve_more', 'success': True,
                                  'done': True, 'terminal_reason': reason,
                                  'state_after': {'terminal': True, 'terminal_reason': reason}}]
            raw = json.dumps(ep, ensure_ascii=False).encode()
            path = f"{i:03}-{name}.json"
            files[path] = raw
            rows.append(dict(review_id=t["review_id"], policy=name, trajectory_id=tid, success=success,
                             trace=path, sha256=sha(raw)))
    seal = {"mode": "formal", "all_episodes_complete": True, "test_delivery": {"source_id": m["source_id"],
            "manifest_sha256": digest(m)}, "episodes": rows}
    return dict(public_tasks=tasks, manifest=m, frozen_papers=PAPERS, execution_dir="/virtual-seal"), files, seal


class RubricEvidenceTests(unittest.TestCase):
    def test_correct_nonprescribed_paper_uses_exact_frozen_source_not_trajectory_or_inventory(self):
        p = prepare_train_review(**train_inputs())
        blind = p["blind_input"]
        self.assertEqual(blind["body_citations"], [{"marker": "[toy/1v2]", "status": "matched", **PAPERS[0]}])
        self.assertEqual(set(blind), {"rubric", "survey_instruction", "body_citations", "bibliography", "final_text"})
        self.assertEqual(blind["survey_instruction"], INSTRUCTION)
        wire = json.dumps(blind, ensure_ascii=False)
        for forbidden in ("hidden-policy", "program-train", "不能从轨迹替代", PAPERS[2]["abstract"], "toy-B"):
            self.assertNotIn(forbidden, wire)
        self.assertIn("任何库内论文只要确实支撑该论断就同等认可", blind["rubric"])

    def test_id_exists_but_abstract_does_not_support_claim_is_not_auto_endorsed(self):
        bad = REPORT.replace("[toy/1v2]", "[toy-B]")
        blind = fixture(bad)["blind_input"]
        self.assertEqual(blind["body_citations"], [{"marker": "[toy-B]", "status": "matched", **PAPERS[1]}])
        self.assertIn("盒子返回输入色块的颜色。 [toy-B]", blind["final_text"])
        self.assertEqual(blind["bibliography"], [{"listed_id": "toy-B", "listed_title": "颜色实验", "frozen_title": "重量实验"}])
        self.assertIn("精确命中 ID 只确认论文身份，不确认支持关系", blind["rubric"])
        self.assertNotIn("subscores", blind)

    def test_unknown_and_unparsed_citations_survive_and_reference_list_is_not_body_evidence(self):
        text = "两项结论 [toy/1v2] [toy/1v1] [混合, ?]；另据 Smith (1900)，[残缺\n\n## 参考文献\r\n\n- [toy-B] 重量实验\r\n- [幽灵] 虚构书目\r\n"
        blind = fixture(text)["blind_input"]
        self.assertEqual(blind["final_text"], text)
        self.assertEqual(blind["body_citations"][1:], [{"marker": "[toy/1v1]", "status": "no_exact_match"},
                                                       {"marker": "[混合, ?]", "status": "no_exact_match"}])
        self.assertEqual([c["marker"] for c in blind["body_citations"]], ["[toy/1v2]", "[toy/1v1]", "[混合, ?]"])
        self.assertNotIn(PAPERS[1]["abstract"], json.dumps(blind, ensure_ascii=False))
        self.assertEqual(blind["bibliography"][1], {"listed_id": "幽灵", "listed_title": "虚构书目", "frozen_title": None})

    def test_omitted_aspect_honest_limit_and_exaggeration_keep_same_evidence_but_distinct_claims(self):
        texts = [REPORT, REPORT.replace(" [toy/1v2]", " 其他颜色的证据不足。 [toy/1v2]"),
                 REPORT.replace(" [toy/1v2]", " 所有颜色已验证。 [toy/1v2]")]
        inputs = [fixture(s)["blind_input"] for s in texts]
        self.assertEqual([i["final_text"] for i in inputs], texts)
        self.assertTrue(all(i["body_citations"] == inputs[0]["body_citations"] for i in inputs))
        self.assertTrue(all(i["survey_instruction"] == INSTRUCTION for i in inputs))
        for rule in ("漏答题点记在 coverage", "诚实说明已有证据不足并限定结论不是事实错误", "不是论文匹配率",
                     "所附摘要不是整个冻结库", "只看终稿内部"):
            self.assertIn(rule, RUBRIC_PROMPT)
        # 这里只能验证裁判拿到不同论断、相同证据和冻结准则；没有检验它会如何打分。

    def test_exact_request_and_version(self):
        prepared = prepare_train_review(**train_inputs())
        fields = ("survey_instruction", "final_text", "body_citations", "bibliography")
        user = {k: prepared['blind_input'][k] for k in fields}
        body = build_judge_request(prepared)
        self.assertEqual(body, {"model": JUDGE_MODEL, "temperature": 0, "thinking": {"type": "disabled"},
            "max_tokens": 2048, "stream": False, "response_format": {"type": "json_object"}, "messages": [
                {"role": "system", "content": RUBRIC_PROMPT}, {"role": "user", "content": json.dumps(
                    user, sort_keys=True, separators=(",", ":"), ensure_ascii=False)}]})
        self.assertEqual(prepared['binding']['request_sha256'], digest(body))
        body['messages'][1]['content'] = json.dumps({k: '{' + k + '}' for k in fields}, sort_keys=True, separators=(',', ':'))
        self.assertEqual(PROMPT_ID, 'dwb-final-v3-sha256:' + digest(body))

    def test_receipt_requires_actual_frozen_request(self):
        p = fixture()
        other = fixture(text='另一份终稿')
        r = receipt(p)
        body = r['request']
        for changed in (None, build_judge_request(other), dict(body, temperature=0.5),
                        dict(body, messages=body['messages'][::-1]), dict(body, stream=True)):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                finalize_judge_result(p, dict(r, request=changed))


class ReceiptTests(unittest.TestCase):
    def test_fixture_and_decimal_half_up_and_raw_utf8(self):
        for split in ("train", "test"):
            p = fixture(" \n" + REPORT, split=split)
            label = finalize_judge_result(p, receipt(p))
            self.assertEqual(label["source_kind"], "synthetic_fixture")
            self.assertEqual(label["model"], FIXTURE_MODEL)
            self.assertEqual(label["split"], split)
            self.assertEqual(label["trajectory_id"], "T")
            self.assertEqual(label["review_id"], "same-task")
            self.assertEqual(label["final_text_sha256"], sha((" \n" + REPORT).encode()))
            self.assertNotEqual(label["final_text_sha256"], sha(REPORT.strip().encode()))
            self.assertEqual(label["prompt_id"], PROMPT_ID)
            self.assertEqual(label["subscores"], {k: SCORES[k] for k in SCORE_KEYS})
            self.assertEqual(label["rationale"], SCORES["rationale"])
            self.assertEqual(label["aggregation"], AGGREGATION)
            self.assertEqual(label["overall_score"], 0.63)

    def test_reject_incomplete_nonnumeric_nonfinite_out_of_range_precision_mean_duplicate_and_empty_reason(self):
        p = fixture()
        malformed = [{k: v for k, v in SCORES.items() if k != "consistency"}, dict(SCORES, unsupported=1)]
        malformed += [dict(SCORES, coverage=v) for v in (True, None, "0.62", float("nan"), float("inf"), -0.1, 1.01, 0.621)]
        malformed += [dict(SCORES, overall=v) for v in (False, 0.62, 0.625)] + [dict(SCORES, rationale=" \n")]
        for content in [json.dumps(o) for o in malformed] + [json.dumps(SCORES).replace("{", '{"coverage":0.62,'), "not JSON"]:
            with self.subTest(content=content), self.assertRaises(ValueError):
                finalize_judge_result(p, receipt(p, content))

    def test_reject_crossed_identity_changed_material_model_or_incomplete_response(self):
        p = fixture()
        for field in ("source_kind", "review_id", "trajectory_id", "split", "prompt_id", "final_text_sha256", "blind_input_sha256"):
            r = receipt(p)
            r["binding"][field] = "different"
            with self.subTest(field=field), self.assertRaises(ValueError):
                finalize_judge_result(p, r)
        r = receipt(p)
        for change in ({"model": JUDGE_MODEL}, {"choices": []}, {"error": "failure"},
                       {"choices": [{"finish_reason": "length", "message": {"content": json.dumps(SCORES)}}]}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                finalize_judge_result(p, dict(r, response=dict(r["response"], **change)))
        altered = copy.deepcopy(p)
        altered["blind_input"]["body_citations"][0]["abstract"] = "被替换的摘要"
        with self.assertRaises(ValueError):
            finalize_judge_result(altered, r)

    def test_saver_uses_distinct_trajectory_paths_and_exclusive_creation(self):
        p1, p2 = fixture(tid="A"), fixture(tid="B")
        l1, l2 = (finalize_judge_result(p, receipt(p)) for p in (p1, p2))
        with patch.object(Path, "mkdir"), patch.object(Path, "open", mock_open()) as writer:
            for invalid in (p1["binding"], dict(l1, overall_score=0), dict(l1, source_kind="independent_judge")):
                with self.assertRaises(ValueError):
                    save_judge_result(invalid, "/virtual-output")
            writer.assert_not_called()
            path1 = save_judge_result(l1, "/virtual-output")
            path2 = save_judge_result(l2, "/virtual-output")
            self.assertNotEqual(path1, path2)
            for call in writer.call_args_list:
                self.assertEqual(call.args, ("x",))
            writer.side_effect = FileExistsError
            with self.assertRaises(FileExistsError):
                save_judge_result(l1, "/virtual-output")


class FormalBoundaryTests(unittest.TestCase):
    def test_train_hash_delivery_split_success_and_public_task_checks(self):
        self.assertIsNone(prepare_train_review(**train_inputs(False)))
        for change in ({"execution_kind": "mock"}, {"collection_kind": "fixture"}, {"split": "test"},
                       {"final_text_sha256": sha(REPORT.strip().encode())}, {"source": {"manifest_sha256": "bad"}},
                       {"success": False}, {"public_task": dict(task("program-train", "train"), hidden_hint="unused")}):
            data = train_inputs()
            data["episode"].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                prepare_train_review(**data)
        for corpus in (PAPERS[:1], [*PAPERS, PAPERS[0]], [dict(PAPERS[0], abstract="替换原文"), *PAPERS[1:]]):
            with self.subTest(corpus=corpus), self.assertRaises(ValueError):
                prepare_train_review(**dict(train_inputs(), frozen_papers=corpus))
        data = train_inputs()
        data["manifest"]["judge"]["prompt_id"] = "old-prompt"
        data["episode"]["source"]["manifest_sha256"] = digest(data["manifest"])
        with self.assertRaises(ValueError):
            prepare_train_review(**data)

    def test_sealed_mocked_batch_matches_train_rubric_and_hides_policy_and_skips_failure(self):
        args, files, seal = batch_inputs()
        files["execution_complete.json"] = json.dumps(seal).encode()
        with patch.object(Path, "read_bytes", lambda path: files[path.name]):
            prepared = prepare_sealed_test_reviews(**args)
        self.assertEqual(len(prepared), 9)
        self.assertNotIn("hidden-rule-0", [p["binding"]["trajectory_id"] for p in prepared])
        train = prepare_train_review(**train_inputs())
        for p in [train, *prepared]:
            self.assertEqual(p["blind_input"], train["blind_input"])
            self.assertEqual(build_judge_request(p), build_judge_request(train))
            self.assertEqual(p["binding"]["model"], JUDGE_MODEL)
            self.assertEqual(p["binding"]["source_kind"], "independent_judge")
            self.assertNotIn("overall_score", p)
            with self.assertRaises(ValueError):
                finalize_judge_result(p, receipt(fixture(split=p["binding"]["split"])))
        # 借用正式回执外形做内存协议测试；它不能证明模型调用或评分质量，也不写磁盘。
        formal_shape_only = finalize_judge_result(train, receipt(train))
        self.assertEqual(formal_shape_only["source_kind"], "independent_judge")
        self.assertEqual(formal_shape_only["subscores"], {k: SCORES[k] for k in SCORE_KEYS})
        self.assertEqual(formal_shape_only["final_text_sha256"], sha(REPORT.encode()))

    def test_three_policy_seal_requires_complete_unique_batch_and_checkpoint_metadata(self):
        policies = ["rule", "v1", "v2"]
        args, files, seal = batch_inputs(count=10, policies=policies)
        ids = {t["review_id"] for t in args["public_tasks"]}
        # Reader 检查权重摘要的键与格式，不读取权重文件。
        header = dict(seal, format="frozen-three-v1", policies=policies)
        weights = {"v1": "a" * 64, "v2": "b" * 64}
        valid = dict(header, checkpoints_sha256=weights)

        def read_batch(index):
            files["execution_complete.json"] = json.dumps(index).encode()
            with patch.object(Path, "read_bytes", lambda path: files[path.name]):
                return list(sealed_episodes(Path(args["execution_dir"]), ids, seal["test_delivery"]))

        rows = seal["episodes"]
        self.assertCountEqual([ep["trajectory_id"] for ep in read_batch(valid)],
                              [f"hidden-{name}-{i}" for i in range(10) for name in policies])
        for case, invalid in (
                ("missing trajectory", dict(valid, episodes=rows[:-1])),
                ("duplicate trajectory", dict(valid, episodes=rows + [rows[-1]])),
                ("missing metadata", header),
                ("missing v2", dict(header, checkpoints_sha256={"v1": weights["v1"]})),
                ("wrong policy key", dict(header, checkpoints_sha256={"v1": weights["v1"], "dqn": weights["v2"]})),
                ("malformed digest", dict(header, checkpoints_sha256=dict(weights, v2="g" * 64)))):
            with self.subTest(case=case), self.assertRaises(ValueError):
                read_batch(invalid)

    def test_unsealed_partial_or_wrong_delivery_rejected_before_opening_reports(self):
        args, _, seal = batch_inputs()
        for change in ({"all_episodes_complete": False}, {"episodes": seal["episodes"][:-1]}, {"mode": "fixture"},
                       {"test_delivery": {"source_id": "fake"}}, {"episodes": seal["episodes"] + seal["episodes"][:1]}):
            files = {"execution_complete.json": json.dumps(dict(seal, **change)).encode()}
            with self.subTest(change=list(change)), patch.object(Path, "read_bytes", lambda path: files[path.name]), self.assertRaises(ValueError):
                prepare_sealed_test_reviews(**args)

    def test_only_completed_failures_can_be_skipped_even_when_the_seal_hash_matches(self):
        args, files, seal = batch_inputs()
        row = seal['episodes'].pop(0)  # 将合法失败轨迹放在已准备好的九份终稿之后。
        seal['episodes'].append(row)
        ep = json.loads(files[row['trace']])
        for reason, done in [('usage_unknown', True), ('token_budget', True), ('max_steps', True),
                             (None, False), ('in_progress', True), ('max_steps', False)]:
            ep['terminal_reason'] = reason
            step = ep['transitions'][-1]
            step.update(done=done, terminal_reason=reason)
            step['state_after'].update(terminal=done, terminal_reason=reason)
            files[row['trace']] = json.dumps(ep).encode()
            row['sha256'] = sha(files[row['trace']])
            files['execution_complete.json'] = json.dumps(seal).encode()
            with self.subTest(reason=reason, done=done), patch.object(Path, 'read_bytes', lambda path: files[path.name]):
                if done and reason in ('usage_unknown', 'token_budget', 'max_steps'):
                    self.assertEqual(len(prepare_sealed_test_reviews(**args)), 9)
                else:
                    with self.assertRaises(ValueError):
                        prepare_sealed_test_reviews(**args)

    def test_sealed_final_requires_a_completed_successful_write(self):
        args, files, seal = batch_inputs()
        row = seal['episodes'][-1]
        ep = json.loads(files[row['trace']])
        last = ep['transitions'][-1]
        for transitions in ([], [dict(last, success=False)], [dict(last, action_id='retrieve_more')],
                            [dict(last, state_after={'terminal': False, 'terminal_reason': None})]):
            ep['transitions'] = transitions
            files[row['trace']] = json.dumps(ep).encode()
            row['sha256'] = sha(files[row['trace']])
            files['execution_complete.json'] = json.dumps(seal).encode()
            with self.subTest(transitions=transitions), patch.object(Path, 'read_bytes', lambda path: files[path.name]), self.assertRaises(ValueError):
                prepare_sealed_test_reviews(**args)

    def test_last_trace_tampering_and_swapped_identity_invalidate_whole_batch(self):
        args, files, seal = batch_inputs()
        row = seal["episodes"][-1]
        ep = json.loads(files[row["trace"]])
        ep["trajectory_id"] = "swapped"
        files[row["trace"]] = json.dumps(ep).encode()
        for digest_matches in (False, True):
            if digest_matches:
                row["sha256"] = sha(files[row["trace"]])
            files["execution_complete.json"] = json.dumps(seal).encode()
            with self.subTest(digest_matches=digest_matches), patch.object(Path, "read_bytes", lambda path: files[path.name]), self.assertRaises(ValueError):
                prepare_sealed_test_reviews(**args)


if __name__ == "__main__":
    unittest.main()
