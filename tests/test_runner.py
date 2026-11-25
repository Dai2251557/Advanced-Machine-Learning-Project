import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from agents.executor import MockBackend
from core.runner import Runner
from core.__main__ import main
from core.rule_scheduler import RuleScheduler
from core.schema import ACTIONS, ReviewTask, RunnerConfig, comparable_delta, verifier_value
from litreview.clients import DeepSeekBackend
from litreview.pipeline import load_corpus

ROOT = Path(__file__).resolve().parents[1]
TASK = ReviewTask("R001", "reinforcement learning scheduling")


class MeteredBackend(MockBackend):
    def complete(self, role, instruction, payload):
        result = super().complete(role, instruction, payload)
        result.prompt_tokens, result.completion_tokens = 10, 5
        return result


class RunnerTests(unittest.TestCase):
    def runner(self, backend=None, **config):
        return Runner(load_corpus(ROOT / "examples/demo_papers.json"), backend,
                      RunnerConfig(**config))

    def test_frozen_action_order_and_initial_mask(self):
        self.assertEqual(ACTIONS, ("retrieve_more", "summarize_fast", "summarize_deep", "write_fast", "write_deep"))
        r = self.runner()
        with self.assertRaises(RuntimeError):
            r.observe()
        state = r.reset(TASK)
        self.assertEqual(r.valid_actions(), [True, False, False, False, False])
        self.assertIsNone(verifier_value(state))
        with self.assertRaises(ValueError):
            r.step("write_fast")
        self.assertEqual(r.observe()["step_count"], 0)

    def test_each_action_and_terminal_write_has_no_verifier(self):
        for summary in ACTIONS[1:3]:
            for write in ACTIONS[3:]:
                with self.subTest(summary=summary, write=write):
                    r = self.runner(MeteredBackend())
                    r.reset(TASK)
                    retrieval = r.step("retrieve_more")
                    self.assertEqual(retrieval["total_workflow_tokens"], 15)
                    self.assertFalse(retrieval["state_after"]["verifier"]["valid_scores"]["factuality"])
                    t = r.step(summary)
                    self.assertIsNone(t["verifier_delta"])
                    self.assertEqual((t["input_tokens"], t["output_tokens"], t["verifier_tokens"]), (10, 5, 15))
                    self.assertEqual(t["state_before"]["action_history"][summary]["call_count"], 0)
                    repeated = r.step(summary)
                    self.assertEqual(repeated["verifier_delta"], 0.0)
                    self.assertEqual(repeated["state_after"]["action_history"][summary]["consecutive_low_gain_or_failures"], 1)
                    final = r.step(write)
                    self.assertTrue(final["done"])
                    self.assertTrue(final["success"])
                    self.assertEqual([c["role"] for c in final["calls"]], ["write"])
                    self.assertIsNone(final["verifier_delta"])
                    self.assertEqual(r.result()["total_workflow_tokens"], 90)
                    self.assertAlmostEqual(sum(h["token_share"] for h in r.observe()["action_history"].values()), 1)
                    self.assertEqual(r.valid_actions(), [False] * 5)
                    with self.assertRaises(ValueError):
                        r.step(write)

    def test_low_scores_do_not_mask_write_and_raw_sources_can_be_written(self):
        r = self.runner()
        r.reset(TASK)
        r.step("retrieve_more")
        self.assertTrue(all(r.valid_actions()[1:]))
        self.assertTrue(r.step("write_deep")["success"])

    def test_snapshot_and_return_values_do_not_mutate_runner(self):
        r = self.runner()
        state = r.reset(TASK)
        state["action_history"][ACTIONS[0]]["call_count"] = 999
        t = r.step("retrieve_more")
        t["state_after"]["terminal"] = True
        r.result()["transitions"].clear()
        self.assertFalse(r.observe()["terminal"])
        self.assertEqual(len(r.transitions), 1)
        self.assertEqual(r.observe()["action_history"][ACTIONS[0]]["call_count"], 1)

    def test_retrieval_invalidates_existing_summary(self):
        r = self.runner(retrieval_batch_size=1)
        r.reset(TASK)
        r.step("retrieve_more")
        r.step("summarize_fast")
        t = r.step("retrieve_more")
        self.assertFalse(t["state_after"]["summary_available"])
        self.assertIsNone(t["verifier_delta"])
        self.assertFalse(r.valid_actions()[0])

    def test_stale_verifier_and_dimension_changes_have_no_delta(self):
        r = self.runner()
        a = r.reset(TASK)
        b = r.step("retrieve_more")["state_after"]
        self.assertIsNone(comparable_delta(a, b))
        stale = copy.deepcopy(b)
        stale["material_revision"] += 1
        self.assertIsNone(verifier_value(stale))
        self.assertIsNone(comparable_delta(stale, b))

    def test_budget_and_steps_terminate(self):
        for kwargs, reason in (({"workflow_tokens_budget": 15}, "token_budget"), ({"max_steps": 1}, "max_steps")):
            r = self.runner(MeteredBackend(), **kwargs)
            r.reset(TASK)
            t = r.step("retrieve_more")
            self.assertTrue(t["done"])
            self.assertEqual(t["terminal_reason"], reason)
            self.assertFalse(r.result()["success"])

    def test_transport_failure_terminates_with_unknown_usage_and_sanitized_error(self):
        class Broken(MockBackend):
            def complete(self, *args):
                raise RuntimeError("secret-api-key")
        r = self.runner(Broken())
        r.reset(TASK)
        t = r.step("retrieve_more")
        self.assertFalse(t["success"])
        self.assertEqual(t["terminal_reason"], "usage_unknown")
        self.assertIsNone(t["total_workflow_tokens"])
        self.assertNotIn("secret-api-key", json.dumps(r.result()))

    def test_output_repair_costs_and_missing_usage(self):
        class Repair(MeteredBackend):
            broken = False
            def complete(self, role, instruction, payload):
                result = super().complete(role, instruction, payload)
                if role == "summarize" and not self.broken:
                    self.broken = True
                    result.content = '{"items":[]}'
                return result
        r = self.runner(Repair())
        r.reset(TASK)
        r.step("retrieve_more")
        t = r.step("summarize_fast")
        self.assertTrue(t["success"])
        self.assertEqual(t["total_workflow_tokens"], 45)
        self.assertEqual([c["status"] for c in t["calls"]], ["error", "ok", "ok"])

    def test_failed_verifier_retains_material_but_marks_scores_stale(self):
        class InvalidVerifier(MeteredBackend):
            def complete(self, role, instruction, payload):
                result = super().complete(role, instruction, payload)
                if role == "verify" and payload["items"]:
                    result.content = '{"next_action":"write_fast"}'
                return result
        r = self.runner(InvalidVerifier())
        r.reset(TASK)
        r.step("retrieve_more")
        t = r.step("summarize_fast")
        self.assertFalse(t["success"])
        self.assertTrue(t["state_after"]["summary_available"])
        self.assertIsNone(verifier_value(t["state_after"]))
        self.assertIsNone(t["verifier_delta"])

    def test_repair_does_not_call_again_after_budget(self):
        class Invalid(MeteredBackend):
            def complete(self, *args):
                result = super().complete(*args)
                result.content = '{}'
                return result
        r = self.runner(Invalid(), workflow_tokens_budget=15)
        r.reset(TASK)
        t = r.step("retrieve_more")
        self.assertEqual(len(t["calls"]), 1)
        self.assertEqual(t["terminal_reason"], "token_budget")

    def test_cache_is_per_episode_and_hits_cost_zero(self):
        r = self.runner(MeteredBackend(), cache_enabled=True)
        r.reset(TASK)
        r.step("retrieve_more")
        r.step("summarize_fast")
        r.step("summarize_fast")
        t = r.step("summarize_fast")
        self.assertTrue(all(c["cache_hit"] for c in t["calls"]))
        self.assertEqual(t["total_workflow_tokens"], 0)
        r.reset(TASK)
        self.assertFalse(r.step("retrieve_more")["calls"][0]["cache_hit"])

    def test_rule_episode_and_artifacts_are_reproducible_and_label_free(self):
        with tempfile.TemporaryDirectory() as folder:
            r = self.runner()
            r.output_dir = Path(folder)
            task = {**TASK.to_dict(), "gold_paper_ids": ["secret-gold"], "judge_score": 0.9}
            result = r.run_episode(RuleScheduler(), task, seed=4)
            self.assertTrue(result["success"])
            self.assertEqual(result["step_count"], 3)
            self.assertEqual(result["execution_kind"], "mock")
            self.assertNotIn("secret-gold", json.dumps(result))
            self.assertNotIn("judge_score", json.dumps(result))
            self.assertTrue(Path(result["final_text_ref"]).is_file())
            for t in result["transitions"]:
                for side in ("before", "after"):
                    self.assertEqual(json.loads(Path(t[f"state_{side}_ref"]).read_text(encoding="utf-8")), t[f"state_{side}"])

    def test_another_policy_uses_the_same_runner(self):
        class Policy:
            def reset(self, seed=None):
                self.actions = iter(("retrieve_more", "summarize_deep", "write_deep"))
            def select_action(self, state, action_mask):
                return next(self.actions)
        self.assertTrue(self.runner().run_episode(Policy(), TASK)["success"])

    def test_missing_usage_is_unknown_not_zero(self):
        class MissingUsage(MockBackend):
            def complete(self, *args):
                result = super().complete(*args)
                result.prompt_tokens = None
                return result
        r = self.runner(MissingUsage())
        r.reset(TASK)
        t = r.step("retrieve_more")
        self.assertEqual(t["terminal_reason"], "usage_unknown")
        self.assertIsNone(t["state_after"]["workflow_tokens_used"])
        self.assertTrue(all(h["token_share"] is None for h in t["state_after"]["action_history"].values()))

    def test_summary_reaching_budget_does_not_start_verifier(self):
        r = self.runner(MeteredBackend(), workflow_tokens_budget=30)
        r.reset(TASK)
        r.step("retrieve_more")
        t = r.step("summarize_fast")
        self.assertEqual([c["role"] for c in t["calls"]], ["summarize"])
        self.assertEqual(t["terminal_reason"], "token_budget")
        self.assertFalse(t["success"])
        self.assertIsNone(verifier_value(t["state_after"]))

    def test_invalid_score_and_fabricated_evidence_are_rejected(self):
        for broken_role in ("summarize", "verify"):
            class Bad(MeteredBackend):
                def complete(self, role, instruction, payload):
                    result = super().complete(role, instruction, payload)
                    if role == broken_role:
                        value = json.loads(result.content)
                        if role == "verify":
                            value["scores"]["coverage"] = float("nan")
                        else:
                            value["items"][0]["evidence"] = "fabricated-evidence"
                        result.content = json.dumps(value)
                    return result
            r = self.runner(Bad())
            r.reset(TASK)
            t = r.step("retrieve_more")
            if broken_role == "summarize":
                t = r.step("summarize_fast")
                self.assertFalse(r.observe()["summary_available"])
            self.assertFalse(t["success"])
            self.assertEqual(len(t["calls"]), 2)

    def test_cli_handoff_example_runs_without_network(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()), \
                patch("litreview.clients.urllib.request.build_opener") as network:
            self.assertEqual(main(["--backend", "mock", "--corpus", str(ROOT / "data/samples/agent_skill/agent_skill_papers.json"),
                                   "--task", str(ROOT / "examples/runner_task.json"), "--output", folder]), 0)
            network.assert_not_called()
            files = list(Path(folder).glob("*/episode.json"))
            self.assertEqual(len(files), 1)
            result = json.loads(files[0].read_text(encoding="utf-8"))
            self.assertTrue(result["success"])
            self.assertEqual(result["data_purpose"], "pilot_only")
            self.assertEqual(result["split"], "pilot")
            self.assertIn("PILOT", result["final_text"])

    def test_pilot_corpus_cannot_be_used_for_formal_splits(self):
        corpus = load_corpus(ROOT / "data/samples/agent_skill/agent_skill_papers.json")
        with tempfile.TemporaryDirectory() as folder:
            runner = Runner(corpus, output_dir=folder)
            for split in ("train", "validation", "test"):
                with self.subTest(split=split), self.assertRaisesRegex(ValueError, "pilot_only"):
                    runner.reset(replace(TASK, split=split))
            self.assertEqual(list(Path(folder).iterdir()), [])
            self.assertIsNone(runner.task)
            self.assertEqual(runner.reset(replace(TASK, split="pilot"))["review_id"], TASK.review_id)

    def test_formal_split_is_not_blocked_for_other_corpora(self):
        corpus = load_corpus(ROOT / "examples/demo_papers.json")
        corpus["purpose"] = "formal_experiment"
        runner = Runner(corpus)
        for split in ("train", "validation", "test"):
            runner.reset(replace(TASK, split=split))
            self.assertEqual(runner.result()["split"], split)
            self.assertEqual(runner.result()["data_purpose"], "formal_experiment")


class TransportAndSchemaTests(unittest.TestCase):
    def test_effort_request_and_actual_model(self):
        for effort in ("low", "high"):
            response = {"model": "actual-model-version", "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
            with patch("litreview.clients.urllib.request.build_opener") as build:
                build.return_value.open.return_value = io.BytesIO(json.dumps(response).encode())
                result = DeepSeekBackend("test-key", reasoning_effort=effort).complete("summarize", "JSON", {})
                body = json.loads(build.return_value.open.call_args.args[0].data)
                self.assertEqual(body["reasoning_effort"], effort)
                self.assertEqual(body["thinking"], {"type": "enabled"})
                self.assertNotIn("temperature", body)
                self.assertEqual(result.model, "actual-model-version")

    def test_shared_runner_rejects_uncontracted_model_or_gateway(self):
        corpus = load_corpus(ROOT / "examples/demo_papers.json")
        for backend in (DeepSeekBackend("x", model="other"), DeepSeekBackend("x", base_url="https://example.com")):
            with self.assertRaises(ValueError):
                Runner(corpus, backend)

    def test_invalid_config_and_tasks(self):
        for value in ({"max_steps": 0}, {"max_steps": True}, {"output_repair_attempts": -1},
                      {"quality_threshold": float("nan")}, {"config_version": 2}):
            with self.assertRaises(ValueError):
                RunnerConfig(**value)
        with self.assertRaises(ValueError):
            replace(TASK, difficulty="unknown")


if __name__ == "__main__":
    unittest.main()
