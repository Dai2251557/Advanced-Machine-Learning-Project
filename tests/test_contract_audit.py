"""Regression checks found while auditing role-1 against CONTRACT.md."""

import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from agents.executor import MockBackend
from core.runner import Runner
from core.rule_scheduler import RuleScheduler
from core.schema import ReviewTask, RunnerConfig
from litreview.clients import DeepSeekBackend, DemoBackend
from litreview.pipeline import load_corpus

ROOT = Path(__file__).resolve().parents[1]
TASK = ReviewTask("audit", "scheduling")


class AuditTests(unittest.TestCase):
    def corpus(self):
        return load_corpus(ROOT / "examples/demo_papers.json")

    def test_summary_limit_applies_after_failed_summary(self):
        class InvalidSummary(MockBackend):
            def complete(self, role, instruction, payload):
                result = super().complete(role, instruction, payload)
                if role == "summarize":
                    result.content = '{"items":[]}'
                return result
        config = RunnerConfig(max_summary_calls=1)
        runner = Runner(self.corpus(), InvalidSummary(), config)
        result = runner.run_episode(RuleScheduler(config), TASK)
        self.assertEqual(result["action_history"]["summarize_fast"]["call_count"], 1)
        self.assertTrue(result["success"])

    def test_unknown_backend_is_not_mislabelled_as_real(self):
        with self.assertRaises(ValueError):
            Runner(self.corpus(), DemoBackend())

    def test_extra_nonfinite_output_does_not_crash_trace_persistence(self):
        class ExtraNan(MockBackend):
            def complete(self, *args):
                result = super().complete(*args)
                value = json.loads(result.content)
                value["extra"] = float("nan")
                result.content = json.dumps(value)
                return result
        with tempfile.TemporaryDirectory() as folder:
            runner = Runner(self.corpus(), ExtraNan(), output_dir=folder)
            runner.reset(TASK)
            result = runner.step("retrieve_more")
            self.assertFalse(result["success"])
            self.assertTrue((runner.episode_dir / "episode.json").is_file())

    def test_http_status_is_logged_without_secrets(self):
        runner = Runner(self.corpus(), DeepSeekBackend("secret-api-key"))
        runner.reset(TASK)
        with patch("litreview.clients.urllib.request.build_opener") as build:
            build.return_value.open.side_effect = urllib.error.HTTPError(
                "https://api.deepseek.com", 429, "secret-api-key", {}, None)
            result = runner.step("retrieve_more")
        self.assertEqual(result["calls"][0]["http_status"], 429)
        self.assertNotIn("secret-api-key", json.dumps(result))

    def test_runner_routes_all_efforts_through_actual_transport(self):
        # Patch only the HTTP boundary: exercise the real Runner/executor/client stack.
        for summary in ("summarize_fast", "summarize_deep"):
            for write in ("write_fast", "write_deep"):
                bodies = []
                def respond(request, **kwargs):
                    body = json.loads(request.data)
                    bodies.append(body)
                    instruction = body["messages"][0]["content"]
                    payload = json.loads(body["messages"][1]["content"])
                    role = "verify" if '"valid_scores"' in instruction else "write" if '"sections"' in instruction else "summarize"
                    response = MockBackend().complete(role, instruction, payload)
                    return io.BytesIO(json.dumps({"model": "actual-version", "choices": [
                        {"message": {"content": response.content}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5}}).encode())
                runner = Runner(self.corpus(), DeepSeekBackend("test-key"))
                runner.reset(TASK)
                with patch("litreview.clients.urllib.request.build_opener") as build:
                    build.return_value.open.side_effect = respond
                    for action in ("retrieve_more", summary, write):
                        self.assertTrue(runner.step(action)["success"])
                expected = ["low", "high" if summary.endswith("deep") else "low", "low",
                            "high" if write.endswith("deep") else "low"]
                self.assertEqual([b["reasoning_effort"] for b in bodies], expected)
                self.assertTrue(all(b["thinking"] == {"type": "enabled"} for b in bodies))
                self.assertEqual(runner.result()["total_workflow_tokens"], 60)


if __name__ == "__main__":
    unittest.main()
