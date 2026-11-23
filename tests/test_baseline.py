import copy
import io
import json
import tempfile
import unittest
import urllib.error
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from litreview.__main__ import main
from litreview.clients import Completion, DeepSeekBackend, DemoBackend, NoRedirect
from litreview.pipeline import FixedRulePipeline, load_corpus


ROOT = Path(__file__).resolve().parents[1]
DEMO_PATH = ROOT / "examples" / "demo_papers.json"


class ControlledBackend(DemoBackend):
    def __init__(self, role=None, output=None, finish_reason="stop", tokens=False):
        self.role = role
        self.output = output
        self.finish_reason = finish_reason
        self.tokens = tokens

    def complete(self, role, instruction, payload):
        result = super().complete(role, instruction, payload)
        if role == self.role:
            result.content = self.output
            result.finish_reason = self.finish_reason
        if self.tokens:
            result.prompt_tokens = 10
            result.completion_tokens = 5
        return result


class CorpusTests(unittest.TestCase):
    def test_demo_is_explicitly_illustrative(self):
        corpus = load_corpus(DEMO_PATH)
        self.assertEqual(corpus["dataset_kind"], "illustrative")
        self.assertEqual(len(corpus["papers"]), 2)

    def test_reject_invalid_corpus(self):
        valid = load_corpus(DEMO_PATH)
        duplicate = copy.deepcopy(valid)
        duplicate["papers"].append(duplicate["papers"][0])
        blank = copy.deepcopy(valid)
        blank["papers"][0]["abstract"] = " "
        unsafe_id = copy.deepcopy(valid)
        unsafe_id["papers"][0]["id"] = "bad]\n# heading"
        for data in ([], {}, {"dataset_kind": "unknown", "papers": []}, duplicate, blank, unsafe_id):
            with self.subTest(data=data), tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "papers.json"
                path.write_text(json.dumps(data), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_corpus(path)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.corpus = load_corpus(DEMO_PATH)

    def test_end_to_end_has_four_roles_and_provenance(self):
        pipeline = FixedRulePipeline(DemoBackend())
        result = pipeline.run(self.corpus, "reinforcement learning scheduling", top_k=1)
        self.assertEqual(result["paper_ids"], ["demo-001"])
        self.assertEqual([e["role"] for e in pipeline.events],
                         ["retrieve", "summarize", "write", "verify"])
        self.assertTrue(all(e["status"] == "ok" for e in pipeline.events))
        self.assertIn("演示输出", result["report"])
        self.assertIn("[demo-001]", result["report"])
        self.assertIn(result["summaries"][0]["evidence"], self.corpus["papers"][0]["abstract"])
        self.assertIsNone(pipeline.trace()["usage"]["prompt_tokens"])
        self.assertIsNone(pipeline.trace()["quality_score"])

    def test_real_usage_is_aggregated_and_reset_for_next_run(self):
        pipeline = FixedRulePipeline(ControlledBackend(tokens=True))
        for _ in range(2):
            pipeline.run(self.corpus, "scheduling")
            self.assertEqual(len(pipeline.events), 4)
            self.assertEqual(pipeline.trace()["usage"], {"prompt_tokens": 40, "completion_tokens": 20})

    def test_invalid_agent_outputs_stop_before_later_stages(self):
        cases = [
            ("retrieve", {"paper_ids": []}),
            ("retrieve", {"paper_ids": ["nonexistent"]}),
            ("retrieve", {"paper_ids": ["demo-001", "demo-001"]}),
            ("retrieve", {"paper_ids": ["demo-001", "demo-002"]}),
            ("summarize", {"items": []}),
            ("summarize", {"items": [{"paper_id": "demo-001", "summary": "claim",
                                      "evidence": "invented evidence"}]}),
            ("write", {"sections": [{"heading": "h", "text": "text", "paper_ids": ["fake"]}]}),
            ("write", {"sections": []}),
            ("verify", {"approved": "true", "issues": []}),
            ("verify", {"approved": True, "issues": ["Unsupported claim"]}),
            ("verify", {"approved": False, "issues": ["Unsupported claim"]}),
        ]
        order = ["retrieve", "summarize", "write", "verify"]
        for role, output in cases:
            with self.subTest(role=role, output=output):
                pipeline = FixedRulePipeline(ControlledBackend(role, json.dumps(output)))
                with self.assertRaises(ValueError):
                    pipeline.run(self.corpus, "scheduling", top_k=1)
                self.assertEqual(len(pipeline.events), order.index(role) + 1 + (role == "summarize"))
                self.assertEqual(pipeline.events[-1]["status"], "error")
                self.assertEqual(pipeline.events[-1]["output"], output)

    def test_summary_case_error_is_repaired_once_and_both_calls_are_billed(self):
        class RepairingBackend(ControlledBackend):
            def __init__(self):
                super().__init__(tokens=True)
                self.summary_calls = 0
                self.feedback = None

            def complete(self, role, instruction, payload):
                result = super().complete(role, instruction, payload)
                if role == "summarize":
                    self.summary_calls += 1
                    if self.summary_calls == 1:
                        value = json.loads(result.content)
                        value["items"][0]["evidence"] = value["items"][0]["evidence"].lower()
                        result.content = json.dumps(value)
                    else:
                        self.feedback = payload["validation_feedback"]
                return result

        backend = RepairingBackend()
        pipeline = FixedRulePipeline(backend)
        result = pipeline.run(self.corpus, "scheduling", top_k=1)
        self.assertEqual(backend.summary_calls, 2)
        self.assertIn("demo-001", backend.feedback)
        attempts = [event for event in pipeline.events if event["role"] == "summarize"]
        self.assertEqual([event["attempt"] for event in attempts], [1, 2])
        self.assertEqual([event["status"] for event in attempts], ["error", "ok"])
        self.assertEqual(pipeline.trace()["usage"], {"prompt_tokens": 50, "completion_tokens": 25})
        self.assertEqual(result["summaries"][0]["evidence"], self.corpus["papers"][0]["abstract"])

    def test_summary_repair_is_bounded_and_does_not_publish_unmatched_evidence(self):
        wrong = json.dumps({"items": [{"paper_id": "demo-001", "summary": "Claim",
                                       "evidence": "Paraphrased evidence"}]})
        pipeline = FixedRulePipeline(ControlledBackend("summarize", wrong))
        with self.assertRaises(ValueError):
            pipeline.run(self.corpus, "scheduling", top_k=1)
        self.assertEqual([event["role"] for event in pipeline.events],
                         ["retrieve", "summarize", "summarize"])
        self.assertTrue(all(event["status"] == "error" for event in pipeline.events[1:]))

    def test_empty_malformed_and_truncated_response(self):
        for content, reason in [(" ", "stop"), ("{bad", "stop"), ("[]", "stop"),
                                ('{"paper_ids":["demo-001"]}', "length")]:
            with self.subTest(content=content, reason=reason):
                pipeline = FixedRulePipeline(ControlledBackend("retrieve", content, reason))
                with self.assertRaises(ValueError):
                    pipeline.run(self.corpus, "scheduling")
                self.assertEqual(pipeline.events[0]["status"], "error")

    def test_invalid_inputs_do_not_call_agents(self):
        for query, top_k in [(" ", 1), ("query", 0), ("query", -1)]:
            pipeline = FixedRulePipeline(DemoBackend())
            with self.assertRaises(ValueError):
                pipeline.run(self.corpus, query, top_k)
            self.assertEqual(pipeline.events, [])

    def test_summary_and_report_must_cover_selected_papers(self):
        partial_summary = {"items": [{"paper_id": "demo-001", "summary": "text",
                                      "evidence": self.corpus["papers"][0]["abstract"]}]}
        partial_draft = {"sections": [{"heading": "h", "text": "text", "paper_ids": ["demo-001"]}]}
        for role, output in [("summarize", partial_summary), ("write", partial_draft)]:
            pipeline = FixedRulePipeline(ControlledBackend(role, json.dumps(output)))
            with self.subTest(role=role), self.assertRaises(ValueError):
                pipeline.run(self.corpus, "reinforcement literature", top_k=2)


class DeepSeekTests(unittest.TestCase):
    def response(self, content='{"paper_ids":["demo-001"]}', usage=None, reason="stop"):
        return {"choices": [{"message": {"content": content}, "finish_reason": reason}],
                "usage": usage}

    def call_with_response(self, data):
        fake_response = io.BytesIO(json.dumps(data).encode("utf-8"))
        with patch("litreview.clients.urllib.request.build_opener") as build:
            build.return_value.open.return_value = fake_response
            result = DeepSeekBackend("test-key").complete("retrieve", "Return JSON.", {"query": "测试"})
            request = build.return_value.open.call_args.args[0]
            return result, request

    def test_request_matches_documented_endpoint_and_json_mode(self):
        result, request = self.call_with_response(
            self.response(usage={"prompt_tokens": 12, "completion_tokens": 8}))
        self.assertEqual(request.full_url, "https://api.deepseek.com/chat/completions")
        body = json.loads(request.data)
        self.assertEqual(body["model"], "deepseek-flash")
        self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertEqual(body["thinking"], {"type": "disabled"})
        self.assertFalse(body["stream"])
        self.assertEqual(json.loads(body["messages"][1]["content"]), {"query": "测试"})
        self.assertEqual((result.prompt_tokens, result.completion_tokens), (12, 8))
        self.assertNotIn("test-key", repr(DeepSeekBackend("test-key")))

    def test_missing_usage_stays_unknown(self):
        result, _ = self.call_with_response(self.response())
        self.assertIsNone(result.prompt_tokens)
        self.assertIsNone(result.completion_tokens)

    def test_invalid_response_schema_is_rejected(self):
        for data in ({}, {"choices": []}, self.response(content=None),
                     self.response(usage={"prompt_tokens": -1}),
                     self.response(usage={"completion_tokens": True})):
            with self.subTest(data=data), self.assertRaises(RuntimeError):
                self.call_with_response(data)

    def test_http_failure_does_not_expose_error_body_or_credentials(self):
        with patch("litreview.clients.urllib.request.build_opener") as build:
            build.return_value.open.side_effect = urllib.error.HTTPError(
                "https://api.deepseek.com", 401, "test-key", {}, None)
            with self.assertRaisesRegex(RuntimeError, "HTTP 401") as caught:
                DeepSeekBackend("test-key").complete("retrieve", "JSON", {})
            self.assertNotIn("test-key", str(caught.exception))

    def test_missing_key_and_unsafe_endpoints_are_rejected(self):
        with self.assertRaises(ValueError):
            DeepSeekBackend("")
        for url in ("http://example.com", "https://user:secret@example.com", "https://example.com?k=x"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                DeepSeekBackend("test-key", base_url=url)

    def test_redirect_does_not_forward_credentials(self):
        self.assertIsNone(NoRedirect().redirect_request(None, None, 302, "", {}, "https://elsewhere.test"))


class CliTests(unittest.TestCase):
    def args(self, output, query="scheduling"):
        return ["--corpus", str(DEMO_PATH), "--query", query, "--output", str(output)]

    def test_cli_success_outputs_and_prevents_overwriting(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            output = Path(folder) / "run"
            self.assertEqual(main(self.args(output)), 0)
            self.assertTrue((output / "report.md").is_file())
            self.assertTrue((output / "result.json").is_file())
            trace = json.loads((output / "trace.json").read_text(encoding="utf-8"))
            self.assertEqual(trace["status"], "success")
            self.assertEqual(trace["dataset_kind"], "illustrative")
            original = (output / "trace.json").read_bytes()
            self.assertEqual(main(self.args(output)), 1)
            self.assertEqual((output / "trace.json").read_bytes(), original)

    def test_cli_failure_preserves_trace_and_does_not_publish_report(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stderr(io.StringIO()):
            output = Path(folder) / "run"
            self.assertEqual(main(self.args(output, "zzzzunmatched")), 1)
            self.assertFalse((output / "report.md").exists())
            trace = json.loads((output / "trace.json").read_text(encoding="utf-8"))
            self.assertEqual(trace["status"], "error")
            self.assertEqual(trace["events"][0]["status"], "error")

    def test_missing_key_makes_no_network_call(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict("os.environ", {}, clear=True), \
                patch("litreview.clients.read_dotenv", return_value={}), \
                patch("litreview.clients.urllib.request.build_opener") as build, redirect_stderr(io.StringIO()):
            self.assertEqual(main(self.args(Path(folder) / "run") + ["--backend", "deepseek"]), 1)
            build.assert_not_called()


if __name__ == "__main__":
    unittest.main()
