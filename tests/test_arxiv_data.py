import copy
import hashlib
import io
import json
import tempfile
import unittest
import urllib.error
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from litreview.__main__ import main
from litreview.clients import DemoBackend
from litreview.import_arxiv import main as import_main, parse_feed, parse_identifier
from litreview.pipeline import load_corpus, load_task


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "agent_skill_papers.json"
TASKS = ROOT / "data" / "review_tasks.json"
IDS = ["2605.23904", "2606.03143", "2604.16778", "2610.21348", "2607.21361"]


def atom_feed(identifier="2605.23904v2", title="Test title", summary="Original abstract."):
    return (f'<feed xmlns="http://www.w3.org/2005/Atom"><entry>'
            f'<id>http://arxiv.org/abs/{identifier}</id><title>{title}</title>'
            f'<summary>{summary}</summary><author><name>Test Author</name></author>'
            '<published>2026-05-22T00:00:00Z</published><updated>2026-05-25T00:00:00Z</updated>'
            '<category term="cs.AI"/></entry></feed>').encode("utf-8")


class ArxivImportTests(unittest.TestCase):
    def test_ids_and_official_urls(self):
        for value in ("2605.23904v2", "https://arxiv.org/pdf/2605.23904v2",
                      "https://arxiv.org/pdf/2605.23904v2.pdf", "http://arxiv.org/abs/2605.23904v2"):
            self.assertEqual(parse_identifier(value), ("2605.23904", 2))
        self.assertEqual(parse_identifier("https://arxiv.org/pdf/2606.03143"), ("2606.03143", None))

    def test_rejects_invalid_ids_and_foreign_urls(self):
        for value in ("2613.12345", "2600.12345", "2605.23904v0", "2605.23904.extra",
                      "https://arxiv.org.attacker.test/pdf/2605.23904", "https://arxiv.org/help/2605.23904",
                      "https://arxiv.org/abs/2605.23904?x=y"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_identifier(value)

    def test_parse_normalizes_whitespace_and_pins_provenance(self):
        papers = parse_feed(atom_feed(title="Test\n title", summary="Original\n abstract."), IDS[:1])
        self.assertEqual(papers[0]["title"], "Test title")
        self.assertEqual(papers[0]["abstract"], "Original abstract.")
        self.assertEqual(papers[0]["version"], 2)
        self.assertEqual(papers[0]["source_url"], "https://arxiv.org/abs/2605.23904v2")
        self.assertEqual(papers[0]["abstract_sha256"], hashlib.sha256(b"Original abstract.").hexdigest())

    def test_incomplete_mismatched_or_invalid_feed_is_rejected(self):
        cases = [
            (b"not XML", IDS[:1]), (b"<html/>", IDS[:1]),
            (atom_feed(summary=""), IDS[:1]), (atom_feed(title=""), IDS[:1]),
            (atom_feed(), ["2606.03143"]), (atom_feed(), ["2605.23904v1"]),
            (atom_feed(identifier="2605.23904"), IDS[:1]),
            (atom_feed(), ["2605.23904", "2605.23904v2"]),
        ]
        for raw, requested in cases:
            with self.subTest(raw=raw, requested=requested), self.assertRaises(ValueError):
                parse_feed(raw, requested)

    def test_import_writes_snapshot_without_overwriting(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()), \
                redirect_stderr(io.StringIO()):
            target = Path(folder) / "papers.json"
            raw = atom_feed()
            with patch("litreview.import_arxiv.urllib.request.urlopen", return_value=io.BytesIO(raw)) as request:
                self.assertEqual(import_main([IDS[0], "--output", str(target)]), 0)
                self.assertIn("id_list=2605.23904", request.call_args.args[0].full_url)
            corpus = load_corpus(target)
            self.assertEqual(corpus["dataset_kind"], "arxiv_abstracts")
            self.assertEqual(corpus["raw_metadata_sha256"], hashlib.sha256(raw).hexdigest())
            self.assertEqual(target.with_suffix(".atom.xml").read_bytes(), raw)
            with patch("litreview.import_arxiv.urllib.request.urlopen") as request:
                self.assertEqual(import_main([IDS[0], "--output", str(target)]), 1)
                request.assert_not_called()

    def test_missing_record_or_network_failure_produces_no_corpus(self):
        for error in (urllib.error.URLError("test failure"), None):
            with self.subTest(error=error), tempfile.TemporaryDirectory() as folder, \
                    redirect_stderr(io.StringIO()):
                target = Path(folder) / "papers.json"
                with patch("litreview.import_arxiv.urllib.request.urlopen") as request:
                    if error:
                        request.side_effect = error
                    else:
                        request.return_value = io.BytesIO(atom_feed())
                    self.assertEqual(import_main([*IDS[:2], "--output", str(target)]), 1)
                self.assertFalse(target.exists())
                self.assertFalse(target.with_suffix(".atom.xml").exists())


class DatasetTests(unittest.TestCase):
    def test_all_five_user_papers_match_saved_official_response(self):
        corpus = load_corpus(CORPUS)
        self.assertEqual(corpus["dataset_kind"], "arxiv_abstracts")
        self.assertEqual(corpus["purpose"], "pilot_only")
        self.assertEqual([paper["id"] for paper in corpus["papers"]], IDS)
        raw = (CORPUS.parent / corpus["raw_metadata_file"]).read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), corpus["raw_metadata_sha256"])
        requested = [paper["requested_source"] for paper in corpus["papers"]]
        self.assertEqual(parse_feed(raw, requested), corpus["papers"])

    def test_all_five_tasks_have_valid_references_and_manual_checks(self):
        corpus = load_corpus(CORPUS)
        data = json.loads(TASKS.read_text(encoding="utf-8"))
        self.assertEqual(data["task_set_kind"], "pilot")
        self.assertEqual(len(data["tasks"]), 5)
        for task in data["tasks"]:
            self.assertEqual(load_task(TASKS, task["id"], corpus), task)
            self.assertTrue(task["evaluation_points"])
            self.assertEqual(len(task["expected_paper_ids"]), task["top_k"])

    def test_task_loader_rejects_invalid_or_unknown_tasks(self):
        corpus = load_corpus(CORPUS)
        valid = json.loads(TASKS.read_text(encoding="utf-8"))
        invalid_values = [[], {}, {"tasks": []}]
        for key, value in (("id", ""), ("query", ""), ("top_k", True), ("top_k", 0),
                           ("expected_paper_ids", ["unknown"])):
            data = copy.deepcopy(valid)
            data["tasks"][0][key] = value
            invalid_values.append(data)
        duplicate = copy.deepcopy(valid)
        duplicate["tasks"].append(copy.deepcopy(duplicate["tasks"][0]))
        invalid_values.append(duplicate)
        for data in invalid_values:
            with self.subTest(data=data), tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "tasks.json"
                path.write_text(json.dumps(data), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_task(path, "review-001", corpus)
        with self.assertRaisesRegex(ValueError, "not found"):
            load_task(TASKS, "unknown", corpus)

    def test_task_cli_uses_query_and_top_k_without_sending_checklist(self):
        class RecordingDemo(DemoBackend):
            def __init__(self):
                self.payloads = []

            def complete(self, role, instruction, payload):
                self.payloads.append(payload)
                return super().complete(role, instruction, payload)

        for override, expected_count in (([], 5), (["--top-k", "1"], 1)):
            with self.subTest(override=override), tempfile.TemporaryDirectory() as folder, \
                    redirect_stdout(io.StringIO()):
                backend = RecordingDemo()
                output = Path(folder) / "run"
                with patch("litreview.__main__.DemoBackend", return_value=backend):
                    code = main(["--corpus", str(CORPUS), "--task-file", str(TASKS),
                                 "--task-id", "review-001", "--output", str(output), *override])
                self.assertEqual(code, 0)
                result = json.loads((output / "result.json").read_text(encoding="utf-8"))
                self.assertEqual(len(result["paper_ids"]), expected_count)
                trace = json.loads((output / "trace.json").read_text(encoding="utf-8"))
                self.assertEqual(trace["task_id"], "review-001")
                self.assertEqual(trace["top_k"], expected_count)
                for payload in backend.payloads:
                    self.assertNotIn("evaluation_points", payload)
                    self.assertNotIn("expected_paper_ids", payload)

    def test_cli_rejects_partial_or_conflicting_task_options(self):
        options = [["--task-file", str(TASKS)], ["--query", "test", "--task-id", "review-001"],
                   ["--query", "test", "--task-file", str(TASKS), "--task-id", "review-001"]]
        for option in options:
            with self.subTest(option=option), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(["--corpus", str(CORPUS), "--output", "unused", *option])


if __name__ == "__main__":
    unittest.main()
