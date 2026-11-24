import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from litreview.__main__ import main
from litreview.clients import DeepSeekBackend, DemoBackend
from litreview.config import read_dotenv


class DotEnvTests(unittest.TestCase):
    def test_loads_bom_comments_quotes_and_literal_values(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / ".env"
            path.write_text(
                '# comment\n\nexport DEEPSEEK_API_KEY = "test#key" # comment\n'
                "DEEPSEEK_MODEL='deepseek-flash'\n"
                "DEEPSEEK_BASE_URL=https://api.deepseek.com # comment\n"
                "LITERAL=${OTHER}\\n\nEMPTY= # comment\n", encoding="utf-8-sig")
            values = read_dotenv(path)
            self.assertEqual(values["DEEPSEEK_API_KEY"], "test#key")
            self.assertEqual(values["DEEPSEEK_MODEL"], "deepseek-flash")
            self.assertEqual(values["DEEPSEEK_BASE_URL"], "https://api.deepseek.com")
            self.assertEqual(values["LITERAL"], "${OTHER}\\n")
            self.assertEqual(values["EMPTY"], "")

    def test_missing_file_is_optional(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(read_dotenv(Path(folder) / ".env"), {})

    def test_malformed_lines_do_not_expose_values(self):
        for text in ("test-secret", 'DEEPSEEK_API_KEY="test-secret',
                     'DEEPSEEK_API_KEY="test-secret" extra'):
            with self.subTest(text=text), tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / ".env"
                path.write_text(text, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "line 1") as caught:
                    read_dotenv(path)
                self.assertNotIn("test-secret", str(caught.exception))

    def test_file_configuration_does_not_modify_process_environment(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {}, clear=True):
            path = Path(folder) / ".env"
            path.write_text("DEEPSEEK_API_KEY=file-test-key\nDEEPSEEK_MODEL=file-model\n", encoding="utf-8")
            backend = DeepSeekBackend.from_env(path)
            self.assertEqual(backend.api_key, "file-test-key")
            self.assertEqual(backend.model, "file-model")
            self.assertEqual(backend.base_url, "https://api.deepseek.com")
            self.assertNotIn("DEEPSEEK_API_KEY", os.environ)

    def test_environment_overrides_file(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {
            "DEEPSEEK_API_KEY": "environment-test-key", "DEEPSEEK_MODEL": "environment-model",
            "DEEPSEEK_BASE_URL": "https://gateway.example/v1",
        }, clear=True):
            path = Path(folder) / ".env"
            path.write_text("DEEPSEEK_API_KEY=file-test-key\nDEEPSEEK_MODEL=file-model\n", encoding="utf-8")
            backend = DeepSeekBackend.from_env(path)
            self.assertEqual(backend.api_key, "environment-test-key")
            self.assertEqual(backend.model, "environment-model")
            self.assertEqual(backend.base_url, "https://gateway.example/v1")

    def test_project_file_is_resolved_independently_of_working_directory(self):
        from litreview.config import PROJECT_ENV
        self.assertEqual(PROJECT_ENV, Path(__file__).resolve().parents[1] / ".env")
        with patch("litreview.clients.read_dotenv", return_value={"DEEPSEEK_API_KEY": "test-key"}) as read, \
                patch.dict(os.environ, {}, clear=True):
            DeepSeekBackend.from_env()
            read.assert_called_once_with(PROJECT_ENV)

    def test_cli_reads_dotenv_without_network_and_never_writes_key_to_artifacts(self):
        root = Path(__file__).resolve().parents[1]
        demo = DemoBackend()
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {}, clear=True):
            path = Path(folder) / ".env"
            path.write_text("DEEPSEEK_API_KEY=private-test-key\n", encoding="utf-8")
            output = Path(folder) / "run"
            stdout = io.StringIO()
            with patch("litreview.clients.PROJECT_ENV", path), \
                    patch.object(DeepSeekBackend, "complete", side_effect=demo.complete), redirect_stdout(stdout):
                code = main(["--backend", "deepseek", "--corpus", str(root / "examples/demo_papers.json"),
                             "--query", "scheduling", "--output", str(output)])
            self.assertEqual(code, 0)
            trace = json.loads((output / "trace.json").read_text(encoding="utf-8"))
            self.assertEqual(trace["backend"], "deepseek")
            self.assertNotIn("private-test-key", stdout.getvalue())
            for artifact in output.iterdir():
                self.assertNotIn("private-test-key", artifact.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
