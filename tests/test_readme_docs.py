"""Offline integrity checks for showcase README.md and README_zh.md."""

import csv
import json
import re
import unittest
from pathlib import Path
from urllib.parse import unquote

from core.schema import ACTIONS

ROOT = Path(__file__).resolve().parents[1]
FENCE_RE = re.compile(r"^```.*?^```", re.MULTILINE | re.DOTALL)
LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def _strip_fenced_code(markdown: str) -> str:
    return FENCE_RE.sub("", markdown)


def _read_csv_by_key(path: Path, *keys: str) -> dict[tuple[str, ...], dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {tuple(row[k] for k in keys): row for row in csv.DictReader(handle)}


class ReadmeDocumentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.readme_en = (ROOT / "README.md").read_text(encoding="utf-8")
        cls.readme_zh = (ROOT / "README_zh.md").read_text(encoding="utf-8")
        cls.docs = {"README.md": cls.readme_en, "README_zh.md": cls.readme_zh}

    def test_bilingual_readmes_exist_and_cross_link(self):
        self.assertIn("(README_zh.md)", self.readme_en)
        self.assertIn("(README.md)", self.readme_zh)
        self.assertIn("Tongji University", self.readme_en)
        self.assertIn("同济大学", self.readme_zh)

    def test_all_relative_links_and_images_resolve(self):
        for name, content in self.docs.items():
            prose = _strip_fenced_code(content)
            targets = LINK_RE.findall(prose)
            self.assertGreaterEqual(len(targets), 20, f"{name} should link key reports, figures, and directories")
            for raw_target in targets:
                target = raw_target.strip()
                if target.startswith(("http://", "https://", "mailto:", "#")):
                    continue
                path_part = unquote(target.split("#", 1)[0])
                resolved = (ROOT / path_part).resolve()
                with self.subTest(doc=name, target=raw_target):
                    self.assertTrue(
                        resolved.is_relative_to(ROOT),
                        f"{name} link escapes repository root: {raw_target}",
                    )
                    self.assertTrue(
                        resolved.exists(),
                        f"{name} contains broken relative path: {raw_target} -> {resolved}",
                    )

    def test_headline_numbers_match_published_csvs_and_config(self):
        config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
        overview = _read_csv_by_key(ROOT / "results/final/report/overview.csv", "policy")
        utility = _read_csv_by_key(ROOT / "results/final/report/utility_summary.csv", "lambda", "policy")
        judge = _read_csv_by_key(ROOT / "results/final/report/judge_summary.csv", "policy")

        rule_tokens = float(overview[("rule",)]["mean_workflow_tokens"])
        v2_tokens = float(overview[("v2",)]["mean_workflow_tokens"])
        savings_pct = (1.0 - v2_tokens / rule_tokens) * 100.0
        total_workflow_tokens = sum(int(overview[(p,)]["workflow_tokens"]) for p in ("rule", "v1", "v2"))
        judge_tokens = int(judge[("all",)]["total_tokens"])

        expected_tokens = [
            f"{config['workflow_tokens_budget']:,}",
            f"{total_workflow_tokens:,}",
            f"{judge_tokens:,}",
            f"{savings_pct:.1f}%",
            "10/10",
        ]
        for policy in ("rule", "v1", "v2"):
            row = overview[(policy,)]
            expected_tokens.extend([
                f"{float(row['mean_overall_score']):.3f}",
                f"{float(row['mean_coverage']):.3f}",
                f"{float(row['mean_factuality']):.3f}",
                f"{float(row['mean_citation_support']):.3f}",
                f"{float(row['mean_consistency']):.3f}",
                f"{float(row['mean_workflow_tokens']):,.1f}",
                f"{float(row['mean_steps']):.1f}",
            ])
        for lam in ("0", "0.25", "0.5", "1"):
            for policy in ("rule", "v1", "v2"):
                expected_tokens.append(f"{float(utility[(lam, policy)]['mean_utility']):.3f}")

        for name, content in self.docs.items():
            for token in expected_tokens:
                with self.subTest(doc=name, token=token):
                    self.assertIn(token, content, f"{name} missing authoritative figure {token}")

    def test_contract_actions_and_offline_verification_commands_documented(self):
        required_snippets = (
            *ACTIONS,
            "arxiv600-review40-v1",
            "python -m unittest discover -s tests -p 'test_*.py' -v",
            "python -m utils.data_release",
            "python -m experiments.final.run_final --check",
            "python -m experiments.final.plot_final --out-dir outputs/redraw-001",
            "--dispatch",
            "results/final/figures/01_quality_by_task.png",
            "results/final/figures/02_cost_and_delivery.png",
            "results/final/figures/03_utility_by_lambda.png",
            "results/final/figures/04_v2_early_ablation.png",
        )
        for name, content in self.docs.items():
            for snippet in required_snippets:
                with self.subTest(doc=name, snippet=snippet):
                    self.assertIn(snippet, content)


if __name__ == "__main__":
    unittest.main()
