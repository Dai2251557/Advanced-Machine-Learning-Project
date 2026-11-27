"""Verify the archived real probe can be replayed without API calls."""

import hashlib
import unittest
from pathlib import Path

from evaluation.judge import build_judge_request, finalize_judge_result, prepare_train_review
from experiments.feasibility import _episodes
from utils.data_release import load_corpus, load_manifest, read_json


ARTIFACTS = Path(__file__).resolve().parents[1] / 'handoff' / 'role1_feasibility'


class FeasibilityArtifactTests(unittest.TestCase):
    def test_trajectory_judge_and_source_bindings(self):
        manifest = load_manifest('train')
        corpus = load_corpus()
        plan = read_json(ARTIFACTS / 'plan.json')
        index = read_json(ARTIFACTS / 'execution_complete.json')
        scores = read_json(ARTIFACTS / 'scores.json')
        self.assertEqual(plan['task_ids'], ['R005', 'R001', 'R022'])
        self.assertEqual(plan['policies'], ['raw_fast', 'rule', 'retrieve_again', 'deep'])
        self.assertFalse(plan['fulltext_arm'])
        self.assertEqual(index['source_id'], manifest['source_id'])
        self.assertEqual(len(scores['rows']), 12)

        scored = {(row['review_id'], row['policy']): row for row in scores['rows']}
        self.assertEqual(len(scored), 12)
        episodes = _episodes(ARTIFACTS, manifest)
        self.assertEqual(len(episodes), 12)
        for row, episode in episodes:
            with self.subTest(task=row['review_id'], policy=row['policy']):
                self.assertTrue(episode['success'])
                self.assertEqual(episode['terminal_reason'], 'final_written')
                key = (row['review_id'], row['policy'])
                self.assertEqual({k: scored[key][k] for k in row}, row)
                prepared = prepare_train_review(episode=episode, frozen_papers=corpus['papers'], manifest=manifest)
                receipt = read_json(ARTIFACTS / 'judge_receipts' / f'{key[0]}--{key[1]}.json')
                self.assertEqual(receipt['request'], build_judge_request(prepared))
                verified = finalize_judge_result(prepared, receipt)
                label_name = 'trajectory-' + hashlib.sha256(row['trajectory_id'].encode()).hexdigest() + '.json'
                self.assertEqual(read_json(ARTIFACTS / 'labels' / label_name), verified)
                self.assertEqual(scored[key]['judge_overall'], verified['overall_score'])
                self.assertEqual(scored[key]['judge_subscores'], verified['subscores'])

        self.assertEqual(sum(row['total_workflow_tokens'] for row in scores['rows']), 196209)
        for task in plan['task_ids']:
            task_rows = [row for row in scores['rows'] if row['review_id'] == task]
            direct = next(row for row in task_rows if row['policy'] == 'raw_fast')
            self.assertEqual(direct['total_workflow_tokens'], min(row['total_workflow_tokens'] for row in task_rows))
            self.assertEqual(direct['judge_overall'], max(row['judge_overall'] for row in task_rows))

    def test_fulltext_diagnostic_is_separate_from_formal_labels(self):
        diagnostic = read_json(ARTIFACTS / 'fulltext_diagnostic_summary.json')
        self.assertEqual(diagnostic['kind'], 'nonformal_single_paper_information_gain_probe')
        self.assertIsNone(diagnostic['formal_v1_judge_score'])
        self.assertFalse(diagnostic['abstract_only']['sufficient'])
        self.assertTrue(diagnostic['abstract_plus_discussion']['sufficient'])


if __name__ == '__main__':
    unittest.main()
