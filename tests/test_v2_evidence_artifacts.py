"""Offline integrity checks for the real, train-only V2 evidence pilot."""

import json
import unittest
from pathlib import Path

from core.v2_evidence import load_pilot_sections
from evaluation.v2_sensitivity import build_request, score_response, sensitivity_check
from utils.data_release import load_corpus, read_json


ARTIFACTS = Path(__file__).resolve().parents[1] / 'handoff' / 'role1_v2_pilot'


class V2EvidenceArtifactTests(unittest.TestCase):
    def test_real_routes_and_provenance(self):
        sections = load_pilot_sections(corpus=load_corpus())
        expected = {
            'runner_baseline': (['retrieve_more', 'write'], 1004, False, False),
            'runner_no_fulltext': (['retrieve_more', 'rewrite_query', 'retrieve_more', 'write'], 1656, False, False),
            'runner_no_rewrite': (['retrieve_more', 'retrieve_more', 'open_section', 'write'], 1671, True, True),
            'runner_real': (['retrieve_more', 'rewrite_query', 'retrieve_more', 'open_section', 'write'], 2197, True, True),
        }
        for name, (actions, tokens, opened, contains_result) in expected.items():
            with self.subTest(route=name):
                result = read_json(ARTIFACTS / (name + '.json'))
                self.assertEqual(result['kind'], 'v2_train_pilot_not_formal')
                self.assertEqual(result['actions'], actions)
                self.assertEqual(result['total_workflow_tokens'], tokens)
                self.assertEqual(result['terminal_reason'], 'final_written')
                self.assertEqual(bool(result['opened_sections']), opened)
                self.assertEqual('HER' in result['final_text'] and '10' in result['final_text'], contains_result)
                self.assertEqual(sum((event['receipt']['prompt_tokens'] + event['receipt']['completion_tokens'])
                                     for event in result['events'] if event['receipt']), tokens)
                for event in result['events']:
                    if event['receipt']:
                        self.assertEqual(event['receipt']['model'], 'deepseek-flash')
                    if event['action'] == 'rewrite_query':
                        self.assertEqual(event['receipt']['request']['messages'][1]['content'].find('discussion-p7-her'), -1)
                    if event['action'] == 'write':
                        payload = json.loads(event['receipt']['request']['messages'][1]['content'])
                        self.assertEqual(bool(payload['opened_sections']), opened)
                if opened:
                    self.assertEqual(result['opened_sections'][0]['sha256'], sections[('2008.02646v1', 'discussion-p7-her')]['sha256'])

        broad = read_json(ARTIFACTS / 'rewrite_original_r022.json')
        self.assertEqual(broad['actions'], ['retrieve_more', 'rewrite_query', 'retrieve_more'])
        self.assertNotIn('2008.02646v1', broad['selected_paper_ids'])
        self.assertNotIn('1807.11174v1', broad['selected_paper_ids'])

    def test_judge_receipts_are_answer_bound_and_detect_omission(self):
        artifact = read_json(ARTIFACTS / 'judge_calibration.json')
        self.assertEqual(artifact['kind'], 'v2_train_pilot_quote_bound_judge_calibration')
        self.assertFalse(artifact['formal_release'])
        corpus = load_corpus()
        abstract = next(p['abstract'] for p in corpus['papers'] if p['id'] == '2008.02646v1')
        passage = load_pilot_sections(corpus=corpus)[('2008.02646v1', 'discussion-p7-her')]['text']
        self.assertEqual([row['condition'] for row in artifact['rows']], ['omitted', 'complete'])
        scores = []
        for row in artifact['rows']:
            self.assertEqual(row['request'], build_request(row['answer'], abstract, passage))
            score = score_response(row['response'], row['answer'])
            self.assertEqual(score, row['validated_score'])
            scores.append(score)
        self.assertTrue(sensitivity_check(*scores))
        self.assertEqual((scores[0]['coverage'], scores[1]['coverage']), (.5, 1.0))
        self.assertEqual(artifact['coverage_gap'], .5)


if __name__ == '__main__':
    unittest.main()
