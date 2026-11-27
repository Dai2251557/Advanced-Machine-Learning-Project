"""Integrity and route checks for the train-only multi-paper benchmark."""

import unittest

from core.v2_evidence import EvidenceRunner, normalized_pdf_text
from experiments.v2_benchmark import load_benchmark, route_actions


class FakeBackend:
    def complete(self, role, instruction, payload):
        if role == 'read_section':
            value = {'summary': '当前段落已经读取', 'evidence': payload['section']['text'][:30],
                     'supports_question': True}
        elif role == 'rewrite_query':
            value = {'gap': '需要具体实验依据', 'query': 'tactile robot experimental evidence'}
        else:
            papers = payload['papers']
            value = {'text': ' '.join('来源[' + p['id'] + ']' for p in papers),
                     'paper_ids': [p['id'] for p in papers]}
        return value, {'prompt_tokens': 20, 'completion_tokens': 10}


class V2BenchmarkTests(unittest.TestCase):
    def test_pdf_layout_normalization_does_not_invent_evidence(self):
        self.assertEqual(normalized_pdf_text('efﬁciency\n composer actu-\nates'), 'efficiency composer actuates')
        self.assertNotIn(normalized_pdf_text('improved 100 times'), normalized_pdf_text('improved 10 times'))

    def test_frozen_training_inventory_and_evidence(self):
        corpus, sections, tasks = load_benchmark()
        self.assertEqual((len(corpus['papers']), len(sections), len(tasks)), (8, 24, 12))
        self.assertEqual({t['kind'] for t in tasks}, {'abstract', 'section', 'cross_paper'})
        for kind in ('abstract', 'section', 'cross_paper'):
            self.assertEqual(sum(t['kind'] == kind for t in tasks), 4)
        self.assertTrue(all(t['split'] == 'train' and t['checklist'] for t in tasks))

    def test_opening_cost_is_measured_and_routes_terminate(self):
        corpus, sections, tasks = load_benchmark()
        direct = EvidenceRunner(corpus, tasks[0], sections, FakeBackend(), batch_size=2,
                                token_budget=4500, read_on_open=True)
        direct_result = route_actions(direct, 'direct')
        self.assertEqual(direct_result['actions'], ['retrieve_more', 'write'])
        self.assertEqual(direct_result['total_workflow_tokens'], 30)
        fixed = EvidenceRunner(corpus, tasks[4], sections, FakeBackend(), batch_size=2,
                               token_budget=4500, read_on_open=True)
        fixed_result = route_actions(fixed, 'fixed_multi')
        self.assertEqual(fixed_result['terminal_reason'], 'final_written')
        self.assertIn('open_section', fixed_result['actions'])
        self.assertGreater(fixed_result['total_workflow_tokens'], direct_result['total_workflow_tokens'])
        for event in fixed_result['events']:
            if event['action'] == 'open_section':
                self.assertEqual(event['receipt']['prompt_tokens'], 20)
                self.assertIn('evidence', event['detail']['reading'])


if __name__ == '__main__':
    unittest.main()
