import unittest
from pathlib import Path
from utils.data_release import read_json


class BenchmarkSmokeTests(unittest.TestCase):
    def test_real_receipts_and_costs(self):
        root = Path(__file__).resolve().parents[1] / 'handoff/v2_benchmark_smoke'
        for route, cost in [('direct', 1090), ('fixed_multi', 4275), ('gap_rule', 3649)]:
            result = read_json(root / (route + '.json'))
            self.assertEqual(result['terminal_reason'], 'final_written')
            self.assertEqual(result['total_workflow_tokens'], cost)
            receipts = [e['receipt'] for e in result['events'] if e['receipt']]
            self.assertEqual(sum(r['prompt_tokens'] + r['completion_tokens'] for r in receipts), cost)
            self.assertTrue(all(r['response']['model'] == 'deepseek-flash' for r in receipts))
            self.assertTrue(all(e['success'] for e in result['events']))
            self.assertEqual(len(result['opened_sections']), 0 if route == 'direct' else 2)


if __name__ == '__main__':
    unittest.main()
