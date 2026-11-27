"""Real-probe orchestration boundaries, using offline transport replacements."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from agents.executor import MockBackend
from experiments.feasibility import _sha, judge_matrix, preflight, run_matrix
from litreview.clients import DeepSeekBackend
from utils.data_release import ROOT, read_json


class FeasibilityTests(unittest.TestCase):
    def test_only_frozen_training_tasks_and_predeclared_routes(self):
        tasks, _, manifest, _ = preflight(('R005', 'R001'), ('raw_fast', 'rule'))
        self.assertEqual(tasks['R005'].split, 'train')
        self.assertIn('R005', manifest['tasks'])
        for ids, policies in [(('T001',), ('rule',)), (('R005', 'R005'), ('rule',)),
                              (('R005',), ('oracle',))]:
            with self.subTest(ids=ids, policies=policies), self.assertRaises(ValueError):
                preflight(ids, policies)

    def test_run_then_judge_uses_complete_immutable_trajectories(self):
        requests = []

        def respond(request, **kwargs):
            body = json.loads(request.data)
            instruction = body['messages'][0]['content']
            payload = json.loads(body['messages'][1]['content'])
            role = 'verify' if '"valid_scores"' in instruction else 'write' if '"sections"' in instruction else 'summarize'
            completion = MockBackend().complete(role, instruction, payload)
            return io.BytesIO(json.dumps({'model': 'test-transport', 'choices': [
                {'message': {'content': completion.content}, 'finish_reason': 'stop'}],
                'usage': {'prompt_tokens': 10, 'completion_tokens': 5}}).encode())

        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()), \
                patch('litreview.clients.urllib.request.build_opener') as opener:
            opener.return_value.open.side_effect = respond
            output = Path(folder) / 'matrix'
            self.assertTrue(run_matrix(output, ('R005',), ('raw_fast', 'rule'), DeepSeekBackend('test-key')))
            index = read_json(output / 'execution_complete.json')
            self.assertEqual(len(index['rows']), 2)
            self.assertFalse((output / 'scores.json').exists())

            def judge(body):
                requests.append(body)
                return {'model': 'deepseek-v4-pro', 'choices': [{'finish_reason': 'stop', 'message': {
                    'content': json.dumps({'coverage': .8, 'factuality': .8, 'citation_support': .8,
                                           'consistency': .8, 'overall': .8, 'rationale': '离线协议测试。'})}}]}

            rows = judge_matrix(output, sender=judge)
            self.assertEqual(len(rows), 2)
            self.assertEqual(len(requests), 2)
            self.assertEqual(len(list((output / 'labels').glob('*.json'))), 2)
            self.assertTrue(all('policy' not in request['messages'][1]['content'] for request in requests))
            self.assertTrue(all(row['judge_overall'] == .8 for row in rows))

            path = output / index['rows'][0]['path']
            path.write_bytes(path.read_bytes() + b' ')
            self.assertNotEqual(_sha(path.read_bytes()), index['rows'][0]['sha256'])
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                judge_matrix(output, sender=judge)
            self.assertEqual(len(requests), 2)


if __name__ == '__main__':
    unittest.main()
