import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

from agents.executor import MockBackend
from core.runner import Runner
from core.schema import ACTIONS, ReviewTask
from experiments.collect import json_sha256, paper_index
from experiments.frozen import execute_pairs
from litreview.clients import DeepSeekBackend
from rl.state_encoder import StateEncoder


class TestDeliveryTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        papers = [{'id': 'A', 'title': 'Title', 'abstract': 'Fixture evidence.'}]
        self.runner = Runner({'papers': papers, 'dataset_kind': 'arxiv_abstracts', 'purpose': 'formal_experiment'},
                             DeepSeekBackend(api_key='fixture', model='deepseek-flash'))
        self.tasks = [ReviewTask(f'test-{i}', f'Topic {i}', split='test') for i in range(6)]
        self.seal = {'source_id': 'sealed-delivery/fixture', 'config_version': 1,
                     'dataset_kind': 'arxiv_abstracts', 'data_purpose': 'formal_experiment',
                     'tasks': {t.review_id: {'split': 'test', 'task_sha256': json_sha256(t.to_dict())}
                               for t in self.tasks}, 'papers': paper_index(papers)}
        spec = StateEncoder().spec()
        self.checkpoint = self.root / 'fixture-policy.json'
        self.checkpoint.write_text(json.dumps({'encoder': spec, 'format': 'dense-relu-v1',
            'layers': [{'weight': [[0.] * len(spec['features'])], 'bias': [0.]}, {'weight': [[0.]], 'bias': [0.]},
                       {'weight': [[0.]] * len(ACTIONS), 'bias': [0., 0., 0., 1., 0.]}],
            'metadata': {'data_source': 'formal_train_runner', 'runner_config': asdict(self.runner.config),
                         'development_review_ids': ['train-0']}}))

    def run_pairs(self, tasks, seal):
        return execute_pairs(self.runner, tasks, self.checkpoint, self.root / 'results',
                             mode='formal', test_manifest=seal)

    def test_preflight_rejects_unsealed_or_modified_inputs_before_any_call(self):
        tasks = self.tasks
        bad = [
            ('no seal', tasks, None),
            ('uncollected training topic relabeled test', tasks[:-1] + [replace(tasks[-1], review_id='train-unused')], self.seal),
            ('modified task text', [replace(tasks[0], survey_instruction='Other topic'), *tasks[1:]], self.seal),
            ('missing sealed topic', tasks[:-1], self.seal),
            ('extra topic', tasks + [replace(tasks[0], review_id='extra')], self.seal),
            ('modified abstract', tasks, {**self.seal, 'papers': {'A': 'a' * 64}}),
            ('missing corpus member', tasks, {**self.seal, 'papers': {**self.seal['papers'], 'B': 'b' * 64}}),
            ('version mismatch', tasks, {**self.seal, 'config_version': 2}),
        ]
        with patch.object(DeepSeekBackend, 'complete', side_effect=AssertionError('no external calls')) as backend:
            for name, batch, seal in bad:
                with self.subTest(name=name), self.assertRaises(ValueError):
                    self.run_pairs(batch, seal)
                self.assertIsNone(self.runner.task)
                self.assertFalse((self.root / 'results').exists())
            backend.assert_not_called()

    def test_complete_seal_is_order_independent_and_saved(self):
        # 测试采用真实接口形状，传输边界完全替换，不发送数据。
        with patch.object(DeepSeekBackend, 'complete', side_effect=MockBackend().complete):
            result = self.run_pairs(reversed(self.tasks), self.seal)
        self.assertEqual(len(result['episodes']), 12)
        persisted = json.loads((self.root / 'results/execution_complete.json').read_text())
        self.assertEqual(persisted['test_delivery'], {'source_id': self.seal['source_id'],
                                                    'manifest_sha256': json_sha256(self.seal)})
