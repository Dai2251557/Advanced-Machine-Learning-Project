"""Offline role-3 delivery -> role-1 Runner -> role-2 state/manifest checks."""

import unittest

from core.runner import Runner
from core.rule_scheduler import RuleScheduler
from core.schema import ReviewTask
from experiments.collect import collect_episode, json_sha256, paper_index
from rl.state_encoder import StateEncoder
from utils.data_release import ROOT, load_corpus, load_manifest, read_json


class RoleIntegrationTests(unittest.TestCase):
    def test_public_training_delivery_matches_sampling_manifest(self):
        corpus = load_corpus()
        manifest = load_manifest('train')
        self.assertEqual(paper_index(corpus['papers']), manifest['papers'])
        tasks = read_json(ROOT / 'data/train_tasks.json')
        self.assertEqual(len(tasks), 30)
        for row in tasks:
            task = ReviewTask.from_dict(row)
            self.assertEqual(manifest['tasks'][task.review_id],
                             {'split': 'train', 'task_sha256': json_sha256(task.to_dict())})

    def test_runner_encoder_and_rule_can_use_training_delivery_offline(self):
        runner = Runner(load_corpus())  # Explicitly offline mock execution.
        task = ReviewTask.from_dict(read_json(ROOT / 'data/train_tasks.json')[0])
        state = runner.reset(task)
        encoder = StateEncoder()
        self.assertEqual(len(encoder.encode(state, runner.valid_actions())), 45)
        result = runner.run_episode(RuleScheduler(runner.config), task)
        self.assertTrue(result['success'])
        self.assertEqual(result['execution_kind'], 'mock')
        self.assertEqual(result['total_workflow_tokens'], 0)
        for step in result['transitions']:
            self.assertEqual(len(encoder.encode(step['state_before'], step['action_mask'])), 45)

    def test_formal_collection_rejects_mock_even_with_valid_delivery(self):
        runner = Runner(load_corpus())
        task = ReviewTask.from_dict(read_json(ROOT / 'data/train_tasks.json')[0])
        with self.assertRaises(ValueError):
            collect_episode(runner, task, mode='formal', manifest=load_manifest('train'))
        self.assertIsNone(runner.task)


if __name__ == '__main__':
    unittest.main()
