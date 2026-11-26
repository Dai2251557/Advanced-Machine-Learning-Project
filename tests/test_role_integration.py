"""Offline role-3 delivery -> role-1 Runner -> role-2 state/manifest checks."""

import hashlib
import unittest

from core.runner import Runner
from core.rule_scheduler import RuleScheduler
from core.schema import ReviewTask
from experiments.collect import collect_episode, json_sha256, paper_index, validate_episode
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

    def test_real_training_probe_handoff_is_bound_to_frozen_delivery(self):
        manifest = load_manifest('train')
        expected = {
            'R005': ('8cef4d21e3ab62c17a2c97097769fb2d7f6ae4124df0a64bd44e6d7751c73421', 11215),
            'R001': ('ac4ea4e4fddc146525a74758a83defbfdeefda6aa62b3ee892645bafa01b781f', 15290),
        }
        for review_id, (digest, tokens) in expected.items():
            with self.subTest(review_id=review_id):
                path = ROOT / 'handoff/role1_train_probes' / f'{review_id}-collection.json'
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
                episode = read_json(path)
                validate_episode(episode, mode='formal', manifest=manifest)
                self.assertEqual(episode['review_id'], review_id)
                self.assertTrue(episode['success'])
                self.assertEqual(episode['terminal_reason'], 'final_written')
                self.assertEqual(episode['total_workflow_tokens'], tokens)
                self.assertEqual(episode['retrieval']['spec']['method'], 'bilingual_bm25_v1')
                self.assertEqual([step['action_id'] for step in episode['transitions']],
                                 ['retrieve_more', 'summarize_fast', 'write_fast'])
                self.assertNotIn('judge_score', episode)


if __name__ == '__main__':
    unittest.main()
