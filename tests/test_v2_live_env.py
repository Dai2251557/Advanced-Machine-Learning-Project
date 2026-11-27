import unittest

from experiments.v2_benchmark import load_benchmark
from rl.v2_live_env import ACTION_ORDER, FEATURE_NAMES, V2LiveEnv
from tests.test_v2_benchmark import FakeBackend


class LiveEnvTests(unittest.TestCase):
    def test_shapes_masks_target_and_terminal_reward(self):
        corpus, sections, tasks = load_benchmark()
        judged = []
        def judge(result):
            judged.append(result)
            return .8
        env = V2LiveEnv(corpus, tasks, sections, FakeBackend(), judge)
        state, info = env.reset(seed=2, options={'task_index': 0})
        self.assertEqual(len(state), len(FEATURE_NAMES))
        self.assertEqual(len(info['action_mask']), len(ACTION_ORDER))
        self.assertEqual(info['action_mask'], [True, False, False, False, False, False])
        self.assertNotIn('checklist', env.runner.task)
        with self.assertRaises(ValueError):
            env.step('read_1')
        _, _, done, _, info = env.step('retrieve_more')
        self.assertFalse(done)
        target = info['candidate_ids'][1]
        _, reward, done, _, info = env.step('read_2')
        self.assertEqual(env.runner.opened[-1]['section_id'], target)
        self.assertEqual(info['cost_tokens'], 30)
        self.assertLess(reward, 0)
        self.assertFalse(judged)
        _, reward, done, truncated, info = env.step('write')
        self.assertTrue(done)
        self.assertFalse(truncated)
        self.assertAlmostEqual(reward, .8 - .1 * 30 / 4500)
        self.assertEqual(len(judged), 1)
        self.assertFalse(any(info['action_mask']))

    def test_requires_an_external_evaluator(self):
        corpus, sections, tasks = load_benchmark()
        with self.assertRaises(ValueError):
            V2LiveEnv(corpus, tasks, sections, FakeBackend(), None)


if __name__ == '__main__':
    unittest.main()
