import unittest

from core.schema import ACTIONS, RunnerConfig, verifier_value
from rl.reward import RewardConfig
from rl.synthetic_env import Outcome, SyntheticEnv, SyntheticOutcomes


class SyntheticTests(unittest.TestCase):
    def test_known_action_tradeoffs_and_return(self):
        def rollout(difficulty, plan):
            env = SyntheticEnv()
            env.reset(seed=4, options={'difficulty': difficulty})
            rewards, tokens = [], 0
            for action in plan:
                state, reward, done, truncated, info = env.step(action)
                step = info['transition']
                self.assertEqual(step['execution_kind'], 'simulated')
                self.assertFalse(truncated)
                tokens += step['total_workflow_tokens']
                rewards.append(reward)
            self.assertTrue(done)
            self.assertEqual(info['action_mask'], [False] * 5)
            self.assertAlmostEqual(sum(rewards), info['estimated_final_quality'] - tokens / 48000)
            return sum(rewards)
        for difficulty, best in [('easy', (1, 3)), ('hard', (2, 4))]:
            returns = {(s, w): rollout(difficulty, [ACTIONS[0], ACTIONS[s], ACTIONS[w]])
                       for s in (1, 2) for w in (3, 4)}
            self.assertEqual(max(returns, key=returns.get), best)

    def test_masks_deltas_and_invalidation(self):
        env = SyntheticEnv(corpus_size=5)
        _, initial = env.reset(seed=1)
        self.assertEqual(initial['action_mask'], [True] + [False] * 4)
        with self.assertRaises(ValueError):
            env.step('write_fast')
        env.step('retrieve_more')
        _, _, _, _, first = env.step('summarize_fast')
        self.assertIsNone(first['transition']['verifier_delta'])
        _, _, _, _, repeat = env.step('summarize_fast')
        self.assertEqual(repeat['transition']['verifier_delta'], 0)
        state, _, _, _, info = env.step('retrieve_more')
        self.assertFalse(info['action_mask'][0])
        self.assertFalse(state['summary_available'])
        self.assertIsNone(info['transition']['verifier_delta'])
        self.assertIsNone(env.step('write_fast')[-1]['transition']['verifier_delta'])
        with self.assertRaises(ValueError):
            env.step('retrieve_more')

    def test_partial_failure_retains_materials_and_stales_score(self):
        base = SyntheticOutcomes()
        env = SyntheticEnv(outcome_model=lambda s,a,r: Outcome(True, 10, 20, 30) if a.startswith('summarize') else base(s,a,r))
        env.reset(seed=1)
        env.step('retrieve_more')
        state, _, done, _, info = env.step('summarize_fast')
        self.assertTrue(state['summary_available'])
        self.assertIsNone(verifier_value(state))
        self.assertFalse(done or info['transition']['success'])
        self.assertEqual(info['transition']['total_workflow_tokens'], 60)

    def test_budget_priority_and_missing_measurement(self):
        for spend, reason in [(40, 'token_budget'), (None, 'usage_unknown'), (39, 'final_written')]:
            def draw(s, a, rng):
                return (Outcome(True, 0, 0, 10, {'coverage': .5}) if a == 'retrieve_more' else
                        Outcome(True, spend, 0, estimated_final_quality=.75, quality_method='fixture'))
            env = SyntheticEnv(config=RunnerConfig(max_steps=2, workflow_tokens_budget=50), outcome_model=draw)
            env.reset(seed=7)
            env.step('retrieve_more')
            state, reward, done, _, info = env.step('write_fast')
            self.assertTrue(done)
            self.assertEqual(state['terminal_reason'], reason)
            self.assertEqual(info['estimated_final_quality'], .75 if reason == 'final_written' else None)
            if spend is None:
                self.assertIsNone(state['workflow_tokens_used'])
                self.assertEqual(reward, -2)

    def test_missing_quality_requires_explicit_cost_only(self):
        def draw(s, a, rng):
            return Outcome(True, 0, 0, 3, {'coverage': .8}) if a == ACTIONS[0] else Outcome(True, 1, 2)
        for config in (RewardConfig(), RewardConfig(quality_weight=0)):
            env = SyntheticEnv(reward_config=config, outcome_model=draw)
            env.reset(seed=0)
            env.step(ACTIONS[0])
            if config.quality_weight:
                with self.assertRaisesRegex(ValueError, 'quality proxy'):
                    env.step('write_fast')
            else:
                self.assertAlmostEqual(env.step('write_fast')[1], -3 / 48000)
