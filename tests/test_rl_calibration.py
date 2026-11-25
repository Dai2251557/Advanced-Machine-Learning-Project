import json
import unittest

from agents.executor import MockBackend
from core.schema import ACTIONS, verifier_value
from experiments.collect import dry_run
from rl.calibration import evaluate_holdout, fit_calibration
from rl.reward import RewardConfig


class CalibrationTests(unittest.TestCase):
    def test_joint_sampling_and_partial_failure(self):
        class Variant(MockBackend):
            def __init__(self, cost, coverage, broken=False):
                self.cost, self.coverage, self.broken = cost, coverage, broken
            def complete(self, role, instruction, payload):
                c = super().complete(role, instruction, payload)
                c.prompt_tokens = self.cost
                if role == 'verify':
                    data = json.loads(c.content)
                    data['scores']['coverage'] = self.coverage
                    c.content = '{}' if self.broken and payload['items'] else json.dumps(data)
                return c
        eps = [dry_run('fast', review_id=f'r{i}', backend=Variant(n,q,b))
               for i,(n,q,b) in enumerate([(10,.6,False),(20,.9,True)])]
        model = fit_calibration(eps, mode='fixture')
        pairs = set()
        partial = seen_success = False
        with self.assertRaisesRegex(ValueError, 'summary/write_fast'):
            model.make_env()
        for seed in range(20):
            env = model.make_env(reward_config=RewardConfig(quality_weight=0))
            env.reset(seed=seed, options={'difficulty':'medium'})
            state, _, _, _, info = env.step('retrieve_more')
            step = info['transition']
            self.assertEqual((step['execution_kind'],step['source']), ('simulated','calibrated'))
            pairs.add((step['total_workflow_tokens'],verifier_value(state)))
            state, _, _, _, info = env.step('summarize_fast')
            if not info['transition']['success']:
                partial = True
                self.assertTrue(state['summary_available'])
                self.assertIsNone(verifier_value(state))
            else:
                seen_success = True
        self.assertTrue(partial and seen_success)
        self.assertEqual(pairs,{(10,.6),(20,.9)})
        self.assertEqual(model.manifest()['action_samples']['retrieve_more'],2)

    def test_missing_verifier_uses_only_fitting_labels_and_balances_topics(self):
        class BadVerifier(MockBackend):
            def complete(self, role, instruction, payload):
                c = super().complete(role, instruction, payload)
                c.prompt_tokens, c.completion_tokens = 10, 5
                if role == 'verify':
                    c.content = '{}'
                return c

        topics = [('A', .2), ('A', .2), ('B', .8), ('C', .95)]
        eps = [dry_run('raw_fast', review_id=rid, backend=BadVerifier()) for rid, _ in topics]
        labels = [{**{k: ep[k] for k in ('review_id', 'trajectory_id', 'split', 'final_text_sha256')},
                   'model': 'fixture', 'prompt_id': 'v1', 'source_kind': 'synthetic_fixture', 'overall_score': score}
                  for ep, (_, score) in zip(eps, topics)]
        for n, expected, method in [(1, .2, 'pooled_mean_low_support'), (3, .5, 'no_verifier_group_mean')]:
            model = fit_calibration(eps[:n], labels[:n], mode='fixture')
            env = model.make_env()
            env.reset(seed=5)
            state, _, _, _, _ = env.step('retrieve_more')
            self.assertIsNone(verifier_value(state))
            after, reward, done, _, info = env.step('write_fast')
            t = info['transition']
            self.assertTrue(done and t['success'])
            self.assertEqual(t['quality_method'], 'synthetic_fixture_' + method)
            self.assertAlmostEqual(t['estimated_final_quality'], expected)
            self.assertAlmostEqual(reward, expected - 15 / 48000)
            self.assertNotIn('overall_score', after)
        report = evaluate_holdout(eps, labels, mode='fixture', holdout_review_ids=['C'])
        self.assertEqual(report['quality']['compared'], 1)
        self.assertAlmostEqual(report['quality']['mae'], .95 - .5)
        # 真正没有质量依据时，默认训练须在开始前拒绝，显式成本诊断仍不捏造分数。
        model = fit_calibration(eps, mode='fixture')
        self.assertEqual(model.manifest()['unscorable_writes'], ['papers/write_fast'])
        with self.assertRaisesRegex(ValueError, 'collect labels'):
            model.make_env()
        env = model.make_env(reward_config=RewardConfig(quality_weight=0))
        env.reset(seed=5)
        env.step('retrieve_more')
        _, reward, done, _, info = env.step('write_fast')
        self.assertTrue(done)
        self.assertIsNone(info['estimated_final_quality'])
        self.assertAlmostEqual(reward, -15 / 48000)

    def test_zero_support_is_not_physical_illegality(self):
        ep = dry_run('raw_fast')
        with self.assertRaises(ValueError): fit_calibration([ep])
        with self.assertRaises(ValueError): fit_calibration([ep],mode='fixture',corpus_size=3)
        env = fit_calibration([ep],mode='fixture').make_env()
        env.reset(seed=3)
        _, _, _, _, info = env.step('retrieve_more')
        self.assertTrue(info['action_mask'][ACTIONS.index('write_deep')])
        state, _, done, _, info = env.step('write_deep')
        self.assertTrue(done)
        self.assertEqual(state['terminal_reason'],'usage_unknown')
        self.assertIsNone(info['transition']['total_workflow_tokens'])
        self.assertEqual(info['transition']['simulation_support']['samples'],0)

    def test_group_holdout_and_proxy_metrics(self):
        eps = [dry_run(p, review_id=rid) for rid in ('A','B','C') for p in ('rule','raw_deep')]
        def label(e):
            return {**{k:e[k] for k in ('review_id','trajectory_id','split','final_text_sha256')},
                    'model':'example','prompt_id':'v1','source_kind':'synthetic_fixture',
                    'overall_score': .95 if e['review_id']=='C' else .65}
        labels = list(map(label,eps))
        report = evaluate_holdout(eps,labels,mode='fixture',holdout_review_ids=['C'],repeats=2)
        self.assertEqual(report['fit_review_ids'],['A','B'])
        self.assertEqual(report['quality']['compared'],2)
        self.assertAlmostEqual(report['quality']['mae'],.3)
        self.assertAlmostEqual(report['quality']['train_mean_baseline_mae'],.3)
        self.assertEqual(report['observed_rule']['episodes'],1)
        self.assertEqual(report['simulated_rule']['episodes'],2)
        self.assertEqual(report['observed_rule']['completion_rate'],1)
        self.assertEqual(report['simulated_rule']['completion_rate'],1)
        with self.assertRaises(ValueError): evaluate_holdout(eps,labels,mode='fixture',holdout_review_ids=['Z'])
        model = fit_calibration(eps,mode='fixture')
        env = model.make_env()
        def run():
            env.reset(seed=0)
            env.step('retrieve_more')
            return env.step('write_deep')[-1]['transition']
        end = run()
        self.assertEqual(end,run())
        self.assertEqual(end['quality_method'],'fresh_valid_verifier_mean')
        self.assertEqual(end['estimated_final_quality'],.8)
        self.assertNotIn('overall_score',end['state_after'])
