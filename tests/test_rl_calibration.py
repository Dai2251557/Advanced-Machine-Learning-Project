import json
import unittest

from agents.executor import MockBackend
from core.schema import ACTIONS, verifier_value
from experiments.collect import dry_run
from rl.calibration import evaluate_holdout, fit_calibration


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
        for seed in range(20):
            env = model.make_env()
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
