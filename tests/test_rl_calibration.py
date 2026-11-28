import json
import random
import unittest

from agents.executor import MockBackend
from agents.retrieval import BilingualRetriever
from core.schema import ACTIONS, verifier_value
from experiments.collect import dry_run, json_sha256
from rl.calibration import evaluate_holdout, fit_calibration
from rl.reward import RewardConfig
from tests.formal_fixture import synthetic_training_pair


class CalibrationTests(unittest.TestCase):
    def test_formal_fit_binds_episodes_to_the_complete_retrieval_spec(self):
        eps, manifest = synthetic_training_pair()
        spec = eps[1]['retrieval']['spec']
        model = fit_calibration(eps, manifest=manifest)
        self.assertEqual(model.manifest()['retrieval_spec_sha256'], json_sha256(BilingualRetriever.spec()))
        for change in ({'method': 'other'}, {'lexicon_sha256': 'f' * 64}, {'b': .7}, {'title_repetitions': 3}):
            bad = {**eps[1], 'retrieval': {**eps[1]['retrieval'], 'spec': {**spec, **change}}}
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'retrieval.spec'):
                fit_calibration([eps[0], bad], manifest=manifest)
        for retrieval in (None, {}, {'spec': {'method': spec['method']}}):
            with self.subTest(retrieval=retrieval), self.assertRaisesRegex(ValueError, 'retrieval.spec'):
                fit_calibration([{**eps[0], 'retrieval': retrieval}], manifest=manifest)

    def test_fixture_does_not_need_formal_retrieval_provenance(self):
        ep = dry_run('rule')
        del ep['retrieval']
        self.assertNotIn('retrieval_spec_sha256', fit_calibration([ep], mode='fixture').manifest())

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

    def test_cost_uses_same_paper_count_before_borrowing_a_nearby_size(self):
        class PerPaper(MockBackend):
            def complete(self, role, instruction, payload):
                result = super().complete(role, instruction, payload)
                result.prompt_tokens = 100 * len(payload['papers'])
                return result

        eps = [dry_run(p, review_id=rid, backend=PerPaper()) for rid, p in
               [('a', 'fast'), ('b', 'fast'), ('c', 'raw_retrieve_fast')]]
        model = fit_calibration(eps, mode='fixture')
        env = model.make_env()
        env.reset(seed=0)
        env.step('retrieve_more')
        env.step('retrieve_more')
        _, _, _, _, info = env.step('summarize_fast')
        step = info['transition']
        self.assertEqual(step['total_workflow_tokens'], 1200)  # 总结和校验各读六篇
        self.assertEqual(step['simulation_support']['donor_papers_before'], 6)
        self.assertTrue(step['simulation_support']['low_support'])  # 只有 c，不能用两道三篇题冒充支持
        self.assertFalse(step['simulation_support']['donor_step_failed'])
        # 语料末批不足三篇：没有七篇供体，只能借六篇，不冒充同材料量。
        env.reset(seed=0)
        for _ in range(3):
            env.step('retrieve_more')
        _, _, _, _, info = env.step('summarize_fast')
        support = info['transition']['simulation_support']
        self.assertEqual((support['level'], support['donor_papers_before']), ('nearest_size', 6))
        self.assertTrue(support['low_support'])

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
        self.assertIsNone(info['transition']['simulation_support']['donor_step_failed'])

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


class CoverageFixture(MockBackend):
    def __init__(self, c, cost=10):
        self.c, self.cost = c, cost

    def complete(self, role, instruction, payload):
        out = super().complete(role, instruction, payload)
        out.prompt_tokens = self.cost
        if role == 'verify':
            data = json.loads(out.content)
            data['scores']['coverage'] = self.c
            out.content = json.dumps(data)
        return out


def fixture_label(ep, y):
    return {**{k: ep[k] for k in ('review_id', 'trajectory_id', 'split', 'final_text_sha256')},
            'model': 'fixture', 'prompt_id': 'v1', 'source_kind': 'synthetic_fixture', 'overall_score': y}


class CalibrationV2Tests(unittest.TestCase):
    def test_supervised_line_and_only_measured_stage_adjustment(self):
        eps, labels = [], []
        for rid, c, raw_y in [('A', .2, .1), ('A', .2, .1), ('B', .6, .7)]:
            for policy, y in [('raw_fast', raw_y), ('fast', raw_y + .2)]:
                ep = dry_run(policy, review_id=rid, backend=CoverageFixture(c))
                eps.append(ep)
                labels.append(fixture_label(ep, y))
        model = fit_calibration(eps, labels, mode='fixture', calibration_version='v2')
        params = model.manifest()['quality_model']
        self.assertEqual((model.calibration_version, params['labels'], params['fresh_reviews']), ('v2', 6, 2))
        self.assertAlmostEqual(params['review_mean'], .5)
        self.assertAlmostEqual(params['slope'], .06 / (.04 + .0025))
        raw, summary = [ep['transitions'][-1]['state_before'] for ep in eps[:2]]
        def predict(state, writer='write_fast'):
            return model.quality.estimate(state, writer)[0]
        # Same coverage: the only stage change is the shrunk +/- .1 measured residual.
        self.assertAlmostEqual(predict(summary) - predict(raw), .2 * 2 / (2 + 4))
        self.assertAlmostEqual(predict(summary, 'write_deep'), predict(raw, 'write_deep'))
        summary['verifier']['scores']['consistency'] = .99
        self.assertAlmostEqual(predict(summary) - predict(raw), .2 / 3)
        self.assertAlmostEqual(predict({**raw, 'material_revision': raw['material_revision'] + 1}), .5)
        self.assertIn('estimated', model.quality.estimate(summary, 'write_fast')[1])

    def test_before_papers_bands_and_cost_only_without_labels(self):
        specs = [('A', 'hard', .39, 11), ('B', 'easy', .40, 15), ('C', 'easy', .70, 19)]
        eps = [dry_run('raw_fast', review_id=rid, difficulty=d, backend=CoverageFixture(c, cost))
               for rid, d, c, cost in specs]
        model = fit_calibration(eps, mode='fixture', calibration_version='v2')
        env = model.make_env(reward_config=RewardConfig(quality_weight=0))
        with self.assertRaisesRegex(ValueError, 'collect labels'):
            model.make_env()
        env.reset(seed=0)
        env.step('retrieve_more')
        self.assertIsNone(env.step('write_fast')[-1]['estimated_final_quality'])
        for ep, expected in zip(eps, ['low', 'mid', 'high']):
            state = {**ep['transitions'][-1]['state_before'], 'difficulty': 'easy'}
            outcome = model(state, 'write_fast', random.Random(0))
            support = outcome.simulation_support
            self.assertEqual((support['coverage_band'], support['donor_coverage_band']), (expected, expected))
            self.assertEqual(outcome.input_tokens, ep['transitions'][-1]['input_tokens'])
            self.assertEqual((support['reviews'], support['low_support'], support['cross_band']), (1, True, False))
        # Two other-band reviews don't rescue confidence for an unrepresented low-coverage case.
        without_low = fit_calibration(eps[1:], mode='fixture', calibration_version='v2')
        support = without_low(eps[0]['transitions'][-1]['state_before'], 'write_fast', random.Random(0)).simulation_support
        self.assertEqual((support['reviews'], support['low_support'], support['cross_band']), (2, True, True))

    def test_holdout_reports_matched_drafts_and_actual_rule_denominators(self):
        fit = [dry_run('rule', review_id=rid) for rid in ('A', 'A', 'B')]
        held = dry_run('rule', review_id='C', backend=CoverageFixture(.2))
        eps = [*fit, held]
        labels = [fixture_label(ep, y) for ep, y in zip(eps, [.2, .2, .8, .9])]
        r = evaluate_holdout(eps, labels, mode='fixture', holdout_review_ids=['C'], repeats=2, calibration_version='v2')
        q, rule = r['quality'], r['rule_by_review']['C']
        row, = q['by_manuscript']
        self.assertEqual((r['calibration_version'], q['held_labels'], q['compared'], q['baseline_compared']), ('v2', 1, 1, 1))
        self.assertEqual((row['review_id'], row['trajectory_id'], row['overall_score']), ('C', held['trajectory_id'], .9))
        self.assertAlmostEqual(row['estimated_final_quality'], .5)
        self.assertAlmostEqual(row['train_review_mean'], .5)
        self.assertAlmostEqual(row['train_label_mean'], .4)
        self.assertAlmostEqual(q['mae'], .4)
        self.assertAlmostEqual(q['train_mean_baseline_mae'], .5)
        self.assertEqual(rule['observed']['long_path_known_cost_episodes'], 1)
        self.assertEqual(rule['observed']['runs'][0]['tokens'], held['total_workflow_tokens'])
        sim = rule['simulated']
        self.assertEqual((sim['episodes'], sim['known_cost_episodes'], sim['long_path_episodes']), (2, 2, 0))
        self.assertEqual(sim['terminal_reasons'], {'final_written': 2})
