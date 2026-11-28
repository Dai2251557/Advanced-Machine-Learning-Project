import copy
import json
import unittest
from unittest.mock import patch

from core.runner import Runner
from core.schema import ACTIONS, ReviewTask, RunnerConfig
from evaluation.judge import JUDGE_MODEL, build_judge_request, finalize_judge_result, prepare_train_review
from experiments.collect import PLANS, collect_episode, dry_run, support_report, validate_collection
from litreview.clients import DeepSeekBackend
from tests.formal_fixture import Metered, formal_inputs


def fixture_label(ep, score=.7):
    return {**{k:ep[k] for k in ('review_id','trajectory_id','split','final_text_sha256')},
            'source_kind':'synthetic_fixture', 'model':'fixture', 'prompt_id':'fixture', 'overall_score':score}


class CollectionTests(unittest.TestCase):
    def test_all_plans_and_support(self):
        episodes = [dry_run(policy, backend=Metered()) for policy in PLANS]
        validate_collection(episodes, mode='fixture')
        self.assertTrue(all(e['source']['corpus_size'] == 7 for e in episodes))
        selected = {r['action_id'] for r in support_report(episodes) if r['selected']}
        self.assertEqual(selected, set(ACTIONS))
        for ep in episodes:
            steps = ep['transitions']
            self.assertEqual(steps[-1]['next_mask'], [False] * 5)
            self.assertEqual([s['next_mask'] for s in steps[:-1]], [s['action_mask'] for s in steps[1:]])
            self.assertEqual(ep['total_workflow_tokens'], sum(len(s['calls'])*20 for s in steps))
            with self.assertRaises(ValueError):
                validate_collection([ep])

    def test_label_join_does_not_mutate_and_rejects_bad_identity(self):
        ep = dry_run('deep')
        label = fixture_label(ep)
        original = copy.deepcopy(ep)
        self.assertEqual(validate_collection([ep], [label], mode='fixture')[ep['trajectory_id']], label)
        self.assertEqual(ep, original)
        for labels in ([label,label], [{**label,'review_id':'wrong'}], [{**label,'split':'test'}],
                       [{**label,'final_text_sha256':'0'*64}], [{**label,'overall_score':float('nan')}],
                       [{**label,'source_kind':'independent_judge'}]):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                validate_collection([ep], labels, mode='fixture')
        with self.assertRaises(ValueError):
            validate_collection([ep,ep], mode='fixture')

    def test_unknown_usage_budget_boundary_and_call_repair(self):
        class Unknown(Metered):
            def complete(self, *args):
                c = super().complete(*args)
                c.prompt_tokens = None
                return c
        class Repair(Metered):
            def complete(self, role, instruction, payload):
                c = super().complete(role,instruction,payload)
                if role == 'summarize' and 'previous_output' not in payload:
                    c.content = '{}'
                return c
        unknown = dry_run(backend=Unknown())
        self.assertIsNone(unknown['total_workflow_tokens'])
        self.assertEqual(unknown['terminal_reason'],'usage_unknown')
        ep = dry_run('raw_fast', config=RunnerConfig(workflow_tokens_budget=40), backend=Metered())
        self.assertEqual(ep['terminal_reason'],'token_budget')
        with self.assertRaises(ValueError):
            validate_collection([ep],[fixture_label(ep)],mode='fixture')
        repaired = dry_run('fast', backend=Repair())
        self.assertEqual(repaired['transitions'][1]['total_workflow_tokens'],60)
        for mut in ('cost','label','mask'):
            broken = copy.deepcopy(repaired)
            t = broken['transitions'][0]
            if mut == 'cost': t['verifier_tokens'] += 1
            if mut == 'label': t['state_before']['judge_score'] = .9
            if mut == 'mask': t['next_mask'][0] = False
            with self.assertRaises(ValueError): validate_collection([broken],mode='fixture')

    def test_formal_sources_checked_before_runner_reset(self):
        runner, corpus, task, manifest = formal_inputs()
        with patch.object(DeepSeekBackend,'complete',side_effect=AssertionError('no network')) as api:
            for bad_task in (ReviewTask('R','topic',split='test'),ReviewTask('R','different topic',split='train')):
                with self.assertRaises(ValueError): collect_episode(runner,bad_task,manifest=manifest)
            with self.assertRaises(ValueError): collect_episode(runner,task,manifest={**manifest,'papers':{}})
            with self.assertRaises(ValueError): collect_episode(Runner(corpus),task,manifest=manifest)
            pilot = Runner({**corpus,'purpose':'pilot_only'}, runner.executor.backend)
            with self.assertRaises(ValueError): collect_episode(pilot,task,manifest=manifest)
            api.assert_not_called()
            self.assertIsNone(runner.task)

    def test_formal_join_accepts_v3_result_and_rejects_malformed_archives(self):
        # 传输替身与伪回执只测试归档协议；不访问网络，不产生正式评分。
        runner, corpus, task, manifest = formal_inputs()
        with patch.object(DeepSeekBackend, 'complete', side_effect=Metered().complete):
            ep = collect_episode(runner, task, manifest=manifest)
        prepared = prepare_train_review(episode=ep, frozen_papers=corpus['papers'], manifest=manifest)
        subscores = {'coverage': .62, 'factuality': .62, 'citation_support': .63, 'consistency': .63}
        scores = {**subscores, 'overall': .63, 'rationale': 'unit test only'}  # .625 rounds half-up.
        receipt = {'binding': prepared['binding'], 'request': build_judge_request(prepared), 'response': {
            'model': JUDGE_MODEL, 'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(scores)}}]}}
        label = finalize_judge_result(prepared, receipt)
        self.assertEqual(validate_collection([ep], [label], manifest=manifest), {ep['trajectory_id']: label})
        self.assertEqual(validate_collection([ep], manifest=manifest), {})

        bad_scores = [{'support': .7}, {k: v for k, v in subscores.items() if k != 'consistency'},
                      {**subscores, 'extra': .5}]
        bad_scores += [dict(subscores, coverage=v) for v in (.621, False, '.62', float('nan'), 1.01)]
        bad_labels = [dict(label, subscores=s) for s in bad_scores]
        bad_labels += [dict(label, overall_score=v) for v in (.62, .625, True)]
        bad_labels += [dict(label, model='other'), dict(label, prompt_id='v2'), dict(label, rationale='  '),
                       dict(label, aggregation='weighted_mean'), dict(label, request_sha256='A' * 64),
                       dict(label, request_sha256='0' * 63), dict(label, request_sha256=None)]
        bad_labels += [{k: v for k, v in label.items() if k != field} for field in ('request_sha256', 'aggregation')]
        for bad in bad_labels:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_collection([ep], [bad], manifest=manifest)
