import copy
import unittest
from unittest.mock import patch

from agents.executor import MockBackend
from core.runner import Runner
from core.schema import ACTIONS, ReviewTask, RunnerConfig
from experiments.collect import (PLANS, collect_episode, dry_run, json_sha256, paper_index,
                                 support_report, validate_collection)
from litreview.clients import DeepSeekBackend


class Metered(MockBackend):
    def complete(self, role, instruction, payload):
        c = super().complete(role, instruction, payload)
        c.prompt_tokens, c.completion_tokens = 12, 8
        return c


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
        task = ReviewTask('R','topic',split='train')
        papers = [{'id':'A','title':'Topic','abstract':'A hypothetical paper.'}]
        corpus = {'papers':papers,'purpose':'formal_experiment','dataset_kind':'arxiv_abstracts'}
        runner = Runner(corpus, DeepSeekBackend(api_key='fixture', model='deepseek-flash'))
        manifest = {'config_version':1,'source_id':'fixture-manifest','dataset_kind':'arxiv_abstracts',
                    'data_purpose':'formal_experiment','papers':paper_index(papers),
                    'tasks':{'R':{'split':'train','task_sha256':json_sha256(task.to_dict())}},
                    'judge':{'model':'fixture','prompt_id':'v1'}}
        with patch.object(DeepSeekBackend,'complete',side_effect=AssertionError('no network')) as api:
            for bad_task in (ReviewTask('R','topic',split='test'),ReviewTask('R','different topic',split='train')):
                with self.assertRaises(ValueError): collect_episode(runner,bad_task,manifest=manifest)
            with self.assertRaises(ValueError): collect_episode(runner,task,manifest={**manifest,'papers':{}})
            with self.assertRaises(ValueError): collect_episode(Runner(corpus),task,manifest=manifest)
            pilot = Runner({**corpus,'purpose':'pilot_only'}, runner.executor.backend)
            with self.assertRaises(ValueError): collect_episode(pilot,task,manifest=manifest)
            api.assert_not_called()
            self.assertIsNone(runner.task)
        # 验证正式分支的连线：替换传输边界，产物只在本测试内存在。
        with patch.object(DeepSeekBackend,'complete',side_effect=Metered().complete):
            ep = collect_episode(runner,task,manifest=manifest)
        label = {**fixture_label(ep), 'source_kind':'independent_judge','prompt_id':'v1',
                 'subscores':{'support':.7}, 'rationale':'unit test only'}
        validate_collection([ep], [label], manifest=manifest)
        with self.assertRaises(ValueError): validate_collection([ep],[{**label,'prompt_id':'v2'}],manifest=manifest)
