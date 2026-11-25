import copy
import json
import unittest

from agents.executor import MockBackend
from core.schema import RunnerConfig, comparable_delta, verifier_value
from experiments.collect import dry_run, support_report, validate_episode


class TraceBindingTests(unittest.TestCase):
    def test_report_must_come_from_this_write_not_merely_match_its_hash(self):
        class Writer(MockBackend):
            def __init__(self, heading):
                self.heading = heading

            def complete(self, role, instruction, payload):
                result = super().complete(role, instruction, payload)
                if role == 'write':
                    ids = [p['id'] for p in reversed(payload['papers'])]
                    sections = [{'heading': self.heading, 'text': '首段\n比较。', 'paper_ids': ids},
                                {'heading': '局限', 'text': '仅据摘要。 ', 'paper_ids': ids[-1:]}]
                    result.content = json.dumps({'sections': sections}, indent=2)
                return result

        ep = dry_run('raw_fast', backend=Writer('甲稿'))
        other = dry_run('raw_fast', backend=Writer('乙稿'))
        self.assertIn('> MOCK ', ep['final_text'])
        self.assertNotEqual(ep['final_text_sha256'], other['final_text_sha256'])
        ep.update({k: other[k] for k in ('final_text', 'final_text_sha256')})
        with self.assertRaisesRegex(ValueError, 'successful write output'):
            validate_episode(ep, mode='fixture')

    @staticmethod
    def change_summary_verifier(ep, change):
        # 从第二步总结开始同步伪造所有快照、状态链、派生分和支持统计。
        for t in ep['transitions']:
            before, after = t['state_before'], t['state_after']
            for state in (before, after):
                if state['step_count'] >= 2:
                    change(state['verifier'])
            t['verifier_before'], t['verifier_after'] = verifier_value(before), verifier_value(after)
            t['verifier_delta'] = (comparable_delta(before, after) if t['success']
                                   and not t['action_id'].startswith('write_') else None)
        ep['support'] = support_report([ep])

    def test_verified_snapshot_is_bound_even_when_chain_and_derived_values_agree(self):
        original = dry_run('fast')
        for field, change in (
                ('scores', lambda v: v['scores'].update(coverage=.2)),
                ('validity', lambda v: v['valid_scores'].update(consistency=False)),
                ('revision', lambda v: v.update(checked_material_revision=1)),
                ('issues', lambda v: v.update(issues=['别的校验意见']))):
            with self.subTest(field=field):
                ep = copy.deepcopy(original)
                self.change_summary_verifier(ep, change)
                with self.assertRaisesRegex(ValueError, 'verifier snapshot'):
                    validate_episode(ep, mode='fixture')

    def test_successful_output_must_match_parsed_raw_response(self):
        original = dry_run('fast')
        for i, role in ((1, 'summarize'), (1, 'verify'), (2, 'write')):
            with self.subTest(role=role):
                ep = copy.deepcopy(original)
                call = next(c for c in ep['transitions'][i]['calls'] if c['role'] == role)
                # 额外字段不影响最终渲染/分数，但与原始响应不符，不能冒充执行器解析的结果。
                call['output']['extra'] = '不在原始响应中'
                with self.assertRaisesRegex(ValueError, 'raw response'):
                    validate_episode(ep, mode='fixture')

    def test_repair_and_cache_use_successful_attempt_only(self):
        class Repair(MockBackend):
            def complete(self, role, instruction, payload):
                result = super().complete(role, instruction, payload)
                if role in ('verify', 'write') and 'previous_output' not in payload:
                    result.finish_reason = 'length'
                    result.content = '{"truncated":'
                return result

        ep = dry_run('revise', config=RunnerConfig(cache_enabled=True), backend=Repair())
        self.assertTrue(ep['success'])
        verify = [c for c in ep['transitions'][2]['calls'] if c['role'] == 'verify']
        self.assertEqual([c['status'] for c in verify], ['error', 'ok'])
        self.assertTrue(verify[-1]['cache_hit'])
        self.assertEqual([c['status'] for c in ep['transitions'][-1]['calls']], ['error', 'ok'])

    def test_failed_or_skipped_verify_preserves_previous_result(self):
        class Failure(MockBackend):
            def __init__(self, broken_role):
                self.broken_role = broken_role

            def complete(self, role, instruction, payload):
                result = super().complete(role, instruction, payload)
                result.prompt_tokens = 10
                if role == self.broken_role and (role == 'summarize' or payload['items']):
                    result.content = '{}'
                return result

        for role, budget in (('summarize', 100), ('verify', 100), (None, 20)):
            with self.subTest(role=role, budget=budget):
                ep = dry_run('fast', backend=Failure(role), config=RunnerConfig(workflow_tokens_budget=budget))
                step = ep['transitions'][1]
                self.assertFalse(step['success'])
                self.assertEqual(step['state_before']['verifier'], step['state_after']['verifier'])
                self.assertEqual(step['verifier_after'] is None, role != 'summarize')
                if role is None:
                    self.assertEqual([c['role'] for c in step['calls']], ['summarize'])
                self.change_summary_verifier(ep, lambda v: v.update(issues=['伪造的旧意见']))
                with self.assertRaisesRegex(ValueError, 'verifier snapshot'):
                    validate_episode(ep, mode='fixture')

    def test_write_needs_a_successful_call_and_cannot_change_verifier(self):
        original = dry_run('raw_fast', config=RunnerConfig(output_repair_attempts=0))
        ep = copy.deepcopy(original)
        call = ep['transitions'][-1]['calls'][-1]
        call.update(status='error', finish_reason='length')
        del call['output']  # 即使失败尝试的原始响应看起来恰好是这篇稿，也不能发布。
        with self.assertRaisesRegex(ValueError, 'step success'):
            validate_episode(ep, mode='fixture')

        original['transitions'][-1]['state_after']['verifier']['issues'] = ['写稿步骤并没有校验']
        with self.assertRaisesRegex(ValueError, 'verifier snapshot'):
            validate_episode(original, mode='fixture')

    def test_ok_call_is_not_always_a_published_report(self):
        class Metered(MockBackend):
            def __init__(self, write_cost):
                self.write_cost = write_cost

            def complete(self, role, instruction, payload):
                result = super().complete(role, instruction, payload)
                result.prompt_tokens = self.write_cost if role == 'write' else 10
                return result

        for budget, cost, reason in ((21, 10, 'final_written'), (20, 10, 'token_budget'),
                                     (19, 10, 'token_budget'), (20, None, 'usage_unknown'), (10, 10, 'token_budget')):
            with self.subTest(budget=budget, cost=cost):
                ep = dry_run('raw_fast', config=RunnerConfig(workflow_tokens_budget=budget), backend=Metered(cost))
                last = ep['transitions'][-1]
                self.assertEqual(last['calls'][-1]['status'], 'ok')
                self.assertEqual(ep['terminal_reason'], reason)
                self.assertEqual(ep['final_text'] is not None, reason == 'final_written')
                self.assertEqual(last['success'], budget == 10 or reason == 'final_written')
