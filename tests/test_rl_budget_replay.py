"""Paired shared Runner / calibrated simulator tests using only local dry_run fixtures."""

import itertools
import unittest
from unittest.mock import patch

from agents.executor import MockBackend
from core.schema import RunnerConfig, verifier_value
from experiments.collect import dry_run
from rl.calibration import fit_calibration


class MeteredFixture(MockBackend):
    """Inject metering and invalid JSON objects; AgentExecutor performs the real repairs."""

    def __init__(self, target, initial_cost, unknown=False):
        self.target, self.initial_cost, self.unknown = target, initial_cost, unknown

    def complete(self, role, instruction, payload):
        result = super().complete(role, instruction, payload)
        items, papers = payload['items'], payload['papers']
        if role == 'verify' and len(papers) == 3 and not items:
            result.prompt_tokens = self.initial_cost
        elif ((self.target == 'retrieve_verify' and role == 'verify' and not items and len(papers) > 3)
              or (self.target == 'summary_verify' and role == 'verify' and items)
              or self.target == role):
            if 'previous_output' not in payload:
                result.content = '{}'  # parseable JSON; fails role-specific validation
                result.prompt_tokens, result.completion_tokens = (None if self.unknown else 1000), 200
            else:
                result.prompt_tokens, result.completion_tokens = 400, 200
        elif self.target == 'cached_verify' and role == 'summarize' and items:
            result.prompt_tokens = None if self.unknown else 1200
        elif self.target == 'summarize' and role == 'verify' and items:
            result.prompt_tokens, result.completion_tokens = 10, 5
        return result


class BudgetReplayTests(unittest.TestCase):
    def replay(self, plan, target, remaining, index, *, unknown=False):
        # Same physical config throughout. The easy donor observes the whole repair;
        # the two medium donors supply a costly first retrieval, but not the tested action.
        budget = 4000
        config = RunnerConfig(workflow_tokens_budget=budget, cache_enabled=target == 'cached_verify')

        def episode(policy, rid, difficulty, initial):
            with patch('agents.executor.time.monotonic', side_effect=itertools.count()):
                return dry_run(policy, review_id=rid, difficulty=difficulty, config=config,
                               backend=MeteredFixture(target, initial, unknown))

        donor = episode(plan, 'donor', 'easy', 0)
        prefixes = [episode('raw_deep', rid, 'medium', budget - remaining) for rid in ('a', 'b')]
        model = fit_calibration([donor, *prefixes], mode='fixture')
        actual = episode(plan, 'paired', 'medium', budget - remaining)
        env = model.make_env()
        env.reset(seed=0, options={'review_id': 'paired', 'difficulty': 'medium'})
        simulated = []
        for ref in actual['transitions']:
            _, _, _, _, info = env.step(ref['action_id'])
            step = info['transition']
            simulated.append(step)
            for k in ('state_before', 'state_after', 'action_mask', 'next_mask', 'input_tokens',
                      'output_tokens', 'verifier_tokens', 'total_workflow_tokens', 'verifier_before',
                      'verifier_after', 'verifier_delta', 'success', 'done', 'terminal_reason'):
                self.assertEqual(step[k], ref[k], (ref['action_id'], k))
            fields = ('role', 'attempt', 'status', 'cache_hit', 'prompt_tokens', 'completion_tokens', 'elapsed_seconds')
            self.assertEqual(step['calls'], [{k: c[k] for k in fields} for c in ref['calls']])
            self.assertEqual(step['simulation_support']['call_seconds'], sum(c['elapsed_seconds'] for c in ref['calls']))
            self.assertEqual(step['execution_kind'], 'simulated')
        return simulated[index], donor['transitions'][index]

    def test_followup_retrieval_repairs_only_while_budget_is_positive(self):
        for remaining, repaired in ((1000, False), (1200, False), (1201, True)):
            with self.subTest(remaining=remaining):
                step, donor = self.replay('retrieve_again', 'retrieve_verify', remaining, 2)
                self.assertEqual([c['status'] for c in donor['calls']], ['error', 'ok'])
                self.assertEqual(donor['verifier_tokens'], 1800)
                self.assertEqual(step['verifier_tokens'], 1800 if repaired else 1200)
                self.assertEqual(step['success'], repaired)
                self.assertEqual(step['terminal_reason'], 'token_budget')
                state = step['state_after']
                self.assertEqual((state['material_revision'], state['retrieved_paper_count']), (3, 6))
                self.assertFalse(state['summary_available'])
                self.assertEqual(verifier_value(state), .8 if repaired else None)

    def test_summary_and_verifier_commit_at_different_calls(self):
        for target, remaining, generated, checked, spend in (
            ('summarize', 1000, False, False, 1200),  # generation repair blocked
            ('summarize', 1201, True, False, 1800),   # generation repaired, Verify blocked
            ('summarize', 1801, True, True, 1815),    # Verify admitted despite only 1 token left
            ('summary_verify', 1000, True, False, 1200),  # Verify failed; its repair is blocked
        ):
            with self.subTest(target=target, remaining=remaining):
                step, donor = self.replay('fast', target, remaining, 1)
                self.assertTrue(donor['success'])
                self.assertEqual(step['total_workflow_tokens'], spend)
                self.assertEqual(step['state_after']['summary_available'], generated)
                self.assertEqual(step['success'], checked)
                self.assertEqual(step['terminal_reason'], 'token_budget')
                if generated and not checked:
                    self.assertIsNone(verifier_value(step['state_after']))

    def test_writer_repair_and_final_cost_priority(self):
        for remaining, attempts, accepted in ((1000, 1, False), (1800, 2, False), (1801, 2, True)):
            with self.subTest(remaining=remaining):
                step, donor = self.replay('raw_fast', 'write', remaining, 1)
                self.assertEqual([c['status'] for c in donor['calls']], ['error', 'ok'])
                self.assertEqual(len(step['calls']), attempts)
                self.assertEqual(step['success'], accepted)
                self.assertEqual(step['estimated_final_quality'], .8 if accepted else None)
                self.assertEqual(step['quality_method'], 'fresh_valid_verifier_mean' if accepted else None)
                self.assertEqual(step['terminal_reason'], 'final_written' if accepted else 'token_budget')

    def test_unknown_usage_before_repair_blocks_later_calls(self):
        step, _ = self.replay('raw_fast', 'write', 3000, 1, unknown=True)
        self.assertEqual([c['status'] for c in step['calls']], ['error'])
        self.assertIsNone(step['input_tokens'])
        self.assertEqual(step['output_tokens'], 200)
        self.assertIsNone(step['total_workflow_tokens'])
        self.assertIsNone(step['estimated_final_quality'])
        self.assertEqual(step['terminal_reason'], 'usage_unknown')

    def test_observed_cache_hit_after_exhausted_or_unknown_generation(self):
        # Re-summarizing unchanged papers emits identical items, so AgentExecutor
        # finds the previous Verify response before it asks Runner._can_call().
        for unknown in (False, True):
            with self.subTest(unknown=unknown):
                step, _ = self.replay('revise', 'cached_verify', 1200, 2, unknown=unknown)
                self.assertEqual([c['cache_hit'] for c in step['calls']], [False, True])
                self.assertTrue(step['success'])
                self.assertEqual(step['verifier_after'], .8)
                self.assertEqual(step['verifier_tokens'], 0)
                self.assertEqual(step['terminal_reason'], 'usage_unknown' if unknown else 'token_budget')
