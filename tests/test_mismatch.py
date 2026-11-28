import unittest

from core.schema import SCORE_KEYS, empty_verifier
from evaluation.mismatch import classify_step, summarize_mismatch
from rl.synthetic_env import Outcome, SyntheticEnv


def state(*scores, revision=1, **updates):
    verifier = empty_verifier()
    if scores:
        verifier['checked_material_revision'] = revision
        for k, score in zip(SCORE_KEYS, scores):
            verifier['scores'][k], verifier['valid_scores'][k] = score, True
    return {'verifier': verifier, 'material_revision': revision, 'summary_available': len(scores) > 1,
            'difficulty': 'medium', 'retrieved_paper_count': 3, 'step_count': 6, 'max_steps': 8,
            'workflow_tokens_used': 36000, 'workflow_tokens_budget': 48000, **updates}


def step(action, before, after=None, **fields):
    return {'action_id': action, 'state_before': before, 'state_after': before if after is None else after,
            'success': True, **fields}


class MismatchTests(unittest.TestCase):
    def test_first_retrieval_and_first_summary_are_not_rework(self):
        env = SyntheticEnv()
        env.reset(seed=1, options={'difficulty': 'medium'})
        for action in ('retrieve_more', 'summarize_fast'):
            transition = env.step(action)[-1]['transition']
            self.assertIsNone(transition['verifier_delta'])
            result = classify_step(transition)
            self.assertEqual(result['by_category'], {'ineffective': False})
            self.assertIs(result['mismatch_any'], False)

    def test_write_risk_only_with_fresh_gap_and_room_for_another_action(self):
        for before, expected in ((state(.74), True), (state(.75), False),  # no summary: coverage only
                                 (state(.92, .9, .71, .9), True), (state(.9, .9, .9, .9), False),
                                 (state(.74, step_count=7), False), (state(.74, workflow_tokens_used=36001), False),
                                 (state(), None), (state(step_count=7), False),
                                 (state(.74, workflow_tokens_used=None), None),
                                 (state(.9, workflow_tokens_used=None), False)):
            with self.subTest(before=before):
                result = classify_step(step('write_fast', before))
                self.assertIs(result['by_category']['premature_write'], expected)
                self.assertIs(result['mismatch_any'], expected)
        stale = state(.1)
        stale['material_revision'] += 1
        self.assertIsNone(classify_step(step('write_fast', stale))['mismatch_any'])
        deep = classify_step(step('write_deep', state(.9)))
        self.assertIsNone(deep['by_category']['redundant'])
        self.assertIsNone(deep['mismatch_any'])

    def test_redundancy_checks_target_gaps_and_individual_gains(self):
        for action, before, after, expected in (
                ('retrieve_more', (.85,), (.87,), True),  # inclusive .02 gain
                ('retrieve_more', (.84,), (.84,), False),
                ('retrieve_more', (.87, .9, .70, .9), (.87,), False),
                ('retrieve_more', (.87, .9, .9, .9), (.87,), True),
                ('summarize_fast', (.8, .8, .8, .8), (.8, .8, .8, .82), True),
                ('summarize_fast', (.8, .8, .8, .8), (.8, .8, .8, .84), False),
                ('summarize_fast', (.9, .7, .9, .9), (.9, .7, .9, .9), False)):
            with self.subTest(action=action, before=before, after=after):
                result = classify_step(step(action, state(*before), state(*after, revision=2)))
                self.assertIs(result['by_category']['redundant'], expected)

    def test_low_gain_rework_requires_two_comparable_checks(self):
        before = state(.7, .7, .7, .7, difficulty='easy')
        both = classify_step(step('summarize_deep', before, state(.72, .72, .72, .72, revision=2)))
        self.assertEqual(both['by_category'], {'redundant': True, 'ineffective': True})
        stale = state(.6, revision=2)
        stale['material_revision'] += 1
        for after in (state(.7, revision=2), stale, before):  # changed validity, stale, or no new check
            result = classify_step(step('retrieve_more', before, after, verifier_delta=0.0))
            self.assertIsNone(result['by_category']['ineffective'])
            self.assertIsNone(result['mismatch_any'])
        # First easy deep summary has no comparable composite delta, not evidence of a zero gain.
        first = classify_step(step('summarize_deep', state(.7, difficulty='easy'), state(.7, .7, .7, .7, revision=2)))
        self.assertIsNone(first['by_category']['redundant'])

    def test_unsupported_censor_not_a_failure_or_known_mismatch(self):
        def episode(write, outcome):
            env = SyntheticEnv(outcome_model=lambda _, a, rng: (
                Outcome(True, 0, 0, 50, scores={'coverage': .68}) if a == 'retrieve_more' else outcome))
            env.reset(seed=3)
            return [env.step(a)[-1]['transition'] for a in ('retrieve_more', write)]

        failed = episode('write_deep', Outcome(False, 8, 4, simulation_support={'level': 'nearest_size', 'low_support': True}))
        censored = episode('write_fast', Outcome(False, None, None, simulation_support={'level': 'unsupported', 'low_support': True}))
        report = summarize_mismatch([failed, censored])
        self.assertEqual((report['any_count'], report['known_steps'], report['unknown_steps'], report['per_episode']),
                         (1, 3, 1, .5))  # supported failure + premature risk count once, across both episodes
        self.assertEqual((report['execution_failure_count'], report['unsupported_censored_steps']), (1, 1))
        for category, expected in (('ineffective', (1, 4, 3, 1)), ('premature_write', (1, 2, 1, 1)),
                                   ('redundant', (0, 1, 0, 1))):
            counts = report['by_category'][category]
            self.assertEqual(tuple(counts[k] for k in ('count', 'applicable_steps', 'known_steps', 'unknown_steps')), expected)
