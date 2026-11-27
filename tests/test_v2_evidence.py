"""V2 pilot action and scoring gates; all model calls are offline fakes."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from core.schema import ACTIONS
from core.v2_evidence import EvidenceRunner, V2_ACTIONS, load_pilot_sections
from evaluation.v2_sensitivity import POINTS, score_response, sensitivity_check
from utils.data_release import load_corpus


class FakeBackend:
    def __init__(self):
        self.calls = []

    def complete(self, role, instruction, payload):
        self.calls.append((role, payload))
        if role == 'rewrite_query':
            value = {'gap': '缺少触觉字母任务的实机证据', 'query': 'braille tactile alphabet sample efficiency'}
        elif role == 'write':
            value = {'text': '触觉字母任务的正文给出 HER 信息。[paper-b]', 'paper_ids': ['paper-b']}
        else:
            raise AssertionError('Unexpected model role.')
        return value, {'prompt_tokens': 13, 'completion_tokens': 7}


def fixture():
    corpus = {'papers': [
        {'id': 'paper-a', 'title': 'Robot simulation policy transfer',
         'abstract': 'A simulation policy uses real robot evaluation.'},
        {'id': 'paper-b', 'title': 'Braille tactile alphabet',
         'abstract': 'Tactile alphabet tasks involve sample efficiency.'},
    ]}
    section_text = 'HER improved sample efficiency in alphabet tasks.'
    section = {'paper_id': 'paper-b', 'section_id': 'discussion', 'heading': 'Discussion',
               'search_terms': ['braille', 'alphabet', 'sample efficiency'],
               'text': section_text, 'sha256': hashlib.sha256(section_text.encode()).hexdigest(),
               'source_url': 'https://arxiv.org/pdf/example', 'source_locator': 'page 2'}
    task = {'review_id': 'pilot-1', 'split': 'train',
            'survey_instruction': 'robot simulation policy transfer'}
    return corpus, task, {('paper-b', 'discussion'): section}


class V2EvidenceTests(unittest.TestCase):
    def test_new_actions_create_a_real_information_boundary(self):
        corpus, task, sections = fixture()
        backend = FakeBackend()
        runner = EvidenceRunner(corpus, task, sections, backend, batch_size=1)
        self.assertEqual(len(ACTIONS), 5)  # Frozen V1 interface has not changed.
        self.assertEqual(V2_ACTIONS, ('retrieve_more', 'rewrite_query', 'open_section', 'write'))
        self.assertEqual(runner.observe()['action_mask'], [True, False, False, False])
        with self.assertRaises(ValueError):
            runner.step('open_section')

        first = runner.step('retrieve_more')
        self.assertEqual(first['detail']['selected'][0]['paper_id'], 'paper-a')
        self.assertEqual(runner.observe()['action_mask'], [True, True, False, True])
        runner.step('rewrite_query')
        self.assertEqual(runner.observe()['identified_gap'], '缺少触觉字母任务的实机证据')
        second = runner.step('retrieve_more')
        self.assertEqual(second['detail']['selected'][0]['paper_id'], 'paper-b')
        opened = runner.step('open_section')
        self.assertEqual(opened['detail']['sha256'], sections[('paper-b', 'discussion')]['sha256'])
        self.assertEqual(backend.calls[0][1]['retrieved'][0]['id'], 'paper-a')
        self.assertNotIn('text', backend.calls[0][1]['retrieved'][0])
        runner.step('write')
        self.assertEqual(backend.calls[1][1]['opened_sections'][0]['text'], sections[('paper-b', 'discussion')]['text'])
        self.assertEqual(runner.result()['terminal_reason'], 'final_written')
        self.assertEqual(runner.result()['total_workflow_tokens'], 40)
        self.assertEqual(runner.observe()['action_mask'], [False] * 4)

    def test_section_release_is_bound_to_current_corpus_and_hash(self):
        corpus = load_corpus()
        sections = load_pilot_sections(corpus=corpus)
        self.assertEqual(set(sections), {('2008.02646v1', 'discussion-p7-her')})
        path = Path('data/v2_pilot_sections.json')
        tampered = json.loads(path.read_text(encoding='utf8'))
        tampered['sections'][0]['text'] += ' invented'
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / 'sections.json'
            file.write_text(json.dumps(tampered), encoding='utf8')
            with self.assertRaisesRegex(ValueError, 'provenance'):
                load_pilot_sections(file, corpus)

    def test_judge_coverage_is_explicit_and_catches_omission(self):
        answer = ('四项仿真、三项实机；连续字母任务有样本效率瓶颈。'
                  'HER 使样本效率最高提高 10 倍；HER 的改善在字母任务尤其明显。'
                  '这些结论只限本论文。[2008.02646v1]')
        quotes = {
            'simulation_real_scope': '四项仿真、三项实机',
            'efficiency_bottleneck': '连续字母任务有样本效率瓶颈',
            'method': 'HER 使样本效率最高提高 10 倍',
            'magnitude': 'HER 使样本效率最高提高 10 倍',
            'where_effective': 'HER 的改善在字母任务尤其明显',
            'evidence_boundary': '这些结论只限本论文',
        }

        def response(covered, spans=quotes):
            return {'model': 'deepseek-v4-pro', 'choices': [{'finish_reason': 'stop',
                    'message': {'content': json.dumps({'covered': {key: {'present': flag,
                        'answer_quote': spans[key] if flag else ''} for key, flag in covered.items()}, 'factuality': .95,
                                                       'citation_support': .95, 'consistency': .95,
                                                       'rationale': 'source-grounded assessment'})}}]}

        omitted_flags = {key: key not in ('method', 'magnitude', 'where_effective') for key in POINTS}
        complete_flags = dict.fromkeys(POINTS, True)
        omitted, complete = score_response(response(omitted_flags), answer), score_response(response(complete_flags), answer)
        self.assertEqual((omitted['coverage'], complete['coverage']), (.5, 1.0))
        self.assertTrue(sensitivity_check(omitted, complete))
        bad_flags = dict(omitted_flags, magnitude=True)
        with self.assertRaisesRegex(ValueError, 'misclassified'):
            sensitivity_check(score_response(response(bad_flags), answer), complete)
        pure_omission = '四项仿真、三项实机；连续字母任务有样本效率瓶颈。这些结论只限本论文。'
        with self.assertRaisesRegex(ValueError, 'not present'):
            score_response(response(complete_flags), pure_omission)
        duplicate = response(complete_flags)
        duplicate['choices'][0]['message']['content'] = '{"covered":{},"covered":{}}'
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            score_response(duplicate, answer)
        nonfinite = response(complete_flags)
        nonfinite['choices'][0]['message']['content'] = '{"factuality":NaN}'
        with self.assertRaisesRegex(ValueError, 'Non-finite'):
            score_response(nonfinite, answer)

    def test_test_split_is_rejected(self):
        corpus, task, sections = fixture()
        task['split'] = 'test'
        with self.assertRaisesRegex(ValueError, 'training task'):
            EvidenceRunner(corpus, task, sections, FakeBackend())

    def test_invalid_model_action_is_recorded_and_stops(self):
        class BadBackend(FakeBackend):
            def complete(self, role, instruction, payload):
                return {'gap': 'missing', 'query': payload['current_query']}, {
                    'prompt_tokens': 10, 'completion_tokens': 5}

        corpus, task, sections = fixture()
        runner = EvidenceRunner(corpus, task, sections, BadBackend(), batch_size=1)
        runner.step('retrieve_more')
        failed = runner.step('rewrite_query')
        self.assertFalse(failed['success'])
        self.assertEqual(runner.result()['terminal_reason'], 'action_error')
        self.assertEqual(runner.result()['total_workflow_tokens'], 15)
        self.assertEqual(runner.observe()['action_mask'], [False] * 4)


if __name__ == '__main__':
    unittest.main()
