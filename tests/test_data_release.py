import unittest
from unittest.mock import patch

from utils.data_release import ROOT, check_evidence, check_public, load_manifest, read_json, validate


class ReleaseTests(unittest.TestCase):
    def test_checked_in_release(self):
        self.assertEqual(validate()['papers'], 600)
        train, test = load_manifest('train'), load_manifest('test')
        self.assertEqual((len(train['tasks']), len(test['tasks'])), (30, 10))
        self.assertTrue(train['tasks'].keys().isdisjoint(test['tasks']))
        self.assertEqual(train['papers'], test['papers'])
        self.assertEqual(train['judge'], test['judge'])

    def test_public_check_never_reads_private_criteria(self):
        def public_only(path):
            self.assertNotIn('.local-data', path.parts)
            return read_json(path)
        with patch('utils.data_release.read_json', side_effect=public_only):
            self.assertFalse(validate()['sealed_evidence_checked'])

    def test_edited_task_does_not_match_preissued_manifest(self):
        def edited(path):
            value = read_json(path)
            if path == ROOT / 'data/train_tasks.json':
                value[0]['survey_instruction'] += '比较额外的方法。'
            return value
        with patch('utils.data_release.read_json', side_effect=edited), self.assertRaisesRegex(ValueError, '冻结清单'):
            validate()

    def test_rejects_private_fields_in_public_task(self):
        row = {'config_version': 1, 'review_id': 'sample', 'split': 'train', 'survey_instruction': '比较两种方法',
               'difficulty': 'medium', 'corpus_scope': 'local_arxiv_ai'}
        self.assertEqual(check_public([row], 'train', 1), {'sample'})
        for field in ('source_papers', 'criteria', 'seed_ids', 'overall_score'):
            with self.subTest(field=field), self.assertRaises(ValueError):
                check_public([{**row, field: []}], 'train', 1)

    def test_evidence_must_match_its_own_abstract(self):
        papers = {'P1': {'abstract': 'First text.'}, 'P2': {'abstract': 'Another text.'}}
        sources = [{'review_id': 'A', 'seed_ids': ['P1']}]
        cited = {'paper_id': 'P2', 'quote': 'Another text.'}
        criteria = [{'review_id': 'A', 'criteria': [{'aspect': '比较条件', 'evidence': [cited]}]}]
        check_evidence(sources, criteria, {'A'}, papers)
        for wrong in ('First text.', 'Fabricated.', ''):
            with self.subTest(wrong=wrong), self.assertRaises(ValueError):
                cited['quote'] = wrong
                check_evidence(sources, criteria, {'A'}, papers)

    def test_duplicate_or_missing_criteria_rejected(self):
        sources = [{'review_id': 'A', 'seed_ids': ['P1']}]
        for criteria in ([], [{'review_id': 'A'}] * 2):
            with self.subTest(criteria=criteria), self.assertRaises(ValueError):
                check_evidence(sources, criteria, {'A'}, {'P1': {'abstract': 'X'}})


if __name__ == '__main__':
    unittest.main()
