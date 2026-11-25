import unittest
import io
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
from contextlib import redirect_stdout

from agents.retrieval import BilingualRetriever, expand_query
from core.runner import Runner
from core.schema import ReviewTask, RunnerConfig
from agents.executor import MockBackend


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.papers = [
            {'id': 'a', 'title': 'Medical images', 'abstract': 'Segmentation of anatomical structures.'},
            {'id': 'b', 'title': 'Traffic ramp metering', 'abstract': 'Reinforcement learning controls traffic using video.'},
            {'id': 'c', 'title': 'Private federated learning', 'abstract': 'Differential privacy protects clients.'},
        ]

    def test_chinese_task_retrieves_english_sources(self):
        ranker = BilingualRetriever(self.papers)
        for query, expected in [('强化学习的交通匝道控制与视频', 'b'), ('医学图像分割', 'a'), ('联邦学习的差分隐私', 'c')]:
            with self.subTest(query=query):
                ranked = ranker.rank(query)
                self.assertEqual(ranked[0]['paper_id'], expected)
                self.assertGreater(ranked[0]['score'], 0)

    def test_query_expansion_is_public_text_only_and_english_is_preserved(self):
        terms = expand_query('FPGA 与联邦学习')
        self.assertIn('fpga', terms)
        self.assertIn('federated', terms)
        self.assertEqual(BilingualRetriever(self.papers).rank('video traffic')[0]['paper_id'], 'b')

    def test_repeat_retrieval_excludes_used_papers_and_logs_zero_support(self):
        runner = Runner({'papers': self.papers}, config=RunnerConfig(retrieval_batch_size=1))
        runner.reset(ReviewTask('test', 'zzzzunknown'))
        runner.step('retrieve_more')
        runner.step('retrieve_more')
        self.assertEqual([p['id'] for p in runner.selected], ['a', 'b'])
        self.assertTrue(all(e['zero_score_fallback'] for e in runner.result()['retrieval']['events']))

    def test_id_and_paper_order_do_not_determine_nonzero_relevance(self):
        ranker = BilingualRetriever(list(reversed(self.papers)))
        self.assertEqual(ranker.rank('匝道控制')[0]['paper_id'], 'b')
        self.assertNotIn('b', [r['paper_id'] for r in ranker.rank('匝道控制', ['b'])])

    def test_training_probe_writes_valid_collection_without_network(self):
        from experiments.retrieval_probe import main
        from litreview.clients import DeepSeekBackend
        requests = []
        def respond(request, **kwargs):
            body = json.loads(request.data)
            instruction = body['messages'][0]['content']
            payload = json.loads(body['messages'][1]['content'])
            requests.append(payload)
            role = 'verify' if '"valid_scores"' in instruction else 'write' if '"sections"' in instruction else 'summarize'
            completion = MockBackend().complete(role, instruction, payload)
            return io.BytesIO(json.dumps({'model': 'test-only-response', 'choices': [
                {'message': {'content': completion.content}, 'finish_reason': 'stop'}],
                'usage': {'prompt_tokens': 10, 'completion_tokens': 5}}).encode())
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()), \
                patch('experiments.retrieval_probe.DeepSeekBackend.from_env', return_value=DeepSeekBackend('test-key')), \
                patch('litreview.clients.urllib.request.build_opener') as opener:
            opener.return_value.open.side_effect = respond
            output = Path(folder) / 'probe'
            self.assertEqual(main(['--live', '--output', str(output)]), 0)
            for rid in ('R005', 'R001'):
                ep = json.loads((output / (rid + '-collection.json')).read_text(encoding='utf-8'))
                self.assertTrue(ep['success'])
                self.assertIn('next_mask', ep['transitions'][0])
                self.assertEqual(ep['retrieval']['spec']['method'], 'bilingual_bm25_v1')
            for payload in requests:
                self.assertNotIn('seed_ids', payload)
                self.assertNotIn('criteria', payload)


if __name__ == '__main__':
    unittest.main()
