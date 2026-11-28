"""仅在测试中模拟正式数据接口；题目、论文、用量与模型响应均为合成数据。"""
from unittest.mock import patch

from agents.executor import MockBackend
from core.runner import Runner
from core.schema import ReviewTask
from evaluation.judge import JUDGE_MODEL, PROMPT_ID
from experiments.collect import collect_episode, json_sha256, paper_index
from litreview.clients import DeepSeekBackend


class Metered(MockBackend):
    def complete(self, role, instruction, payload):
        c = super().complete(role, instruction, payload)
        c.prompt_tokens, c.completion_tokens = 12, 8
        return c


def formal_inputs(*extra_tasks):
    task = ReviewTask('R', 'topic', split='train')
    papers = [{'id': 'A', 'title': 'Topic', 'abstract': 'A hypothetical paper.'}]
    corpus = {'papers': papers, 'purpose': 'formal_experiment', 'dataset_kind': 'arxiv_abstracts'}
    runner = Runner(corpus, DeepSeekBackend(api_key='fixture', model='deepseek-flash'))
    manifest = {'config_version': 1, 'source_id': 'fixture-manifest', 'dataset_kind': 'arxiv_abstracts',
                'data_purpose': 'formal_experiment', 'papers': paper_index(papers),
                'tasks': {t.review_id: {'split': t.split, 'task_sha256': json_sha256(t.to_dict())}
                          for t in (task, *extra_tasks)},
                'judge': {'model': JUDGE_MODEL, 'prompt_id': PROMPT_ID}}
    return runner, corpus, task, manifest


def synthetic_training_pair():
    holdout = ReviewTask('fixture-holdout', 'Compare the methods in the hypothetical paper.', split='train')
    runner, _, fit, manifest = formal_inputs(holdout)
    with patch.object(DeepSeekBackend, 'complete', side_effect=Metered().complete):
        episodes = [collect_episode(runner, task, policy='fast', manifest=manifest) for task in (fit, holdout)]
    return episodes, manifest
