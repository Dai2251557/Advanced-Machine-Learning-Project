"""Train-only retrieval diagnosis and small authorized real Runner probes."""

import argparse
import json
import re
from pathlib import Path

from agents.retrieval import BilingualRetriever
from core.runner import Runner
from core.schema import ReviewTask, RunnerConfig
from experiments.collect import collect_episode, validate_episode
from litreview.clients import DeepSeekBackend
from utils.data_release import ROOT, load_corpus, load_manifest, read_json


def evaluate():
    corpus = load_corpus()
    tasks = read_json(ROOT / 'data/train_tasks.json')
    # Diagnostic references are never passed to the ranker/Runner.
    references = {r['review_id']: set(r['seed_ids']) for r in read_json(ROOT / 'curation/train_sources.json')}
    ranker = BilingualRetriever(corpus['papers'])
    rows = []
    for task in tasks:
        query = task['survey_instruction']
        terms = set(re.findall(r'[a-z0-9]+|[\u4e00-\u9fff]', query.lower()))
        def legacy(p):
            words = set(re.findall(r'[a-z0-9]+|[\u4e00-\u9fff]', (p['title'] + ' ' + p['abstract']).lower()))
            return (-len(terms & words), p['id'])
        old = [p['id'] for p in sorted(corpus['papers'], key=legacy)]
        new = [r['paper_id'] for r in ranker.rank(query)]
        refs = references[task['review_id']]
        row = {'review_id': task['review_id'], 'reference_count': len(refs), 'old_top9': old[:9], 'new_top9': new[:9]}
        for label, ranked in [('old', old), ('new', new)]:
            for k in (3, 6, 9):
                row[f'{label}_recall_at_{k}'] = len(set(ranked[:k]) & refs) / len(refs)
        rows.append(row)
    summary = {}
    for label in ('old', 'new'):
        for k in (3, 6, 9):
            key = f'{label}_recall_at_{k}'
            summary[key] = sum(r[key] for r in rows) / len(rows)
            summary[f'{label}_hit_tasks_at_{k}'] = sum(r[key] > 0 for r in rows)
    return {'split': 'train', 'tasks': len(rows), 'retriever': ranker.spec(), 'summary': summary,
            'note': 'Training development diagnostic against seed papers, not exhaustive relevance labels or independent test quality.',
            'rows': rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--live', action='store_true', help='Send training tasks/selected sources to official DeepSeek')
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    if not args.live:
        report = evaluate()
        (args.output / 'retrieval.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(report['summary']), flush=True)
        return 0
    tasks = {r['review_id']: r for r in read_json(ROOT / 'data/train_tasks.json')}
    corpus, manifest = load_corpus(), load_manifest('train')
    config = RunnerConfig(**read_json(ROOT / 'config.json'))
    # Preselected from role-3 starter set: one focused and one comparative task.
    results = []
    for task_id in ('R005', 'R001'):
        runner = Runner(corpus, DeepSeekBackend.from_env(), config, args.output / 'traces')
        episode = collect_episode(runner, ReviewTask.from_dict(tasks[task_id]), 'rule', mode='formal', manifest=manifest)
        path = args.output / f'{task_id}-collection.json'
        path.write_text(json.dumps(episode, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        validate_episode(episode, mode='formal', manifest=manifest)
        result = {k: episode[k] for k in ('review_id', 'trajectory_id', 'success', 'terminal_reason', 'total_workflow_tokens')}
        results.append(result)
        print(json.dumps(result), flush=True)
        if episode['terminal_reason'] == 'usage_unknown':
            break  # Do not continue paid calls after an unmetered transport failure.
    (args.output / 'summary.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    return 0 if len(results) == 2 and all(r['success'] for r in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
