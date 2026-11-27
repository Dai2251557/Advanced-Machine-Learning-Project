"""Train-only eight-paper evidence benchmark; never reads sealed test material."""

import argparse
import hashlib
import json
from pathlib import Path

from agents.retrieval import expand_query, tokens
from core.v2_evidence import EvidenceBackend, EvidenceRunner
from utils.data_release import ROOT, load_corpus, read_json


TASK_FILE = ROOT / 'data/v2_benchmark_tasks.json'
SECTION_FILE = ROOT / 'data/v2_benchmark_sections.json'
ROUTES = ('direct', 'fixed_multi', 'gap_rule')


def load_benchmark():
    spec, release = read_json(TASK_FILE), read_json(SECTION_FILE)
    if spec['kind'] != 'train_pilot_tasks_not_formal_test' or release['kind'] != 'manually_selected_train_benchmark_v2':
        raise ValueError('Wrong benchmark release.')
    full = load_corpus()
    index = {p['id']: p for p in full['papers']}
    ids = spec['corpus_paper_ids']
    if len(ids) != 8 or len(set(ids)) != 8 or not set(ids) <= index.keys():
        raise ValueError('Benchmark requires eight known papers.')
    corpus = {'papers': [index[pid] for pid in ids]}
    sections = {}
    for row in release['sections']:
        pid, sid = row['paper_id'], row['section_id']
        if (pid not in ids or sid in {s['section_id'] for s in sections.values()}
                or row['source_url'] != 'https://arxiv.org/pdf/' + pid
                or hashlib.sha256(row['text'].encode()).hexdigest() != row['sha256']
                or hashlib.sha256(index[pid]['abstract'].encode()).hexdigest() != row['abstract_sha256']
                or not row['source_locator'] or row['page'] < 1):
            raise ValueError('Section provenance mismatch.')
        sections[(pid, sid)] = row
    if len(sections) != 24 or any(sum(s['paper_id'] == pid for s in sections.values()) != 3 for pid in ids):
        raise ValueError('Benchmark requires three sections per paper.')
    tasks = spec['tasks']
    if len(tasks) != 12 or len({t['review_id'] for t in tasks}) != 12:
        raise ValueError('Benchmark requires twelve unique tasks.')
    for t in tasks:
        if (t['split'] != 'train' or not t['survey_instruction'] or not t['checklist']
                or not set(t['required_paper_ids']) <= set(ids)
                or not set(t['source_section_ids']) <= {s['section_id'] for s in sections.values()}):
            raise ValueError('Invalid training-side task and evidence.')
    return corpus, sections, tasks


def available_sections(runner):
    selected = {p['id'] for p in runner.selected}
    opened = {s['section_id'] for s in runner.opened}
    query = set(expand_query(runner.task['survey_instruction'] + ' ' + (runner.gap or '')))
    candidates = [s for s in runner.sections.values() if s['paper_id'] in selected and s['section_id'] not in opened]
    return sorted(candidates, key=lambda s: (-len(query & set(tokens(' '.join(s['search_terms']) + ' ' + s['heading']))),
                                             s['paper_id'], s['section_id']))


def route_actions(runner, route):
    if route not in ROUTES:
        raise ValueError('Unknown route.')
    runner.step('retrieve_more')
    if route == 'direct' or runner.terminal_reason:
        if not runner.terminal_reason:
            runner.step('write')
        return runner.result()
    if route == 'fixed_multi':
        runner.step('retrieve_more')
    else:
        instruction = runner.task['survey_instruction']
        if any(word in instruction for word in ('多少', '幅度', '实验', '比较', '对比', '局限')):
            runner.step('rewrite_query')
            if not runner.terminal_reason:
                runner.step('retrieve_more')
    for _ in range(2):
        if runner.terminal_reason or runner.tokens_used >= runner.token_budget - 1800:
            break
        sections = available_sections(runner)
        if not sections:
            break
        runner.step('open_section', sections[0]['section_id'])
    if not runner.terminal_reason and runner.valid_actions()[-1]:
        runner.step('write')
    return runner.result()


def run(output, repeats=3, backend=None):
    corpus, sections, tasks = load_benchmark()
    if type(repeats) is not int or repeats < 1:
        raise ValueError('Positive repeat count required.')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    backend = backend or EvidenceBackend.from_env()
    rows = []
    for task in tasks:
        for route in ROUTES:
            for repeat in range(repeats):
                runner = EvidenceRunner(corpus, task, sections, backend, batch_size=2, max_steps=8,
                                        token_budget=4500, read_on_open=True)
                result = route_actions(runner, route)
                path = output / f"{task['review_id']}--{route}--{repeat}.json"
                path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
                row = {'task': task['review_id'], 'route': route, 'repeat': repeat,
                       'tokens': result['total_workflow_tokens'], 'success': result['terminal_reason'] == 'final_written',
                       'actions': result['actions'], 'opened': [s['section_id'] for s in result['opened_sections']]}
                rows.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
    (output / 'index.json').write_text(json.dumps({'kind': 'v2_train_pilot_not_formal', 'rows': rows},
                                                  ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    run(args.output, args.repeats)


if __name__ == '__main__':
    main()
