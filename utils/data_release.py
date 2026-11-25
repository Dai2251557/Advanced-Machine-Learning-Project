"""读取冻结的数据交付包。默认只验证公开材料，不生成清单、不调用 Judge。"""

import argparse
import hashlib
import json
import re
from collections import Counter
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_FIELDS = {'config_version', 'review_id', 'split', 'difficulty', 'corpus_scope', 'survey_instruction'}
TRAIN_FILES = ('curation/train_sources.json', 'evaluation/train_criteria.json', 'curation/train_audit.json')
PRIVATE_FILES = ('.local-data/sealed/test_sources.json', '.local-data/sealed/test_criteria.json')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()


def require(ok, problem):
    if not ok:
        raise ValueError(problem)


def load_corpus(root=ROOT):
    rows = read_json(root / 'data/arxiv_corpus.json')
    return {'dataset_kind': 'arxiv_abstracts', 'purpose': 'formal_experiment', 'papers': [
        {'id': row['paper_id'], 'title': row['title'], 'abstract': row['abstract']} for row in rows]}


def load_manifest(split, root=ROOT):
    """从数据负责人预先冻结的清单分发；不在运行时由候选任务或轨迹自签。"""
    require(split in ('train', 'test'), '未知的切分')
    release = read_json(root / 'data/release.json')
    common = {k: release[k] for k in ('config_version', 'source_id', 'dataset_kind', 'data_purpose', 'papers')}
    common['tasks'] = {k: v for k, v in release['tasks'].items() if v['split'] == split}
    common['judge'] = release['judge']
    return common


def check_public(rows, split, expected_count):
    require(len(rows) == expected_count, f'{split} 任务数量不符')
    ids = set()
    for row in rows:
        require(set(row) == PUBLIC_FIELDS, f'{split} 任务存在非公开字段或缺字段')
        require(row['config_version'] == 1 and type(row['config_version']) is int and
                row['split'] == split and row['corpus_scope'] == 'local_arxiv_ai', '任务版本、切分或范围不符')
        require(row['difficulty'] in ('easy', 'medium', 'hard') and row['survey_instruction'].strip(), '任务缺难度或题面')
        require(not re.search(r'\b\d{4}\.\d{4,5}(?:v\d+)?\b', row['survey_instruction']), '公开题面中出现参考 ID')
        rid = row['review_id']
        require(isinstance(rid, str) and rid and rid not in ids, '任务 ID 为空或重复')
        ids.add(rid)
    return ids


def by_task(rows, ids):
    indexed = {r['review_id']: r for r in rows}
    require(len(indexed) == len(rows) and set(indexed) == ids, '任务与配套材料不是一一对应')
    return indexed


def check_evidence(sources, criteria, ids, papers):
    """检查引用可追溯；语义支持和题面覆盖仍须逐题审查。"""
    sources, criteria = by_task(sources, ids), by_task(criteria, ids)
    for rid in ids:
        seeds = sources[rid]['seed_ids']
        require(seeds and len(set(seeds)) == len(seeds) and set(seeds) <= papers.keys(), '出题来源悬空或重复')
        points = criteria[rid]['criteria']
        require(points and len({p['aspect'] for p in points}) == len(points), '评分要点为空或重复')
        for point in points:
            require(point['aspect'].strip() and point['evidence'], '要点缺说明或证据')
            for item in point['evidence']:
                pid, quote = item['paper_id'], item['quote']
                require(pid in papers and quote.strip() and quote in papers[pid]['abstract'], '证据不是对应摘要的连续原文')


def coverage(groups, first74):
    sets = [set(group) for group in groups]
    counts = Counter(pid for group in sets for pid in group)
    return {'tasks': len(sets), 'unique_papers': len(counts), 'assignments': sum(counts.values()),
            'after_first_74': len(counts.keys() - first74), 'reused_papers': sum(n > 1 for n in counts.values()),
            'pairs_sharing_two': sum(len(a & b) >= 2 for a, b in combinations(sets, 2)),
            'nested_pairs': sum(a < b or b < a for a, b in combinations(sets, 2))}


def validate(root=ROOT, with_private=False):
    release = read_json(root / 'data/release.json')
    from evaluation.judge import JUDGE_MODEL, PROMPT_ID
    require(release['config_version'] == 1 and release['source_id'] and
            release['judge'] == {'model': JUDGE_MODEL, 'prompt_id': PROMPT_ID}, '交付或 Judge 标识与冻结配置不符')
    raw = read_json(root / 'data/arxiv_corpus.json')
    require(digest(raw) == release['raw_corpus_sha256'], '原始语料与冻结快照不符')
    for start in range(0, len(raw), 100):
        chunk = root / f'data/corpus_chunk/arxiv_corpus_{start}_{start + 100}.json'
        require(read_json(chunk) == raw[start:start + 100], '语料分片与主文件不符')
    corpus = load_corpus(root)
    papers = {p['id']: p for p in corpus['papers']}
    require(len(papers) == len(corpus['papers']) == 600 and all(all(isinstance(v, str) and v.strip() for v in p.values())
            for p in papers.values()), '论文数量、唯一性或必填字段不符')
    require(release['dataset_kind'] == corpus['dataset_kind'] and release['data_purpose'] == corpus['purpose'] and
            {pid: digest(p) for pid, p in papers.items()} == release['papers'], '论文内容与冻结清单不符')
    require(set(release['curation_digests']) == set(TRAIN_FILES + PRIVATE_FILES), '缺失质检材料摘要')
    for path in TRAIN_FILES + (PRIVATE_FILES if with_private else ()):
        require(digest(read_json(root / path)) == release['curation_digests'][path], '质检材料与冻结摘要不符：' + path)
    tasks = {}
    all_ids, instructions = set(), set()
    summary = {}
    for split, count in [('train', 30), ('test', 10)]:
        rows = read_json(root / f'data/{split}_tasks.json')
        ids = check_public(rows, split, count)
        require(not all_ids.intersection(ids), '跨切分 ID 重叠')
        all_ids.update(ids)
        for row in rows:
            text = ''.join(row['survey_instruction'].lower().split())
            require(text not in instructions, '存在完全重复的题面')
            instructions.add(text)
            tasks[row['review_id']] = {'split': split, 'task_sha256': digest(row)}
        summary[split] = dict(Counter(row['difficulty'] for row in rows))
        if split == 'train' or with_private:
            paths = TRAIN_FILES if split == 'train' else PRIVATE_FILES
            check_evidence(read_json(root / paths[0]), read_json(root / paths[1]), ids, papers)
        if split == 'train':
            audit = read_json(root / TRAIN_FILES[2])
            by_task(audit, ids)
            require({r['original_id'] for r in audit} == {f'R{i:03}' for i in range(1,31)} and
                    all(r['action'] in ('keep','revise','replace') and r['audit_note'].strip() for r in audit), '旧题审查缺项')
            starter = release['calibration_review_ids']
            require(len(starter) == len(set(starter)) == 10 and set(starter) <= ids and
                    {r['difficulty'] for r in rows if r['review_id'] in starter} == {'easy', 'medium', 'hard'}, '首批标定题不合格')
    require(tasks == release['tasks'], '任务集与冻结清单不符')
    first74 = set(list(papers)[:74])
    groups = [r['seed_ids'] for r in read_json(root / TRAIN_FILES[0])]
    stats = {'old_30_train': coverage([r['original_seed_ids'] for r in audit], first74),
             'new_30_train': coverage(groups, first74)}
    if with_private:
        test = [r['seed_ids'] for r in read_json(root / PRIVATE_FILES[0])]
        stats['full_40'] = coverage(groups + test, first74)
        stats['cross_split_shared_seed_papers'] = len(set().union(*groups) & set().union(*test))
    return {'papers': len(papers), 'difficulty': summary, 'source_coverage': stats,
            'sealed_evidence_checked': with_private}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--with-private', action='store_true', help='数据持有人本地核对封存的测试证据，不输出内容')
    print(json.dumps(validate(with_private=parser.parse_args().with_private), ensure_ascii=False, indent=2))
