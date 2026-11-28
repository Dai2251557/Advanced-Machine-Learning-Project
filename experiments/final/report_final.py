"""从原始封存的 execution/ 和 judge/ 重算统计表；不调用模型，公开 CSV 不是封存输入。

默认输入 outputs/course-final，--batch-dir 可指定只读归档。--early-from 则只读历史
训练内仿真的消融结果目录（5 组 × 3 个种子）。只需重图请使用 plot_final。
"""
import argparse
import csv
import hashlib
import json
import math
from itertools import combinations
from pathlib import Path
from statistics import fmean, stdev

from core.judge_result import SCORE_KEYS, validate_archived_assessment
from core.schema import ACTIONS
from evaluation.judge import JUDGE_MODEL, PROMPT_ID
from evaluation.mismatch import CATEGORIES, summarize_mismatch

from . import BATCH, CHECKPOINT_SHA256, POLICIES, new_output

LAMBDAS = (0, .25, .5, 1)
BUDGET = 48000
CONFIGS = ('base', 'buffer_small', 'buffer_large', 'target_frequent', 'target_rare')
SEEDS = (7, 17, 27)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_bytes())


def write_tables(directory, tables):
    directory.mkdir(parents=True, exist_ok=False)
    for name, rows in tables.items():
        with (directory / f'{name}.csv').open('x', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)  # CSV empty = None; genuine zero is written as 0.


def early_tables(source):
    """Exactly the first 15 evaluations, not smoothed training returns or final-test outcomes."""
    grid = list(range(100, 1501, 100))
    curves, summaries = [], []
    for config in CONFIGS:
        series = {}
        for seed in SEEDS:
            r = read(source / f'{config}-seed{seed}' / 'result.json')
            require((r['name'], r['seed'], r['data_source'], r['threshold'], r['consecutive_evaluations']) ==
                    (config, seed, 'formal_train_runner', .5, 3), f'{config}/{seed}: wrong run or threshold')
            es = [e for e in r['evaluations'] if e['step'] <= 1500]
            require([e['step'] for e in es] == grid, f'{config}/{seed}: wrong early grid')
            ys = [e['mean_return'] for e in es]
            require(all(math.isfinite(y) for y in ys), f'{config}/{seed}: invalid early return')
            series[seed] = ys
            first = next((s for s, y in zip(grid, ys) if y >= .5), None)
            confirm = next((grid[i] for i in range(2, len(ys)) if min(ys[i-2:i+1]) >= .5), None)
            def first_dip(step):
                return next((s for s, y in zip(grid, ys) if step is not None and s > step and y < .5), None)
            drops = [max(ys[:i+1]) - y for i, y in enumerate(ys)]
            worst = drops.index(max(drops))
            summaries.append(dict(config=config, seed=seed, threshold=.5, first_cross_step=first,
                confirm_step=confirm, first_dip_after_cross=first_dip(first), first_dip_after_confirm=first_dip(confirm),
                below_threshold_points=sum(y < .5 for y in ys), mean_return_100_1500=fmean(ys),
                max_drawdown=max(drops), drawdown_trough_step=grid[worst] if drops[worst] else None,
                capacity=r['config']['capacity'], target_interval=r['config']['target_interval']))
        for i, step in enumerate(grid):
            v = [series[s][i] for s in SEEDS]
            curves.append(dict(config=config, step=step, **{f'seed{s}': y for s, y in zip(SEEDS, v)},
                mean_return=fmean(v), min_return=min(v), max_return=max(v), sd_return=stdev(v)))
    return {'early': curves, 'early_summary': summaries}


def known_sum(values):
    """Known components only: no events is 0; events but no known component is unknown."""
    values = list(values)
    return sum(v for v in values if v is not None) if not values or any(v is not None for v in values) else None


def strict_sum(values):
    values = list(values)
    return None if None in values else sum(values)


def mean_known(values):
    values = [v for v in values if v is not None]
    return fmean(values) if values else None


def ratio(a, b):
    return a / b if a is not None and b else None


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def indexed(rows, key):
    out = {r[key]: r for r in rows}
    require(len(out) == len(rows), f'duplicate {key}')
    return out


def sealed(root):
    seal = read(root / 'execution_complete.json')
    require(seal['all_episodes_complete'] is True and seal['mode'] == 'formal', 'expected sealed formal execution')
    require(seal['format'] == 'frozen-three-v1' and seal['policies'] == list(POLICIES) and
            seal['checkpoints_sha256'] == CHECKPOINT_SHA256, 'not the frozen three-policy batch')
    manifest = indexed(seal['episodes'], 'trace')
    indexed(seal['episodes'], 'trajectory_id')
    require(set(manifest) == {f'{i:03d}-{p}.json' for i in range(10) for p in POLICIES}, 'expected 10 × 3 traces')
    seen = set()
    for i in range(10):
        rid = manifest[f'{i:03d}-rule.json']['review_id']
        require(rid not in seen, 'duplicate review across triplets')
        seen.add(rid)
        for p in POLICIES:
            name = f'{i:03d}-{p}.json'
            m = manifest[name]
            raw = (root / name).read_bytes()
            require(digest(raw) == m['sha256'], f'{name}: trace digest mismatch')
            ep = json.loads(raw)
            tid = ep['trajectory_id']
            require(tid and tid == m['trajectory_id'] and ep['review_id'] == rid == m['review_id'] and m['policy'] == p
                    and ep['split'] == 'test' and ep['execution_kind'] == 'real' and type(ep['success']) is bool
                    and ep['success'] is m['success'], f'{name}: wrong trace identity or status')
            steps = ep['transitions']
            require(steps and ep['step_count'] == len(steps) and steps[-1]['done'] is True and
                    steps[-1]['terminal_reason'] == ep['terminal_reason'] and ep['config']['workflow_tokens_budget'] == BUDGET,
                    f'{name}: invalid final state or budget')
            text = ep['final_text']
            require((ep['success'] and ep['terminal_reason'] == 'final_written' and isinstance(text, str) and text.strip()) or
                    (not ep['success'] and ep['terminal_reason'] != 'final_written' and text is None), f'{name}: invalid delivery')
            sha = None if text is None else digest(text.encode('utf-8'))
            require(ep.get('final_text_sha256', sha) == sha and m.get('final_text_sha256', sha) == sha,
                    f'{name}: wrong final-text digest')
            yield p, ep, sha


def judge_data(root):
    book = read(root / 'summary.json')
    require((book['model'], book['prompt_id']) == (JUDGE_MODEL, PROMPT_ID), 'not the frozen v3 Judge batch')
    jobs = indexed(book['results'], 'trajectory_id')
    labels = indexed([read(path) for path in sorted((root / 'labels').glob('trajectory-*.json'))], 'trajectory_id')
    require(set(labels) == {tid for tid, job in jobs.items() if job['status'] == 'labeled'}, 'Judge ledger/sidecar mismatch')
    for label in labels.values():
        require((label['split'], label['source_kind'], label['model'], label['prompt_id']) ==
                ('test', 'independent_judge', JUDGE_MODEL, PROMPT_ID), 'wrong Judge split/source/model/version')
        validate_archived_assessment(label)
    return jobs, labels


def bound_judge(p, ep, sha, jobs, labels):
    tid = ep['trajectory_id']
    job, label = jobs.get(tid), labels.get(tid)
    if job is not None:
        require(ep['success'] and job['review_id'] == ep['review_id'] and job['policy'] == p and
                job.get('final_text_sha256', sha) == sha and job.get('split', 'test') == 'test', f'{tid}: wrong Judge dispatch')
    if label is not None:
        require(label['review_id'] == ep['review_id'] and label['final_text_sha256'] == sha, f'{tid}: wrong Judge binding')
    status = job['status'] if job is not None else ('not_sent' if ep['success'] else 'no_manuscript')
    usage = (job or {}).get('usage') or {}
    account = {k: usage.get(k) for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')}
    require(all(n is None or (type(n) is int and n >= 0) for n in account.values()), f'{tid}: invalid Judge usage')
    return label, dict(review_id=ep['review_id'], policy=p, trajectory_id=tid, dispatched=job is not None,
                       judge_status=status, **account)


CALL_COUNTS = ('logged_calls', 'api_calls', 'cache_hits', 'verifier_calls', 'repair_calls', 'failed_calls', 'unknown_usage_calls')


def call_load(calls):
    tokens = [c[k] for c in calls for k in ('prompt_tokens', 'completion_tokens')]
    require(all(n is None or (type(n) is int and n >= 0) for n in tokens), 'invalid Runner usage')
    return dict(workflow_tokens=strict_sum(tokens), reported_workflow_tokens=known_sum(tokens), logged_calls=len(calls),
        api_calls=sum(not c['cache_hit'] for c in calls), cache_hits=sum(c['cache_hit'] for c in calls),
        verifier_calls=sum(c['role'] == 'verify' for c in calls), repair_calls=sum(c['attempt'] > 1 for c in calls),
        failed_calls=sum(c['status'] != 'ok' for c in calls),
        unknown_usage_calls=sum(c['prompt_tokens'] is None or c['completion_tokens'] is None for c in calls))


def workload(rows):
    return dict(workflow_tokens=strict_sum(r['workflow_tokens'] for r in rows),
                reported_workflow_tokens=known_sum(r['reported_workflow_tokens'] for r in rows),
                **{k: sum(r[k] for r in rows) for k in CALL_COUNTS})


def episode_row(p, ep, label, judge_status):
    tid, rid, steps = ep['trajectory_id'], ep['review_id'], ep['transitions']
    details = []
    for i, t in enumerate(steps, 1):
        mask, a = t['action_mask'], t['action_id']
        require((t['trajectory_id'], t['review_id'], t['step_index']) == (tid, rid, i) and len(mask) == len(ACTIONS)
                and all(type(b) is bool for b in mask) and mask[ACTIONS.index(a)], f'{tid}/{i}: wrong identity/action mask')
        cost = call_load(t['calls'])  # includes verifier, failed attempts and repairs; cache hits carry recorded zero.
        require(cost['workflow_tokens'] == t['total_workflow_tokens'], f'{tid}/{i}: calls do not reconcile to step tokens')
        details.append(dict(review_id=rid, policy=p, trajectory_id=tid, step=i, action=a, success=t['success'], error=t['error'],
                            **cost, **{f'can_{a}': b for a, b in zip(ACTIONS, mask)}))
    load = workload(details)
    require(load['workflow_tokens'] == ep['total_workflow_tokens'], f'{tid}: step and episode usage disagree')
    m = summarize_mismatch([steps])
    row = dict(review_id=rid, policy=p, trajectory_id=tid, overall_score=None if label is None else label['overall_score'],
        **{k: None if label is None else label['subscores'][k] for k in SCORE_KEYS}, **load,
        delivered=ep['success'], terminal_reason=ep['terminal_reason'], judge_status=judge_status,
        steps=len(steps), action_path=' > '.join(t['action_id'] for t in steps),
        step_failures=m['execution_failure_count'], mismatch_count=m['any_count'], mismatch_known_steps=m['known_steps'],
        mismatch_unknown_steps=m['unknown_steps'])
    for category, counters in m['by_category'].items():
        row.update({f'{category}_{k}': v for k, v in counters.items()})
    return row, details


SCORES = ('overall_score', *SCORE_KEYS)


def comparisons(tasks):
    """Paired estimands only. U at zero penalty does not depend on cost being observed."""
    wide, utility, summary, pairs = [], [], [], []
    for rid, group in tasks.items():
        wide.append({'review_id': rid, **{f'{p}_{k}': group[p][k] for p in POLICIES for k in
            ('overall_score', 'coverage', 'workflow_tokens', 'reported_workflow_tokens', 'action_path', 'terminal_reason', 'judge_status')}})
    def paired(metric, lam, group, higher):
        for a, b in combinations(POLICIES, 2):
            obs = [(rid, v[a] - v[b]) for rid, v in group.items() if v[a] is not None and v[b] is not None]
            # Exact cents for frozen quality; tolerance only removes float noise in mathematically tied utilities.
            diffs = [0 if abs(d) < 1e-12 else d for _, d in obs]
            signed = [d if higher else -d for d in diffs]
            pairs.append(dict(metric=metric, **{'lambda': lam}, policy_a=a, policy_b=b, higher_is_better=higher,
                n_paired=len(diffs), wins_a=sum(d > 0 for d in signed), ties=signed.count(0), losses_a=sum(d < 0 for d in signed),
                mean_delta_a_minus_b=mean_known(diffs), paired_review_ids='|'.join(rid for rid, _ in obs)))
    for k in (*SCORES, 'workflow_tokens'):
        paired(k, None, {rid: {p: row[k] for p, row in group.items()} for rid, group in tasks.items()}, k != 'workflow_tokens')
    for lam in LAMBDAS:
        common = {}
        for rid, group in tasks.items():
            vals = {}
            for p, row in group.items():
                q, t = row['overall_score'], row['workflow_tokens']
                vals[p] = q if lam == 0 else (None if q is None or t is None else q - lam * t / BUDGET)
            ok = None not in vals.values()
            if ok:
                common[rid] = vals
            for p in POLICIES:
                utility.append(dict(review_id=rid, policy=p, **{'lambda': lam}, overall_score=group[p]['overall_score'],
                    workflow_tokens=group[p]['workflow_tokens'], utility=vals[p], in_common=ok))
        for p in POLICIES:
            summary.append(dict(**{'lambda': lam}, policy=p, n_common=len(common), mean_utility=mean_known(v[p] for v in common.values()),
                                common_review_ids='|'.join(common)))
        paired('utility', lam, common, True)  # same denominator as the three-policy main table
    return {'task_wide': wide, 'pairwise': pairs, 'utility_per_task': utility, 'utility_summary': summary}


def final_tables(root):
    jobs, labels = judge_data(root / 'judge')
    rows, steps, judge_rows, tasks = [], [], [], {}
    for p, ep, sha in sealed(root / 'execution'):
        label, j = bound_judge(p, ep, sha, jobs, labels)
        row, details = episode_row(p, ep, label, j['judge_status'])
        rows.append(row); steps.extend(details); judge_rows.append(j)
        tasks.setdefault(row['review_id'], {})[p] = row
    require(jobs.keys() <= {r['trajectory_id'] for r in rows}, 'orphan Judge dispatch')
    common_quality = [rid for rid, group in tasks.items() if all(r['overall_score'] is not None for r in group.values())]
    common_cost = [rid for rid, group in tasks.items() if all(r['workflow_tokens'] is not None for r in group.values())]
    overview, actions, judge_summary = [], [], []
    for p in POLICIES:
        group = [r for r in rows if r['policy'] == p]
        ts = [r for r in steps if r['policy'] == p]
        load = workload(group)
        n = len(group)
        delivered = sum(r['delivered'] for r in group)
        s = dict(policy=p, n_tasks=n, n_delivered=delivered, n_failed=n-delivered, completion_rate=delivered/n,
            n_scored=sum(r['overall_score'] is not None for r in group), n_known_cost=sum(r['workflow_tokens'] is not None for r in group),
            n_common_scored=len(common_quality), n_common_cost=len(common_cost), steps=len(ts), mean_steps=len(ts)/n, **load,
            mean_workflow_tokens=mean_known(r['workflow_tokens'] for r in group),
            mean_common_workflow_tokens=mean_known(tasks[rid][p]['workflow_tokens'] for rid in common_cost))
        for k in SCORES:
            s[f'mean_{k}'] = mean_known(r[k] for r in group)
            s[f'mean_common_{k}'] = mean_known(tasks[rid][p][k] for rid in common_quality)
        for k in ('mismatch_count', 'mismatch_known_steps', 'mismatch_unknown_steps', 'step_failures',
                  *(f'{c}_{k}' for c in CATEGORIES for k in ('count', 'applicable_steps', 'known_steps', 'unknown_steps'))):
            s[k] = sum(r[k] for r in group)
        s['mismatch_per_episode'] = s['mismatch_count'] / n
        overview.append(s)
        for a in ACTIONS:
            chosen = [t for t in ts if t['action'] == a]
            opp = sum(t[f'can_{a}'] for t in ts)
            cost = workload(chosen)
            actions.append(dict(policy=p, action=a, selections=len(chosen), opportunities=opp, selection_rate=ratio(len(chosen), opp),
                step_share=len(chosen)/len(ts), **cost, token_share=ratio(cost['workflow_tokens'], load['workflow_tokens']),
                reported_token_share=ratio(cost['reported_workflow_tokens'], load['reported_workflow_tokens'])))
    for p in (*POLICIES, 'all'):
        js = [j for j in judge_rows if j['dispatched'] and p in (j['policy'], 'all')]
        item = dict(policy=p, dispatched=len(js), n_labeled=sum(j['judge_status'] == 'labeled' for j in js))
        for k in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
            xs = [j[k] for j in js]
            item.update({k: strict_sum(xs), f'reported_{k}': known_sum(xs), f'unknown_{k}_calls': xs.count(None)})
        judge_summary.append(item)
    return {'per_task': rows, 'overview': overview, **comparisons(tasks), 'actions': actions, 'steps': steps,
            'judge_per_task': judge_rows, 'judge_summary': judge_summary, 'definitions': DEFINITIONS}


DEFINITIONS = [dict(field=k, definition=v) for k, v in (
    ('missing', 'CSV 空白=未观测或无分母，不是0。工作流失败、无终稿、Judge失败均不补质量分。'),
    ('overview', 'n_tasks=每策略全部10题；交付率=final_written数/10，与是否有Judge真分独立。'),
    ('mean_common_*', '三方比较用mean_common_*：质量均值只取三方同题均有分，成本均值只取三方同题均有成本，见对应n_common列。其余mean_*为各自有效子集的描述性均值，不可直接相减。'),
    ('pairwise', '每个指标用A、B同题均有效的分母和逐题A−B原量纲差，不是各自均值之差。成本少为胜，可含未交付题；不是质量优势。utility的成对分母与同λ三方主表一致。'),
    ('utility', 'U=独立v3总分−λ×workflow_tokens/48000。λ=0只需真分；λ>0还需完整工作流成本。主表只取in_common=True即三策略同题全有效；不生成带失败惩罚的伪质量。'),
    ('workflow_tokens', '逐调用prompt+completion（含verifier、失败与修复），任一分量缺失则总数未知。reported_workflow_tokens是已报告分量的小计，不用来估算效用。'),
    ('calls', 'logged_calls含缓存命中；api_calls=非缓存事件。repair_calls是attempt>1的事件，failed_calls是status!=ok；两者可重叠，与策略动作/失败步数不同。'),
    ('actions', 'selection_rate=选中步数/该动作mask可用步数。token_share=归该动作的完整工作流成本/该策略全部成本，分母缺失时为空。reported_token_share仅指已报告分量。'),
    ('mismatch', 'evaluation.mismatch.summarize_mismatch，按步去重。分类可能重叠；每局=已确认错配步/全部10局；unknown不是无错配。premature_write是过程代理，不是Judge质量。'),
    ('judge', 'Judge使用独立账本，不计入工作流或效用。judge_per_task每轨迹一行；未发请求与已发但无usage分别标记。'),
)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    inputs = p.add_mutually_exclusive_group()
    inputs.add_argument('--batch-dir', type=Path, help='同时含 execution/ 和 judge/ 的封存批次，默认 outputs/course-final')
    inputs.add_argument('--early-from', type=Path, metavar='ABLATION_DIR', help='仅重算早期表：显式提供含 15 份 result.json 的训练内目录')
    p.add_argument('--out-dir', type=Path, help='outputs/ 下的新目录，默认 outputs/course-final/report（早期表为 early）')
    args = p.parse_args()
    output = new_output(args.out_dir or BATCH / ('early' if args.early_from else 'report'))
    if args.early_from:
        tables = early_tables(args.early_from)
    else:
        batch = args.batch_dir or BATCH
        for path in (batch / 'execution/execution_complete.json', batch / 'judge/summary.json'):
            if not path.is_file():
                p.error(f'缺少原始封存输入 {path}；请用 --batch-dir 指向归档，或用 plot_final 从公开 CSV 重图')
        tables = final_tables(batch)
    write_tables(output, tables)
    print(f'{output}: ' + ', '.join(f'{name}={len(rows)}' for name, rows in tables.items()))


if __name__ == '__main__':
    main()
