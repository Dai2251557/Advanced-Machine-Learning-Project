"""Collect *training* transitions, never judge them. CLI is synthetic/offline only.

Wire protocol: an episode is the unwrapped Runner.result() with next_mask on each
transition, collection_kind ('formal' or 'fixture'), public_task, policy/plan,
source (source_id and provenance digests), final_text_sha256 and support added.
Only collect_episode accepts a configured live Runner. Neither import nor the CLI
constructs a live backend. Runner's optional on-disk traces are unchanged; save
this function's RETURN value to retain the masks/metadata.

The independent, trusted formal manifest is supplied out-of-band by the data
owner, NOT derived from a candidate episode. JSON shape:
  {config_version: 1, source_id: 'delivery/version', dataset_kind: 'arxiv_abstracts',
   data_purpose: 'formal_experiment', papers: {paper_id: paper_sha256, ...},
   tasks: {review_id: {split: 'train'|'test', task_sha256: hex}, ...},
   judge: {model: frozen_model_id, prompt_id: frozen_rubric_prompt_id}}
Each paper hash covers exactly {id,title,abstract}, each task hash the complete
public ReviewTask.to_dict(). json_sha256 uses UTF-8 canonical compact sorted
JSON, no NaN. The corpus digest covers the entire paper-ID/hash mapping. This
checks consistency with a trusted delivery, not authorship or API attestation.

validate_collection returns an external trajectory_id -> label map; absent labels
stay absent. Each label has source_kind ('independent_judge'|'synthetic_fixture'),
review_id, trajectory_id, split, final_text_sha256 (raw UTF-8), overall_score,
model and prompt_id. Formal labels must also satisfy the frozen assessment and
request fingerprint contract. No Judge is invoked. Keep the sidecar and any
Judge usage outside workflow traces.
"""

import argparse
import hashlib
import json
import math
from collections import Counter
from functools import wraps
from pathlib import Path

from agents.executor import MockBackend
from core.judge_result import validate_archived_assessment
from core.rule_scheduler import RuleScheduler
from core.runner import Runner, total
from core.schema import (ACTIONS, CONFIG_VERSION, SCORE_KEYS, ReviewTask, RunnerConfig,
                         comparable_delta, empty_verifier, verifier_value)
from litreview.clients import DeepSeekBackend
from litreview.pipeline import validate_ids
from rl.adapter import step_with_masks

# Planned alternatives are preferences, never new physical masks. Index by the
# number of completed decisions. RuleScheduler handles unavailable/exhausted plans.
PLANS = {
    'rule': (),
    'fast': ('retrieve_more', 'summarize_fast', 'write_fast'),
    'deep': ('retrieve_more', 'summarize_deep', 'write_deep'),
    'raw_fast': ('retrieve_more', 'write_fast'),
    'raw_deep': ('retrieve_more', 'write_deep'),
    'retrieve_again': ('retrieve_more', 'summarize_fast', 'retrieve_more', 'summarize_deep', 'write_fast'),
    'revise': ('retrieve_more', 'summarize_deep', 'summarize_fast', 'write_deep'),
}
# 配对专用计划显式可选，不改变已有 all / DQN 夹具的采样组成。
REGISTERED_PLANS = {
    **PLANS,
    'raw_retrieve_fast': ('retrieve_more', 'retrieve_more', 'summarize_fast', 'write_fast'),
    'revise_fast_deep': ('retrieve_more', 'summarize_fast', 'summarize_deep', 'write_fast'),
    'R-Sd-Wf': ('retrieve_more', 'summarize_deep', 'write_fast'),
    'R-R-Sd-Wf': ('retrieve_more', 'retrieve_more', 'summarize_deep', 'write_fast'),
    'R-R-R-Sd-Wd': ('retrieve_more', 'retrieve_more', 'retrieve_more', 'summarize_deep', 'write_deep'),
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _wire(fn):
    """Missing/malformed JSON is an invalid artifact, not an internal KeyError."""
    @wraps(fn)
    def validate(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise ValueError(f'{fn.__name__}: malformed wire data ({exc})') from exc
    return validate


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _int(value):
    return type(value) is int and value >= 0


def _score(value):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def _hex(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def json_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                                     separators=(',', ':')).encode('utf-8')).hexdigest()


def final_text_sha256(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def paper_index(papers):
    """Build the projected ID -> digest index, ignoring unused arXiv metadata."""
    index = {}
    for row in papers:
        paper = {k: row[k] for k in ('id', 'title', 'abstract')}
        _require(all(map(_text, paper.values())) and paper['id'] not in index, 'invalid or duplicate source paper')
        index[paper['id']] = json_sha256(paper)
    return index


def _task(value):
    if isinstance(value, ReviewTask):
        value = value.to_dict()
    _require(set(value) == set(ReviewTask.__dataclass_fields__), 'expected only complete public ReviewTask fields')
    task = ReviewTask(**value)
    _require(task.split == 'train' and type(task.config_version) is int, 'training collection requires split=train')
    return task


def _formal_manifest(manifest):
    _require(isinstance(manifest, dict), 'trusted external formal manifest required')
    _require(type(manifest['config_version']) is int and manifest['config_version'] == CONFIG_VERSION
             and _text(manifest['source_id']), 'bad source delivery/version')
    _require((manifest['dataset_kind'], manifest['data_purpose']) == ('arxiv_abstracts', 'formal_experiment'),
             'formal arXiv training delivery required')
    _require(isinstance(manifest['papers'], dict) and bool(manifest['papers']) and
             all(_text(k) and _hex(v) for k, v in manifest['papers'].items()), 'invalid formal corpus allowlist')
    _require(isinstance(manifest['tasks'], dict) and bool(manifest['tasks']) and all(
        _text(k) and row['split'] in ('train', 'test') and _hex(row['task_sha256'])
        for k, row in manifest['tasks'].items()), 'invalid formal task allowlist')
    _require(all(_text(manifest['judge'][k]) for k in ('model', 'prompt_id')), 'frozen Judge identifiers required')
    return manifest


def _source(task, kind, purpose, index, mode, manifest):
    _require(mode in ('formal', 'fixture'), 'expected formal or fixture collection')
    if mode == 'fixture':
        _require(manifest is None and (kind, purpose) == ('synthetic_fixture', 'dry_run_only'),
                 'fixture inputs must be explicitly synthetic, not pilot/test data')
        source_id, digest = 'local/synthetic-fixture', None
    else:
        m = _formal_manifest(manifest)
        _require((kind, purpose) == (m['dataset_kind'], m['data_purpose']), 'unapproved corpus kind or purpose')
        _require(index == m['papers'], 'complete Runner corpus does not match trusted paper allowlist')
        entry = m['tasks'].get(task.review_id)
        _require(entry == {'split': 'train', 'task_sha256': json_sha256(task.to_dict())},
                 'unapproved task, split or public task contents')
        source_id, digest = m['source_id'], json_sha256(m)
    return {'source_id': source_id, 'manifest_sha256': digest, 'task_sha256': json_sha256(task.to_dict()),
            'corpus_sha256': json_sha256(index), 'corpus_size': len(index)}


def sampling_plan(policy):
    _require(policy in REGISTERED_PLANS, 'unknown sampling policy')
    return {'policy': policy, 'preferences_by_step': list(REGISTERED_PLANS[policy]), 'fallback': 'RuleScheduler',
            'selection': 'deterministic'}


def _choose(rule, plan, state, mask):
    i = state['step_count']
    if i < len(plan):
        preferred = plan[i]
        if mask[ACTIONS.index(preferred)]:
            return preferred
    return rule.select_action(state, mask)


def collection_inputs(runner, task, mode, manifest):
    task = _task(task)
    _require(isinstance(runner, Runner), 'a configured shared Runner is required')
    if mode == 'formal':
        _require(type(runner.executor.backend) is DeepSeekBackend, 'formal collection requires configured DeepSeekBackend')
    else:
        _require(isinstance(runner.executor.backend, MockBackend) and runner.executor.backend.name == 'mock',
                 'fixture mode requires an explicit mock backend')
    source = _source(task, runner.dataset_kind, runner.data_purpose, paper_index(runner.papers), mode, manifest)
    return task, source


def finish_collection(runner, task, policy, plan, source, steps, mode, manifest):
    episode = runner.result()
    episode.update(transitions=steps, collection_kind=mode, public_task=task.to_dict(), source=source,
                   policy=policy, plan=plan, plan_sha256=json_sha256(plan), config_sha256=json_sha256(episode['config']),
                   final_text_sha256=final_text_sha256(episode['final_text']) if episode['success'] else None)
    episode['support'] = support_report([episode])
    validate_episode(episode, mode=mode, manifest=manifest)
    return episode


@_wire
def collect_episode(runner, task, policy='rule', *, mode='formal', manifest=None, seed=0):
    """Preflight sources/backend before reset or any call; return a JSON-ready episode."""
    task, source = collection_inputs(runner, task, mode, manifest)
    plan = sampling_plan(policy)
    rule = RuleScheduler(runner.config)
    rule.reset(seed=seed)
    state = runner.reset(task, seed=seed)
    steps = []
    while not state['terminal']:
        action = _choose(rule, plan['preferences_by_step'], state, runner.valid_actions())
        step = step_with_masks(runner, action)
        steps.append(step)
        state = step['state_after']
    return finish_collection(runner, task, policy, plan, source, steps, mode, manifest)


def dry_run(policy='rule', *, review_id='fixture-001', seed=0, difficulty='medium', config=None, backend=None):
    """Local synthetic train-shaped fixture; never a real training observation."""
    _require(backend is None or isinstance(backend, MockBackend), 'dry_run only accepts mock backends')
    papers = [{'id': f'fixture-{i}', 'title': f'Synthetic study {i}',
               'abstract': f'Fictional evidence for an offline interface test, example {i}.'} for i in range(7)]
    runner = Runner({'papers': papers, 'dataset_kind': 'synthetic_fixture', 'purpose': 'dry_run_only'},
                    backend, config=config)
    task = ReviewTask(review_id, 'Compare the fictional examples using only their abstracts.',
                      split='train', difficulty=difficulty, corpus_scope='synthetic_fixture')
    return collect_episode(runner, task, policy, mode='fixture', seed=seed)


def support_report(episodes):
    """Pre-action opportunity denominators; eligible-but-unselected actions retain zero support."""
    buckets = {}
    for ep in episodes:
        for step in ep['transitions']:
            s = step['state_before']
            stage = 'summary' if s['summary_available'] else 'raw' if s['retrieved_paper_count'] else 'empty'
            v = verifier_value(s)
            band = 'unavailable' if v is None else 'low' if v < .75 else 'high'
            for a, ok in zip(ACTIONS, step['action_mask']):
                key = (s['difficulty'], stage, band, a)
                n = buckets.setdefault(key, {'opportunities': 0, 'selected': 0, 'reviews': set(),
                                             'known_cost_steps': 0, 'unknown_cost_steps': 0})
                n['opportunities'] += int(ok)
                if a == step['action_id']:
                    n['selected'] += 1
                    n['reviews'].add(ep['review_id'])
                    n['unknown_cost_steps' if step['total_workflow_tokens'] is None else 'known_cost_steps'] += 1
    return [{'difficulty': d, 'stage': stage, 'verifier_band': band, 'action_id': a,
             **{k: v for k, v in n.items() if k != 'reviews'}, 'independent_reviews': len(n['reviews']),
             'selection_rate': n['selected'] / n['opportunities'] if n['opportunities'] else None}
            for (d, stage, band, a), n in sorted(buckets.items())]


# Block known independent/evaluation information even if inserted into a nested
# call, alongside exact public snapshot fields. Never open artifact-provided refs.
def _no_labels(value):
    if isinstance(value, dict):
        for k, v in value.items():
            _require(not k.lower().startswith(('judge', 'gold', 'estimated_final_quality', 'simulation_support'))
                     and k not in ('overall_score', 'evaluation_points', 'expected_paper_ids', 'terminal_reward'),
                     'hidden label or simulated evidence inside workflow trace')
            _no_labels(v)
    elif isinstance(value, list):
        for item in value:
            _no_labels(item)


def _usage(calls, action, config, used_before):
    _require(isinstance(calls, list) and len(calls) > 0, 'missing Runner call log')
    roles = {'retrieve_more': {'verify'}, 'summarize': {'summarize', 'verify'}, 'write': {'write'}}
    family = action if action == 'retrieve_more' else action.split('_')[0]
    by_role = {'input_tokens': [], 'output_tokens': [], 'verifier_tokens': []}
    previous = None
    attempts = Counter()
    for call in calls:
        role = call['role']
        _require(role in roles[family], 'unexpected or Judge call in workflow')
        if family == 'summarize':
            _require((previous is not None or role == 'summarize') and
                     (not previous or previous['role'] != 'verify' or role == 'verify') and
                     (not previous or previous['role'] == role or previous['status'] == 'ok'), 'invalid verifier ordering')
        attempts[role] += 1
        _require(call['attempt'] == attempts[role] <= config.output_repair_attempts + 1 and
                 (not previous or previous['role'] != role or previous['status'] == 'error') and
                 call['status'] in ('ok', 'error') and call['reasoning_effort'] ==
                 ('high' if action.endswith('_deep') and role != 'verify' else 'low'), 'bad effort/repair history')
        _require(call['cache_hit'] or (used_before is not None and used_before < config.workflow_tokens_budget),
                 'external call started after budget boundary')
        p, c = call['prompt_tokens'], call['completion_tokens']
        _require(all(v is None or _int(v) for v in (p, c)), 'invalid call usage')
        _require(type(call['cache_hit']) is bool and (not call['cache_hit'] or (p, c) == (0, 0)), 'bad cache billing')
        if role == 'verify':
            by_role['verifier_tokens'].extend((p, c))
        else:
            by_role['input_tokens'].append(p)
            by_role['output_tokens'].append(c)
        used_before = total((used_before, p, c))
        previous = call
    parts = {k: total(v) for k, v in by_role.items()}
    return {**parts, 'total_workflow_tokens': total(parts.values())}


def _check_state(s, task, config, index, tokens, corpus_size):
    from core.schema import WorkspaceSnapshot
    _require(set(s) == set(WorkspaceSnapshot.__annotations__), 'unexpected or missing Workspace field')
    for k, expected in {'config_version': CONFIG_VERSION, 'review_id': task.review_id, 'difficulty': task.difficulty,
                        'max_steps': config.max_steps, 'workflow_tokens_budget': config.workflow_tokens_budget,
                        'step_count': index, 'workflow_tokens_used': tokens}.items():
        _require(type(s[k]) is type(expected) and s[k] == expected, f'bad state {k}')
    _require(_int(s['retrieved_paper_count']) and s['retrieved_paper_count'] <= corpus_size and
             _int(s['material_revision']) and type(s['summary_available']) is bool and type(s['terminal']) is bool,
             'invalid public state flags/counts')
    v = s['verifier']
    _require(set(v) == {'scores', 'valid_scores', 'checked_material_revision', 'issues'} and
             set(v['scores']) == set(v['valid_scores']) == set(SCORE_KEYS) and
             all(_score(v['scores'][k]) and type(v['valid_scores'][k]) is bool for k in SCORE_KEYS)
             and type(v['checked_material_revision']) is int and -1 <= v['checked_material_revision'] <= s['material_revision']
             and isinstance(v['issues'], list) and all(map(_text, v['issues'])), 'invalid verifier snapshot')
    _require(set(s['action_history']) == set(ACTIONS), 'invalid action history')
    for h in s['action_history'].values():
        _require(set(h) == {'call_count', 'consecutive_low_gain_or_failures', 'last_verifier_delta', 'token_share'}
                 and _int(h['call_count']) and _int(h['consecutive_low_gain_or_failures']), 'invalid history fields')
    return [not s['terminal'] and (s['retrieved_paper_count'] < corpus_size if a == 'retrieve_more'
                                   else s['retrieved_paper_count'] > 0) for a in ACTIONS]


def _mask(value, expected):
    _require(isinstance(value, list) and all(type(x) is bool for x in value) and value == expected,
             'incorrect physical mask or mask alignment')


def _same_usage(actual, expected, name):
    _require(type(actual) is type(expected) and actual == expected, f'inconsistent {name} usage')


def _successful_outputs(calls):
    """只取本步各角色成功的尝试。修复及缓存命中也应记录原始响应。"""
    outputs = {}
    for call in calls:
        if call['status'] == 'error':
            _require('output' not in call, 'failed call cannot supply an accepted output')
            continue
        output = call['output']
        _require(call['finish_reason'] == 'stop' and isinstance(call['response_text'], str)
                 and isinstance(output, dict) and json_sha256(json.loads(call['response_text'])) == json_sha256(output),
                 'successful call output does not match raw response')
        # _usage 已核对角色顺序，每个角色至多一次成功。
        outputs[call['role']] = output
    return outputs


def _check_step_effects(t, outputs, budget_ok):
    before, after, action = t['state_before'], t['state_after'], t['action_id']
    retrieved = action == 'retrieve_more'
    summarized = 'summarize' in outputs
    revision = before['material_revision'] + int(retrieved or summarized)
    has_summary = not retrieved and (summarized or before['summary_available'])
    _require(after['material_revision'] == revision and after['summary_available'] is has_summary,
             'material change does not match executed action')

    # 材料更新后即使校验失败也不回滚材料，但旧校验的版本和内容必须原样保留。
    expected = before['verifier']
    if 'verify' in outputs:
        result = outputs['verify']
        expected = {k: result[k] for k in ('scores', 'valid_scores', 'issues')}
        expected['checked_material_revision'] = revision
        _require('next_action' not in result and all(result['valid_scores'][k] is (has_summary or k == 'coverage')
                 and (result['valid_scores'][k] or result['scores'][k] == 0) for k in SCORE_KEYS),
                 'invalid verifier scoring scope')
    elif (retrieved or summarized) and budget_ok:
        _require(any(c['role'] == 'verify' for c in t['calls']), 'missing verifier attempt after material update')
    _require(json_sha256(after['verifier']) == json_sha256(expected), 'verifier snapshot does not match call history')

    # Step.success 还受自动校验或写稿预算影响，并不等于动作本身执行成功。
    success = ('write' in outputs and budget_ok) if action.startswith('write_') else 'verify' in outputs
    _require(t['success'] is success, 'step success does not match call history and budget')


def _render_accepted_write(draft, papers, execution_kind):
    """仅对账 Runner.step 的发布格式；T4 入口已排除 pilot_only，不应出现 PILOT 横幅。"""
    lines = ['# 文献综述', '', '范围：仅依据提供的论文摘要。', '']
    if execution_kind == 'mock':
        lines += ['> MOCK 合成演示，不是实际模型输出或实验结果。', '']
    ids = {p['id'] for p in papers}
    cited = set()
    _require(isinstance(draft['sections'], list) and draft['sections'], 'missing write sections')
    for section in draft['sections']:
        _require(_text(section['heading']) and _text(section['text']), 'invalid write section')
        validate_ids(section['paper_ids'], ids)
        cited.update(section['paper_ids'])
        citations = ' '.join(f'[{pid}]' for pid in section['paper_ids'])
        lines += ['## ' + section['heading'], '', section['text'] + ' ' + citations, '']
    _require(cited == ids, 'missing write citations')
    lines += ['## 参考文献', ''] + [f"- [{p['id']}] {p['title']}" for p in papers]
    return '\n'.join(lines) + '\n'


@_wire
def validate_episode(ep, *, mode='formal', manifest=None):
    """Validate provenance, step chain, masks, call ledger, budget and accepted final text."""
    _no_labels(ep)
    task = _task(ep['public_task'])
    _require(_text(ep['trajectory_id']) and ep['review_id'] == task.review_id and ep['split'] == task.split,
             'episode/task identity mismatch')
    _require(mode in ('formal', 'fixture') and ep['collection_kind'] == mode and
             ep['execution_kind'] == ('real' if mode == 'formal' else 'mock'), 'ineligible collection/execution kind')
    source = ep['source']
    selected = paper_index(ep['source_papers'])
    if mode == 'formal':
        m = _formal_manifest(manifest)
        _require(all(m['papers'].get(k) == v for k, v in selected.items()), 'non-allowlisted episode papers')
        expected = _source(task, ep['dataset_kind'], ep['data_purpose'], m['papers'], mode, m)
        _require(source == expected, 'episode source/manifest binding failed')
    else:
        _require(manifest is None and ep['dataset_kind'] == 'synthetic_fixture' and ep['data_purpose'] == 'dry_run_only'
                 and source['source_id'] == 'local/synthetic-fixture' and source['manifest_sha256'] is None and
                 source['task_sha256'] == json_sha256(task.to_dict()) and _hex(source['corpus_sha256']),
                 'not an explicit synthetic fixture')
    size = source['corpus_size']
    _require(_int(size) and 0 < len(selected) <= size, 'bad corpus size or empty trajectory sources')
    config = RunnerConfig(**ep['config'])
    _require(type(ep['config_version']) is int and ep['config_version'] == config.config_version == CONFIG_VERSION
             and set(ep['config']) == set(RunnerConfig.__dataclass_fields__) and
             ep['config_sha256'] == json_sha256(ep['config']), 'episode configuration mismatch')
    plan = sampling_plan(ep['policy'])
    _require(ep['plan'] == plan and ep['plan_sha256'] == json_sha256(plan), 'invalid sampling plan')
    rule = RuleScheduler(config)
    steps = ep['transitions']
    _require(isinstance(steps, list) and _int(ep['step_count']) and 0 < len(steps) == ep['step_count'] <= config.max_steps,
             'expected a complete, nonempty episode')
    used, previous = 0, None
    counts, costs = Counter(), dict.fromkeys(ACTIONS, 0)
    for i, t in enumerate(steps, 1):
        _require(all(t[k] == ep[k] for k in ('trajectory_id', 'review_id', 'config_version', 'execution_kind')) and
                 type(t['step_index']) is int and t['step_index'] == i, 'step provenance/index mismatch')
        before, after, action = t['state_before'], t['state_after'], t['action_id']
        if previous is None:
            _require(before['retrieved_paper_count'] == 0 and not before['summary_available'] and
                     before['material_revision'] == 0 and before['verifier'] == empty_verifier() and
                     all(h['call_count'] == 0 and h['token_share'] == 0 and h['last_verifier_delta'] is None and
                         h['consecutive_low_gain_or_failures'] == 0 for h in before['action_history'].values()),
                     'not the reset state')
        else:
            _require(before == previous, 'broken state chain')
        mask = _check_state(before, task, config, i - 1, used, size)
        _mask(t['action_mask'], mask)
        _require(not before['terminal'] and before['terminal_reason'] is None and action in ACTIONS and
                 mask[ACTIONS.index(action)] and action == _choose(rule, plan['preferences_by_step'], before, mask),
                 'illegal or off-plan action')
        bill = _usage(t['calls'], action, config, used)
        for key, amount in bill.items():
            _same_usage(t[key], amount, key)
        outputs = _successful_outputs(t['calls'])
        paper_count = after['retrieved_paper_count']
        for call in t['calls']:
            payload = call['payload']
            # Runner 只追加检索结果；写稿参考文献沿用调用时的论文顺序，而非按 ID 排序。
            _require(payload['query'] == task.survey_instruction and 0 < len(payload['papers']) == paper_count
                     and payload['papers'] == ep['source_papers'][:paper_count], 'call/task/source mismatch')
            if call['role'] == 'verify':
                _require(payload['items'] == (outputs['summarize']['items'] if 'summarize' in outputs else []),
                         'verifier input does not match updated material')
            if mode == 'formal':
                _require(call['model_requested'] == 'deepseek-flash', 'not a formal workflow model')
        used = total((used, bill['total_workflow_tokens']))
        counts[action] += 1
        costs[action] = total((costs[action], bill['total_workflow_tokens']))
        _mask(t['next_mask'], _check_state(after, task, config, i, used, size))
        for a in ACTIONS:
            h = after['action_history'][a]
            _require(h['call_count'] == counts[a] and h['token_share'] ==
                     (None if used is None else costs[a] / used if used else 0), 'history count/token share mismatch')
        _require(type(t['success']) is bool and type(t['done']) is bool and
                 t['done'] == after['terminal'] == (i == len(steps)), 'incomplete or post-terminal step')
        is_write = action.startswith('write_')
        _check_step_effects(t, outputs, used is not None and used < config.workflow_tokens_budget)
        reason = ('usage_unknown' if used is None else 'token_budget' if used >= config.workflow_tokens_budget else
                  'final_written' if is_write and t['success'] else 'max_steps' if i == config.max_steps else None)
        _require(t['terminal_reason'] == after['terminal_reason'] == reason and t['done'] == (reason is not None)
                 and not (is_write and t['success'] and reason != 'final_written'), 'inconsistent budget/terminal reason')
        _require(t['verifier_before'] == verifier_value(before) and t['verifier_after'] == verifier_value(after)
                 and t['verifier_delta'] == (comparable_delta(before, after) if t['success'] and not is_write else None),
                 'inconsistent verifier values or delta')
        previous = after
    _same_usage(ep['total_workflow_tokens'], used, 'episode')
    success = reason == 'final_written'
    _require(ep['success'] is success and ep['terminal_reason'] == reason and ep['action_history'] == previous['action_history']
             and previous['retrieved_paper_count'] == len(selected), 'episode summary mismatch')
    if success:
        text = _render_accepted_write(outputs['write'], ep['source_papers'], ep['execution_kind'])
        _require(ep['final_text'] == text and ep['final_text_sha256'] == final_text_sha256(text),
                 'accepted final text/hash does not match successful write output')
    else:
        _require(ep['final_text'] is None and ep['final_text_sha256'] is None and ep['final_text_ref'] is None,
                 'failed episode must not publish a final text')
    _require(ep['support'] == support_report([ep]), 'inconsistent opportunity/support counts')


@_wire
def validate_collection(episodes, labels=(), *, mode='formal', manifest=None):
    """Fail closed; return joined independent labels only (no mutation, no missing-score imputation)."""
    by_id, joined, label_ids, tasks = {}, {}, set(), {}
    for ep in episodes:
        _require('shared_prefix' not in ep, 'shared-prefix arms are paired observations, not independent episodes')
        validate_episode(ep, mode=mode, manifest=manifest)
        tid, rid = ep['trajectory_id'], ep['review_id']
        _require(tid not in by_id, 'duplicate trajectory_id')
        _require(tasks.setdefault(rid, ep['source']['task_sha256']) == ep['source']['task_sha256'],
                 'same review_id used for different tasks')
        by_id[tid] = ep
    _require(bool(by_id), 'empty training collection')
    for label in labels:
        tid = label['trajectory_id']
        _require(tid in by_id and tid not in joined, 'orphan or duplicate Judge label')
        ep = by_id[tid]
        _require(ep['success'] and all(label[k] == ep[k] for k in ('review_id', 'split', 'final_text_sha256')),
                 'label on missing final text or wrong text/identity/split')
        _require(label['source_kind'] == ('independent_judge' if mode == 'formal' else 'synthetic_fixture'),
                 'invalid label source')
        if 'label_id' in label:
            _require(_text(label['label_id']) and label['label_id'] not in label_ids, 'duplicate or empty label ID')
            label_ids.add(label['label_id'])
        if mode == 'formal':
            _require(all(label[k] == manifest['judge'][k] for k in ('model', 'prompt_id')), 'wrong frozen Judge')
            validate_archived_assessment(label)
        else:
            _require(_score(label['overall_score']) and all(_text(label[k]) for k in ('model', 'prompt_id')),
                     'invalid fixture label score or identifiers')
        joined[tid] = dict(label)
    return joined


def main(argv=None):
    parser = argparse.ArgumentParser(description='Offline synthetic collection only; no live CLI.')
    parser.add_argument('--policy', choices=[*REGISTERED_PLANS, 'all'], default='rule')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', type=Path, help='optional NEW JSON file; never overwrites existing results')
    args = parser.parse_args(argv)
    names = PLANS if args.policy == 'all' else [args.policy]
    episodes = [dry_run(name, seed=args.seed) for name in names]
    payload = json.dumps({'collection_kind': 'fixture', 'episodes': episodes, 'support': support_report(episodes)},
                         ensure_ascii=False, allow_nan=False)
    if args.output:
        with args.output.open('x', encoding='utf-8') as stream:
            stream.write(payload + '\n')
    else:
        print(payload)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
