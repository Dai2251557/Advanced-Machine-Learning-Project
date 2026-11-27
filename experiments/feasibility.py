"""Small train-only, real-call decision-space probe; Judge runs after all trajectories."""

import argparse
import hashlib
import json
import urllib.error
import urllib.request
from pathlib import Path

from core.runner import Runner
from core.schema import ReviewTask, RunnerConfig
from evaluation.judge import (build_judge_request, finalize_judge_result,
                              prepare_train_review, save_judge_result)
from experiments.collect import collect_episode, validate_episode
from litreview.clients import DeepSeekBackend, NoRedirect
from utils.data_release import ROOT, load_corpus, load_manifest, read_json


TASK_IDS = ('R005', 'R001', 'R022')
POLICIES = ('raw_fast', 'rule', 'retrieve_again', 'deep')


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def preflight(task_ids=TASK_IDS, policies=POLICIES):
    """Reject missing or non-training IDs before creating files or making paid calls."""
    if not task_ids or not policies or len(set(task_ids)) != len(task_ids) or len(set(policies)) != len(policies):
        raise ValueError('Task IDs and policies must be nonempty and unique.')
    if not set(policies) <= {'raw_fast', 'rule', 'retrieve_again', 'deep'}:
        raise ValueError('Only predeclared probe routes are allowed.')
    tasks = {row['review_id']: ReviewTask.from_dict(row) for row in read_json(ROOT / 'data/train_tasks.json')}
    if not set(task_ids) <= tasks.keys() or any(tasks[rid].split != 'train' for rid in task_ids):
        raise ValueError('The feasibility probe accepts frozen training tasks only.')
    return tasks, load_corpus(), load_manifest('train'), RunnerConfig(**read_json(ROOT / 'config.json'))


def run_matrix(output, task_ids=TASK_IDS, policies=POLICIES, backend=None):
    tasks, corpus, manifest, config = preflight(task_ids, policies)
    backend = backend or DeepSeekBackend.from_env()
    if type(backend) is not DeepSeekBackend:
        raise ValueError('Real probe requires the official DeepSeek backend.')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    _write(output / 'plan.json', {'split': 'train', 'task_ids': list(task_ids), 'policies': list(policies),
                                  'source_id': manifest['source_id'], 'config': read_json(ROOT / 'config.json'),
                                  'judge_after_all_episodes': True, 'fulltext_arm': False})
    rows = []
    for rid in task_ids:
        for policy in policies:
            episode = collect_episode(Runner(corpus, backend, config, output / 'traces'), tasks[rid], policy,
                                      mode='formal', manifest=manifest)
            path = output / 'episodes' / f'{rid}--{policy}.json'
            _write(path, episode)
            rows.append({'review_id': rid, 'policy': policy, 'path': path.relative_to(output).as_posix(),
                         'sha256': _sha(path.read_bytes()), 'trajectory_id': episode['trajectory_id'],
                         'success': episode['success'], 'terminal_reason': episode['terminal_reason'],
                         'total_workflow_tokens': episode['total_workflow_tokens'],
                         'actions': [step['action_id'] for step in episode['transitions']]})
            print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
            if episode['terminal_reason'] == 'usage_unknown':
                _write(output / 'incomplete.json', {'reason': 'usage_unknown', 'rows': rows})
                return False
    _write(output / 'execution_complete.json', {'rows': rows, 'task_ids': list(task_ids),
                                                 'policies': list(policies), 'source_id': manifest['source_id']})
    return True


def _episodes(output, manifest):
    index = read_json(output / 'execution_complete.json')
    expected = {(rid, policy) for rid in index['task_ids'] for policy in index['policies']}
    rows = index['rows']
    if len(rows) != len(expected) or {(row['review_id'], row['policy']) for row in rows} != expected:
        raise ValueError('Incomplete task-policy matrix.')
    episodes = []
    for row in rows:
        path = output / row['path']
        if path.resolve().parent != (output / 'episodes').resolve() or _sha(path.read_bytes()) != row['sha256']:
            raise ValueError('Trajectory path or hash mismatch.')
        episode = read_json(path)
        validate_episode(episode, mode='formal', manifest=manifest)
        if (episode['review_id'], episode['policy'], episode['trajectory_id']) != (
                row['review_id'], row['policy'], row['trajectory_id']):
            raise ValueError('Trajectory identity mismatch.')
        episodes.append((row, episode))
    return episodes


def _send_judge(body):
    backend = DeepSeekBackend.from_env()
    if backend.base_url.rstrip('/') not in ('https://api.deepseek.com', 'https://api.deepseek.com/v1'):
        raise ValueError('Judge requires the official DeepSeek endpoint.')
    request = urllib.request.Request(backend.base_url.rstrip('/') + '/chat/completions',
                                     data=json.dumps(body, ensure_ascii=False).encode('utf-8'),
                                     headers={'Authorization': 'Bearer ' + backend.api_key,
                                              'Content-Type': 'application/json'}, method='POST')
    try:
        with urllib.request.build_opener(NoRedirect()).open(request, timeout=120) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f'Judge HTTP {exc.code}; response not accepted.') from None
    except (urllib.error.URLError, TimeoutError, OSError):
        raise RuntimeError('Judge connection failed; response not accepted.') from None


def judge_matrix(output, sender=_send_judge):
    """Score successful training outputs only, after the complete matrix is hash-checked."""
    output = Path(output)
    corpus, manifest = load_corpus(), load_manifest('train')
    episodes = _episodes(output, manifest)
    rows = []
    for row, episode in episodes:
        record = dict(row)
        if episode['success']:
            prepared = prepare_train_review(episode=episode, frozen_papers=corpus['papers'], manifest=manifest)
            body = build_judge_request(prepared)
            response = sender(body)
            receipt = {'binding': prepared['binding'], 'request': body, 'response': response}
            _write(output / 'judge_receipts' / f"{row['review_id']}--{row['policy']}.json", receipt)
            label = finalize_judge_result(prepared, receipt)
            save_judge_result(label, output / 'labels')
            record['judge_overall'] = label['overall_score']
            record['judge_subscores'] = label['subscores']
        else:
            record['judge_overall'] = None
            record['judge_subscores'] = None
        rows.append(record)
        print(json.dumps({'review_id': row['review_id'], 'policy': row['policy'],
                          'judge_overall': record['judge_overall']}, ensure_ascii=False), flush=True)
    _write(output / 'scores.json', {'split': 'train', 'score_source': 'independent_judge', 'rows': rows})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('run', 'judge'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.phase == 'run':
        return 0 if run_matrix(args.output) else 1
    judge_matrix(args.output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
