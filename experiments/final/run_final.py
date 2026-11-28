"""十道冻结公开测试题 × 三策略。--check 只核验本地材料；仅 --dispatch 发工作流请求。

正式运行只写 outputs/course-final/execution，不调用 Judge，不恢复或重跑已有批次。
"""
import argparse
import hashlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone

from core.rule_scheduler import RuleScheduler
from core.runner import Runner
from core.schema import ReviewTask, RunnerConfig
from evaluation._sealed import sealed_episodes
from experiments.collect import json_sha256
from experiments.frozen import _verify_test_seal
from rl.scheduler import DQNScheduler
from utils.data_release import load_corpus, load_manifest, read_json

from . import BATCH, CHECKPOINT_SHA256, POLICIES, ROOT, new_output

EXEC = BATCH / 'execution'
WORKERS = 10


def save(path, obj):
    data = json.dumps(obj, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode()
    with path.open('xb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    return hashlib.sha256(data).hexdigest()


def preflight():
    tasks = [ReviewTask(**t) for t in read_json(ROOT / 'data/test_tasks.json')]
    ids = {t.review_id for t in tasks}
    if len(tasks) != 10 or len(ids) != 10 or any(t.split != 'test' for t in tasks):
        raise ValueError('需完整十道独立公开测试题')
    config = RunnerConfig(**read_json(ROOT / 'config.json'))
    if config.cache_enabled or config.workflow_tokens_budget != 48000:
        raise ValueError('工作流配置不匹配')
    corpus = load_corpus()
    runner = Runner(corpus, config=config)  # No paid backend; do not start any episode here.
    delivery = _verify_test_seal(runner, tasks, load_manifest('test'))
    spec = json_sha256(runner.retriever.spec())
    artifacts = {}
    for name, expected in CHECKPOINT_SHA256.items():
        folder = ROOT / 'checkpoints' / name
        data = (folder / 'policy.json').read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f'{name}: 权重原始字节不匹配')
        artifact = json.loads(data)
        m = artifact['metadata']
        experiment = read_json(folder / 'experiment.json')
        if (m['data_source'] != 'formal_train_runner' or m['runner_config'] != asdict(config) or
                m['retrieval_spec_sha256'] != spec or ids.intersection(m['development_review_ids']) or
                m['experiment_sha256'] != json_sha256(experiment) or
                m['name'] not in experiment['config_grid'] or m['seed'] not in experiment['seeds']):
            raise ValueError(f'{name}: 权重与冻结训练、配置或测试环境不符')
        DQNScheduler(artifact)
        artifacts[name] = artifact
    header = {'format': 'frozen-three-v1', 'mode': 'formal', 'policies': list(POLICIES),
              'checkpoints_sha256': CHECKPOINT_SHA256, 'runner_config': asdict(config),
              'test_delivery': delivery, 'seed': 0, 'retrieval_spec_sha256': spec,
              'exploration_epsilon': 0, 'workers': WORKERS}
    return tasks, corpus, config, artifacts, header


class PauseBatch(Exception):
    pass


class Book:
    def __init__(self, stream):
        self.stream = stream
        self.lock = threading.Lock()
        self.sent = self.received = self.streak = 0
        self.stop = None

    def log(self, record):
        self.stream.write(json.dumps(record, ensure_ascii=False) + '\n')
        self.stream.flush()
        os.fsync(self.stream.fileno())

    def reserve(self, tid, rid, policy, step):
        with self.lock:
            if self.stop:
                raise PauseBatch(self.stop)
            self.sent += 1
            self.log({'event': 'dispatch', 'w': self.sent, 'trajectory_id': tid, 'review_id': rid,
                      'policy': policy, 'step': step, 'at': datetime.now(timezone.utc).isoformat()})
            return self.sent

    def settle(self, w, event):
        with self.lock:
            self.streak = self.streak + 1 if event.get('error_code') is not None else 0
            if self.streak >= 2:
                self.stop = 'two consecutive API errors'
            self.received += 1
            fields = ('role', 'attempt', 'status', 'http_status', 'error_code', 'prompt_tokens', 'completion_tokens')
            self.log({'event': 'result', 'w': w, **{k: event.get(k) for k in fields}})


def episode(i, name, task, corpus, config, artifacts, backend, book):
    with book.lock:
        if book.stop:
            return None
    runner = Runner(corpus, backend, config, EXEC / 'raw' / f'{i:03d}-{name}')
    original = runner._can_call
    policy = RuleScheduler(config) if name == 'rule' else DQNScheduler(artifacts[name])
    ids = []
    processed = 0

    def settle():
        nonlocal processed
        for ev in runner.executor.events[processed:]:
            book.settle(ids[processed], ev)
            processed += 1

    def before_call():
        settle()
        if not original():
            return False
        ids.append(book.reserve(runner.trajectory_id, task.review_id, name, runner.state['step_count']))
        return True

    runner._can_call = before_call
    state = runner.reset(task, seed=i)
    policy.reset(seed=i)
    try:
        while not state['terminal']:
            action = policy.select_action(state, runner.valid_actions())
            transition = runner.step(action)
            settle()
            state = transition['state_after']
    except PauseBatch:
        save(runner.episode_dir / 'interrupted.json', {'reason': book.stop, 'partial_events': runner.executor.events})
        return None
    ep = runner.result()
    calls = [e for tr in ep['transitions'] for e in tr['calls']]
    if processed != len(ids) or len(calls) != processed or any(e['cache_hit'] for e in calls):
        raise ValueError('实发调用账本不符')
    ep['final_text_sha256'] = hashlib.sha256(ep['final_text'].encode()).hexdigest() if ep['success'] else None
    trace = f'{i:03d}-{name}.json'
    return {'review_id': task.review_id, 'policy': name, 'trajectory_id': ep['trajectory_id'],
            'trace': trace, 'sha256': save(EXEC / trace, ep), 'success': ep['success'],
            'terminal_reason': ep['terminal_reason'], 'final_text_sha256': ep['final_text_sha256'], 'calls': len(calls)}


def run():
    new_output(EXEC)
    if (BATCH / 'judge').exists():
        raise FileExistsError('本批已有 Judge 记录，不能重新运行工作流')
    tasks, corpus, config, artifacts, header = preflight()
    from litreview.clients import DeepSeekBackend
    client = DeepSeekBackend.from_env()
    if client.base_url.rstrip('/') not in ('https://api.deepseek.com', 'https://api.deepseek.com/v1') or client.model != 'deepseek-flash':
        raise ValueError('非冻结官方后端')
    EXEC.mkdir(parents=True, exist_ok=False)  # 一次性占用：中断后也不能在相同批次重跑。
    schedule = [(i, POLICIES[(i + shift) % 3]) for shift in range(3) for i in range(10)]
    save(EXEC / 'started.json', {**header, 'schedule': [[tasks[i].review_id, n] for i, n in schedule]})
    rows = []
    with (EXEC / 'calls.jsonl').open('x', encoding='utf-8') as f, ThreadPoolExecutor(max_workers=WORKERS) as pool:
        book = Book(f)
        futures = [pool.submit(episode, i, n, tasks[i], corpus, config, artifacts, client, book) for i, n in schedule]
        for future in as_completed(futures):
            row = future.result()
            if row is not None:
                rows.append(row)
                print(len(rows), '/30', row['review_id'], row['policy'], row['terminal_reason'], flush=True)
    if len(rows) != 30 or book.sent != book.received:
        save(EXEC / 'incomplete.json', {'reason': book.stop, 'completed': len(rows), 'W': book.sent,
                                        'responses_accounted': book.received})
        raise SystemExit('未完整封存，不得启动 Judge')
    rows.sort(key=lambda r: r['trace'])
    save(EXEC / 'execution_complete.json', {**header, 'all_episodes_complete': True, 'episodes': rows,
                                            'workflow_calls': book.sent})
    if len(list(sealed_episodes(EXEC, {t.review_id for t in tasks}, header['test_delivery']))) != 30:
        raise ValueError('封存轨迹数不符')
    print('已封存', len(rows), '轨；成功', sum(r['success'] for r in rows), 'W', book.sent, 'J', 0, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--check', action='store_true', help='仅本地预检，不需要 Key')
    group.add_argument('--dispatch', action='store_true', help='发起计费的正式工作流，仅限单独授权后的新批次')
    if parser.parse_args().check:
        preflight()
        print('本地预检通过：十题与语料符合原清单，两份权重与训练设置匹配，未调用模型。')
    else:
        run()


if __name__ == '__main__':
    main()
