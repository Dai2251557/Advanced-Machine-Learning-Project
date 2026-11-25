"""使用同一 Runner 的冻结双策略执行入口；不导入训练框架或 Judge。CLI 仅演练。"""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from agents.executor import MockBackend
from core.rule_scheduler import RuleScheduler
from core.runner import Runner
from core.schema import ReviewTask
from litreview.clients import DeepSeekBackend
from rl.scheduler import DQNScheduler


def execute_pairs(runner, tasks, checkpoint, output, *, mode='fixture', seed=0):
    """正式执行只接受调用方显式提供的后端；整批完成后才写可交付索引。"""
    tasks = list(tasks)
    checkpoint = Path(checkpoint).read_bytes()
    artifact = json.loads(checkpoint)
    frozen = DQNScheduler(artifact)
    if not tasks or not all(isinstance(t, ReviewTask) for t in tasks):
        raise ValueError('Expected public ReviewTask instances')
    ids = {t.review_id for t in tasks}
    if len(ids) != len(tasks):
        raise ValueError('Duplicate review_id')
    if mode == 'formal':
        meta = artifact['metadata']
        if (type(runner.executor.backend) is not DeepSeekBackend or runner.data_purpose != 'formal_experiment' or
                runner.dataset_kind != 'arxiv_abstracts' or len(ids) < 5 or any(t.split != 'test' for t in tasks) or
                meta['data_source'] != 'formal_train_runner' or meta['runner_config'] != asdict(runner.config) or
                ids.intersection(meta['development_review_ids'])):
            raise ValueError('Formal run needs frozen compatible weights and at least five disjoint test tasks')
    elif mode != 'fixture' or not isinstance(runner.executor.backend, MockBackend) or any(t.split != 'pilot' for t in tasks):
        raise ValueError('Rehearsal requires mock backend and pilot tasks')
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    policies = [('rule', RuleScheduler(runner.config)), ('dqn', frozen)]
    rows = []
    for i, task in enumerate(tasks):
        for name, policy in (policies if i % 2 == 0 else reversed(policies)):
            ep = runner.run_episode(policy, task, seed=seed+i)
            path = out / f'{i:03d}-{name}.json'
            raw = json.dumps(ep, ensure_ascii=False,allow_nan=False).encode('utf-8')
            with path.open('xb') as stream:
                stream.write(raw)
            rows.append({'review_id':task.review_id, 'policy':name, 'trajectory_id':ep['trajectory_id'],
                         'success':ep['success'], 'trace':path.name, 'sha256':hashlib.sha256(raw).hexdigest()})
    manifest = {'mode':mode,'checkpoint_sha256':hashlib.sha256(checkpoint).hexdigest(),
                'runner_config':asdict(runner.config),'seed':seed,'all_episodes_complete':True,'episodes':rows}
    with (out/'execution_complete.json').open('x',encoding='utf-8') as stream:
        json.dump(manifest,stream,ensure_ascii=False,indent=2)
    return manifest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args(argv)
    papers = [{'id':str(i),'title':f'Fixture {i}','abstract':f'Synthetic paper {i}.'} for i in range(5)]
    runner = Runner({'papers':papers,'dataset_kind':'synthetic_fixture','purpose':'pilot_only'})
    tasks = [ReviewTask(f'pilot-{i}','Synthetic paper',difficulty=d) for i,d in enumerate(('easy','medium','hard'))]
    result = execute_pairs(runner,tasks,args.checkpoint,args.output)
    print(f"{len(result['episodes'])} mock trajectories sealed; no Judge called")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
