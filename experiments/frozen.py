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
from experiments.collect import json_sha256, paper_index
from litreview.clients import DeepSeekBackend
from rl.scheduler import DQNScheduler


def _verify_test_seal(runner, tasks, seal):
    """清单来自数据负责人预先封存的独立交付，不在执行时用候选输入自签。"""
    if (not isinstance(seal, dict) or not isinstance(seal.get('source_id'), str) or
            not seal['source_id'].strip() or type(seal.get('config_version')) is not int or
            seal['config_version'] != runner.config.config_version or
            seal.get('dataset_kind') != runner.dataset_kind or seal.get('data_purpose') != runner.data_purpose):
        raise ValueError('A compatible sealed public test manifest is required')
    expected_tasks = {t.review_id: {'split': 'test', 'task_sha256': json_sha256(t.to_dict())} for t in tasks}
    if seal.get('tasks') != expected_tasks:
        raise ValueError('Test tasks must exactly match the sealed IDs and public task hashes')
    if seal.get('papers') != paper_index(runner.papers):
        raise ValueError('Entire Runner corpus must match the sealed paper hashes')
    return {'source_id': seal['source_id'], 'manifest_sha256': json_sha256(seal)}


def execute_pairs(runner, tasks, checkpoint, output, *, mode='fixture', seed=0, test_manifest=None):
    """正式模式先核验外部测试清单；整批完成后才写可交付索引。"""
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
        seal_ref = _verify_test_seal(runner, tasks, test_manifest)
    elif (mode != 'fixture' or test_manifest is not None or not isinstance(runner.executor.backend, MockBackend)
          or any(t.split != 'pilot' for t in tasks)):
        raise ValueError('Rehearsal requires mock backend and pilot tasks, without a formal test manifest')
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
    if mode == 'formal':
        manifest['test_delivery'] = seal_ref
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
