"""离线单因素消融。默认只演练，正式数据须显式传入并通过 T4 来源验证。"""

import argparse
import hashlib
import json
import platform
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from dataclasses import asdict, replace
from pathlib import Path
from statistics import fmean, stdev

import torch

from core.schema import ACTIONS, CONFIG_VERSION
from evaluation.mismatch import MISMATCH_RULES, summarize_mismatch
from experiments.collect import PLANS, dry_run, json_sha256
from rl.calibration import evaluate_holdout, fit_calibration, validate_calibration_inputs
from rl.dqn import TrainConfig, train
from rl.state_encoder import DIFFICULTIES
from rl.synthetic_env import SyntheticEnv


def grid(size):
    if size == 'full':
        base = TrainConfig()
        capacities, intervals = (512, 8192), (50, 800)
    elif size == 'smoke':
        base = TrainConfig(steps=160, capacity=64, target_interval=8, warmup=32, batch=16)
        capacities, intervals = (32, 128), (4, 16)
    else:
        raise ValueError('Unknown ablation size')
    return {'base': base, 'buffer_small': replace(base, capacity=capacities[0]),
            'buffer_large': replace(base, capacity=capacities[1]),
            'target_frequent': replace(base, target_interval=intervals[0]),
            'target_rare': replace(base, target_interval=intervals[1])}


def _inputs_sha256(episodes, labels, manifest, holdout):
    # Keep the original experiment.json digest format, including the historical key 'holdout'.
    return json_sha256({'episodes': episodes, 'labels': labels, 'manifest': manifest, 'holdout': holdout})


def prepare(bundle=None, fixture_calibration=False, *, calibration_version='v1'):
    if bundle is None and not fixture_calibration:
        if calibration_version != 'v1':
            raise ValueError('A calibrated run requires training episodes')
        return SyntheticEnv, ['easy','medium','hard'], {'data_source':'analytic_fixture'}
    if bundle is None:
        eps = [dry_run(p,review_id=f'fixture-{i}',difficulty=d) for i,d in enumerate(('easy','hard','medium')) for p in PLANS]
        labels, manifest, holdout, mode = [], None, ['fixture-2'], 'fixture'
    else:
        eps, labels = bundle['episodes'], bundle['labels']
        manifest, holdout, mode = bundle['manifest'], bundle['holdout_review_ids'], 'formal'
    report = evaluate_holdout(eps,labels,mode=mode,manifest=manifest,holdout_review_ids=holdout,
                              calibration_version=calibration_version)
    fitting = [ep for ep in eps if ep['review_id'] not in holdout]
    tids = {ep['trajectory_id'] for ep in fitting}
    model = fit_calibration(fitting,[x for x in labels if x['trajectory_id'] in tids],mode=mode,manifest=manifest,
                            calibration_version=calibration_version)
    difficulties = [e['public_task']['difficulty'] for e in {ep['review_id']:ep for ep in eps if ep['review_id'] in holdout}.values()]
    meta = {'data_source':model.source, 'fit':model.manifest(), 'holdout':report,
            'inputs_sha256':_inputs_sha256(eps,labels,manifest,holdout)}
    return model.make_env, difficulties, meta


def metrics(episodes):
    steps = [t for ep in episodes for t in ep]
    usage = [t['total_workflow_tokens'] for t in steps]
    known = [ep[-1]['state_after']['workflow_tokens_used'] for ep in episodes
             if ep[-1]['state_after']['workflow_tokens_used'] is not None]
    total = sum(usage) if None not in usage else None
    actions = {}
    for i,a in enumerate(ACTIONS):
        part = [t['total_workflow_tokens'] for t in steps if t['action_id'] == a]
        opportunities = sum(t['action_mask'][i] for t in steps)
        actions[a] = {'calls':len(part),'opportunities':opportunities,
                      'selection_rate':len(part)/opportunities if opportunities else None,
                      'token_share':sum(part)/total if total else None}
    return {'episodes':len(episodes), 'completion_rate':fmean(ep[-1]['terminal_reason']=='final_written' for ep in episodes),
            'mean_known_episode_tokens':fmean(known) if known else None, 'known_cost_episodes':len(known),
            'unknown_cost_steps':usage.count(None), 'steps':len(steps), 'actions':actions,
            'low_support_steps':sum(bool((t.get('simulation_support') or {}).get('low_support')) for t in steps),
            'mismatch':summarize_mismatch(episodes)}


def evaluate(agent, env_factory, difficulties):
    env, episodes, returns = env_factory(), [], []
    for i, difficulty in enumerate(difficulties):
        state, info = env.reset(seed=4373+i,options={'difficulty':difficulty, 'review_id':f'sim-val-{i}'})
        steps, gain = [], 0
        while not state['terminal']:
            mask = info['action_mask']
            action = ACTIONS[agent.choose(agent.encoder.encode(state,mask),mask)]
            state, r, _, _, info = env.step(action)
            steps.append(info['transition'])
            gain += r
        episodes.append(steps)
        returns.append(gain)
    return {'mean_return':fmean(returns),'returns':returns,**metrics(episodes)}, episodes


def write_json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,allow_nan=False,separators=(',',':'))
        stream.write('\n')


def bind_legacy_checkpoint(checkpoint, experiment, training_bundle, output):
    """Copy an old formal policy only after validating both links back to its training inputs."""
    raw = Path(checkpoint).read_bytes()
    policy = json.loads(raw)
    meta = policy['metadata']
    experiment = json.loads(Path(experiment).read_bytes())
    if (meta.get('data_source') != 'formal_train_runner' or 'retrieval_spec_sha256' in meta or
            experiment.get('data_source') != 'formal_train_runner' or
            meta.get('experiment_sha256') != json_sha256(experiment)):
        raise ValueError('Expected an unbound formal checkpoint and its matching original experiment.json')
    bundle = json.loads(Path(training_bundle).read_bytes())
    episodes, labels, manifest = bundle['episodes'], bundle['labels'], bundle['manifest']
    if experiment.get('inputs_sha256') != _inputs_sha256(episodes, labels, manifest, bundle['holdout_review_ids']):
        raise ValueError('Training inputs do not match the original experiment')
    _, _, fingerprint = validate_calibration_inputs(episodes, labels, mode='formal', manifest=manifest)
    meta.update(retrieval_spec_sha256=fingerprint, legacy_checkpoint_sha256=hashlib.sha256(raw).hexdigest())
    write_json(Path(output), policy)  # exclusive create; keep the original policy and experiment intact


def run_one(root, name, cfg, seed, env_factory, difficulties, meta, *, interval, threshold):
    env = env_factory()
    metadata = {'data_source':meta['data_source'],'experiment_sha256':json_sha256(meta), 'seed':seed, 'name':name,
                'runner_config':asdict(env.config),
                'development_review_ids':sorted(set(meta.get('fit',{}).get('training_review_ids',[])) |
                                                set(meta.get('holdout',{}).get('holdout_review_ids',[])))}
    if meta['data_source'] == 'formal_train_runner':
        fingerprint = meta['fit'].get('retrieval_spec_sha256')
        if not fingerprint or fingerprint != env.model.retrieval_spec_sha256:
            raise ValueError('Training environment does not match the fitted retrieval specification')
        metadata['retrieval_spec_sha256'] = fingerprint
    run = root / f'{name}-seed{seed}'
    run.mkdir(exist_ok=False)
    evaluations = []
    start = time.perf_counter()
    # 保存每个评估步的完整前后状态与有效位。正式错配由三号位对这些数据统一复算。
    with (run/'eval_steps.jsonl').open('x',encoding='utf-8') as traces:
        def periodic(step, agent):
            if step % interval == 0 or step == cfg.steps:
                summary, episodes = evaluate(agent,env_factory,difficulties)
                evaluations.append({'step':step, **summary})
                for ep in episodes:
                    traces.write(json.dumps({'step':step,'transitions':ep},ensure_ascii=False,allow_nan=False)+'\n')
        agent, raw = train(env,cfg,seed,on_step=periodic)
    converged = next((evaluations[i]['step'] for i in range(2,len(evaluations)) if all(
        e['mean_return'] >= threshold for e in evaluations[i-2:i+1])), None)
    agent.save(run/'policy.json', metadata)
    write_json(run/'train.json',raw)
    result = {**metadata,'config':asdict(cfg), 'evaluations':evaluations, 'gradient_updates':agent.updates,
              'target_syncs':agent.target_syncs, 'threshold':threshold,'consecutive_evaluations':3,
              'steps_to_converge':converged, 'elapsed_seconds':time.perf_counter()-start}
    write_json(run/'result.json',result)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--size',choices=('smoke','full'),default='smoke')
    p.add_argument('--seeds',type=int,nargs='+',default=[7,17,27])
    p.add_argument('--config',choices=('all',*grid('full')),default='all')
    inputs = p.add_mutually_exclusive_group()
    inputs.add_argument('--fixture-calibration',action='store_true')
    inputs.add_argument('--training-bundle',type=Path)
    p.add_argument('--threshold',type=float,help='正式实验须提供训练内预先登记的回报门槛')
    p.add_argument('--eval-per-difficulty',type=int,help='每个难度的固定仿真验证局数（各配置与种子共用）')
    p.add_argument('--eval-interval',type=int,help='两次固定仿真评估间的环境步数；full 默认 500')
    p.add_argument('--calibration-version',choices=('v1','v2'),default='v1')
    p.add_argument('--workers',type=int,default=1,help='并行的独立配置/种子数；每个进程用一个 PyTorch 计算线程')
    p.add_argument('--output',type=Path,required=True,help='必须是不存在的新目录')
    args = p.parse_args(argv)
    import math
    if args.training_bundle and args.threshold is None:
        p.error('Formal runs require a preregistered --threshold')
    threshold = .5 if args.threshold is None else args.threshold
    if not math.isfinite(threshold) or len(set(args.seeds)) != len(args.seeds):
        p.error('Expected finite threshold and distinct seeds')
    if args.workers < 1:
        p.error('--workers must be positive')
    torch.set_num_threads(1)
    bundle = json.loads(args.training_bundle.read_text(encoding='utf-8')) if args.training_bundle else None
    interval = args.eval_interval if args.eval_interval is not None else (500 if args.size=='full' else 40)
    if interval < 1:
        p.error('--eval-interval must be positive')
    factory, difficulties, meta = prepare(bundle,args.fixture_calibration,calibration_version=args.calibration_version)
    if args.eval_per_difficulty is not None:
        if args.eval_per_difficulty < 1:
            p.error('--eval-per-difficulty must be positive')
        difficulties = list(DIFFICULTIES) * args.eval_per_difficulty
    configs = grid(args.size)
    meta.update(config_version=CONFIG_VERSION,python=platform.python_version(),torch=str(torch.__version__),
                config_grid={k:asdict(v) for k,v in configs.items()},seeds=args.seeds,size=args.size,
                selected=args.config,validation_difficulties=difficulties,threshold=threshold,mismatch_rules=MISMATCH_RULES,
                eval_interval=interval,workers=args.workers,reward_config=asdict(factory().reward_config))
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'experiment.json',meta)
    names = [n for n in configs if args.config in ('all',n)]
    jobs = [(args.output,n,configs[n],seed,factory,difficulties,meta) for n in names for seed in args.seeds]
    options = {'interval':interval,'threshold':threshold}
    by_run = {}
    def record(result):
        name, seed = result['name'], result['seed']
        by_run[name,seed] = result
        print(f"{name} seed={seed} steps={configs[name].steps} seconds={result['elapsed_seconds']:.2f}",flush=True)
    if args.workers == 1:
        for job in jobs:
            record(run_one(*job,**options))
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers,len(jobs)),mp_context=get_context('spawn'),
                                 initializer=torch.set_num_threads,initargs=(1,)) as pool:
            futures = [pool.submit(run_one,*job,**options) for job in jobs]
            for future in as_completed(futures):
                record(future.result())
    comparison = {}
    for name in names:
        results = [by_run[name,seed] for seed in args.seeds]
        last = [r['evaluations'][-1]['mean_return'] for r in results]
        comparison[name] = {'seeds': args.seeds, 'last_mean_return': fmean(last),
                            'last_return_sd': stdev(last) if len(last) > 1 else None,
                            'steps_to_converge_by_seed': [r['steps_to_converge'] for r in results],
                            'last_evaluation_by_seed': [r['evaluations'][-1] for r in results]}
    write_json(args.output/'comparison.json', comparison)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
