"""离线单因素消融。默认只演练，正式数据须显式传入并通过 T4 来源验证。"""

import argparse
import json
import platform
import time
from dataclasses import asdict, replace
from pathlib import Path
from statistics import fmean, stdev

import torch

from core.schema import ACTIONS, CONFIG_VERSION
from experiments.collect import PLANS, dry_run, json_sha256
from rl.calibration import evaluate_holdout, fit_calibration
from rl.dqn import TrainConfig, train
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


def prepare(bundle=None, fixture_calibration=False):
    if bundle is None and not fixture_calibration:
        return SyntheticEnv, ['easy','medium','hard'], {'data_source':'analytic_fixture'}
    if bundle is None:
        eps = [dry_run(p,review_id=f'fixture-{i}',difficulty=d) for i,d in enumerate(('easy','hard','medium')) for p in PLANS]
        labels, manifest, holdout, mode = [], None, ['fixture-2'], 'fixture'
    else:
        eps, labels = bundle['episodes'], bundle['labels']
        manifest, holdout, mode = bundle['manifest'], bundle['holdout_review_ids'], 'formal'
    report = evaluate_holdout(eps,labels,mode=mode,manifest=manifest,holdout_review_ids=holdout)
    fitting = [ep for ep in eps if ep['review_id'] not in holdout]
    tids = {ep['trajectory_id'] for ep in fitting}
    model = fit_calibration(fitting,[x for x in labels if x['trajectory_id'] in tids],mode=mode,manifest=manifest)
    difficulties = [e['public_task']['difficulty'] for e in {ep['review_id']:ep for ep in eps if ep['review_id'] in holdout}.values()]
    meta = {'data_source':model.source, 'fit':model.manifest(), 'holdout':report,
            'inputs_sha256':json_sha256({'episodes':eps,'labels':labels,'manifest':manifest,'holdout':holdout})}
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
            'mismatch_status':'awaiting_frozen_role3_classifier'}


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


def run_one(root, name, cfg, seed, env_factory, difficulties, meta, *, interval, threshold):
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
        agent, raw = train(env_factory(),cfg,seed,on_step=periodic)
    converged = next((evaluations[i]['step'] for i in range(2,len(evaluations)) if all(
        e['mean_return'] >= threshold for e in evaluations[i-2:i+1])), None)
    metadata = {'data_source':meta['data_source'],'experiment_sha256':json_sha256(meta), 'seed':seed, 'name':name,
                'runner_config':asdict(env_factory().config),
                'development_review_ids':sorted(set(meta.get('fit',{}).get('training_review_ids',[])) |
                                                set(meta.get('holdout',{}).get('holdout_review_ids',[])))}
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
    p.add_argument('--output',type=Path,required=True,help='必须是不存在的新目录')
    args = p.parse_args(argv)
    import math
    if args.training_bundle and args.threshold is None:
        p.error('Formal runs require a preregistered --threshold')
    threshold = .5 if args.threshold is None else args.threshold
    if not math.isfinite(threshold) or len(set(args.seeds)) != len(args.seeds):
        p.error('Expected finite threshold and distinct seeds')
    torch.set_num_threads(1)
    bundle = json.loads(args.training_bundle.read_text(encoding='utf-8')) if args.training_bundle else None
    factory, difficulties, meta = prepare(bundle,args.fixture_calibration)
    configs = grid(args.size)
    meta.update(config_version=CONFIG_VERSION,python=platform.python_version(),torch=str(torch.__version__),
                config_grid={k:asdict(v) for k,v in configs.items()},seeds=args.seeds,size=args.size,
                selected=args.config,validation_difficulties=difficulties,threshold=threshold,
                eval_interval=500 if args.size=='full' else 40,reward_config=asdict(factory().reward_config))
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'experiment.json',meta)
    comparison = {}
    for name, cfg in configs.items():
        if args.config not in ('all', name):
            continue
        results = []
        for seed in args.seeds:
            result = run_one(args.output, name, cfg, seed, factory, difficulties, meta,
                             interval=meta['eval_interval'], threshold=threshold)
            results.append(result)
            print(f"{name} seed={seed} steps={cfg.steps} seconds={result['elapsed_seconds']:.2f}")
        last = [r['evaluations'][-1]['mean_return'] for r in results]
        comparison[name] = {'seeds': args.seeds, 'last_mean_return': fmean(last),
                            'last_return_sd': stdev(last) if len(last) > 1 else None,
                            'steps_to_converge_by_seed': [r['steps_to_converge'] for r in results],
                            'last_evaluation_by_seed': [r['evaluations'][-1] for r in results]}
    write_json(args.output/'comparison.json', comparison)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
