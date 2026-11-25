import hashlib
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name != 'torch':
        raise
    raise unittest.SkipTest('T6 training rehearsal requires requirements-rl.txt')

from core.runner import Runner
from core.schema import ReviewTask
from experiments.ablate import grid, main as train_main, metrics
from experiments.frozen import execute_pairs
from rl.synthetic_env import SyntheticEnv


class ExperimentTests(unittest.TestCase):
    def test_grid_shares_one_control_and_changes_one_factor(self):
        for size in ('smoke','full'):
            configs = grid(size)
            self.assertEqual(len(configs),5)
            base = asdict(configs['base'])
            for name, cfg in list(configs.items())[1:]:
                self.assertEqual([k for k,v in asdict(cfg).items() if v != base[k]],
                                 ['capacity' if name.startswith('buffer') else 'target_interval'])
                self.assertGreaterEqual(cfg.steps,cfg.capacity)
                self.assertGreaterEqual((cfg.steps-cfg.warmup+1)//cfg.target_interval,2)

    def test_action_opportunities_and_token_shares(self):
        env = SyntheticEnv(corpus_size=3)
        env.reset(seed=10)
        steps = [env.step(a)[-1]['transition'] for a in ('retrieve_more','write_fast')]
        m = metrics([steps])
        self.assertEqual(m['actions']['retrieve_more']['selection_rate'],1)
        self.assertEqual(m['actions']['write_deep']['selection_rate'],0)
        self.assertAlmostEqual(sum(row['token_share'] for row in m['actions'].values()),1)
        self.assertEqual(m['completion_rate'],1)

    def test_end_to_end_fixture_and_pair_seal(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            train_main(['--output',str(root/'sim'),'--fixture-calibration','--seeds','5','--config','base'])
            run = root/'sim'/'base-seed5'
            ckpt = run/'policy.json'
            training = json.loads((run/'train.json').read_text())
            self.assertEqual(len(training),160)
            evaluations = json.loads((run/'result.json').read_text())['evaluations']
            self.assertEqual([e['step'] for e in evaluations],[40,80,120,160])
            trace = json.loads((run/'eval_steps.jsonl').read_text().splitlines()[0])
            self.assertIn('state_before',trace['transitions'][0])
            self.assertEqual(trace['transitions'][0]['execution_kind'],'simulated')
            runner = Runner({'papers':[{'id':'x','title':'T','abstract':'An example.'}]})
            tasks = [ReviewTask('p1','example'),ReviewTask('p2','example')]
            res = execute_pairs(runner,tasks,ckpt,root/'pairs')
            self.assertTrue(res['all_episodes_complete'])
            self.assertEqual([r['policy'] for r in res['episodes']],['rule','dqn','dqn','rule'])
            for row in res['episodes']:
                raw = (root/'pairs'/row['trace']).read_bytes()
                self.assertEqual(hashlib.sha256(raw).hexdigest(),row['sha256'])
                self.assertEqual(json.loads(raw)['execution_kind'],'mock')
            with self.assertRaises(FileExistsError): execute_pairs(runner,tasks,ckpt,root/'pairs')
            with self.assertRaises(ValueError): execute_pairs(runner,tasks,ckpt,root/'formal',mode='formal')
            self.assertFalse((root/'formal').exists())
