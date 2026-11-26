import hashlib
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name != 'torch':
        raise
    raise unittest.SkipTest('T6 training rehearsal requires requirements-rl.txt')

from core.runner import Runner
from core.schema import ReviewTask
from experiments.ablate import bind_legacy_checkpoint, grid, main as train_main, metrics
from experiments.collect import json_sha256
from experiments.frozen import execute_pairs
from rl.dqn import TrainConfig
from rl.synthetic_env import SyntheticEnv
from utils.data_release import ROOT, load_manifest, read_json


class FormalRetrievalTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        eps = [read_json(ROOT / 'handoff/role1_train_probes' / f'{rid}-collection.json') for rid in ('R005', 'R001')]
        self.bundle = {'episodes': eps, 'labels': [], 'manifest': load_manifest('train'), 'holdout_review_ids': ['R001']}
        self.input = self.save('bundle.json', self.bundle)
        self.out = self.root / 'training'
        self.args = ['--training-bundle', str(self.input), '--threshold', '0', '--output', str(self.out),
                     '--seeds', '7', '--config', 'base']

    def save(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value), encoding='utf-8')
        return path

    def test_changed_held_out_spec_rejected_before_training_or_outputs(self):
        self.bundle['episodes'][1]['retrieval']['spec']['method'] = 'different'
        self.save('bundle.json', self.bundle)
        with patch('experiments.ablate.train') as train, self.assertRaisesRegex(ValueError, 'retrieval.spec'):
            train_main(self.args)
        train.assert_not_called()
        self.assertFalse(self.out.exists())

    def test_training_propagates_the_fitting_fingerprint_and_legacy_recovery_preserves_weights(self):
        cfg = TrainConfig(steps=4, capacity=2, warmup=2, batch=1, hidden=2)
        with patch('experiments.ablate.grid', return_value={'base': cfg}):
            train_main(self.args)
        experiment = read_json(self.out / 'experiment.json')
        policy = read_json(self.out / 'base-seed7' / 'policy.json')
        expected = json_sha256(self.bundle['episodes'][0]['retrieval']['spec'])
        self.assertEqual(experiment['fit']['retrieval_spec_sha256'], expected)
        self.assertEqual(policy['metadata']['experiment_sha256'], json_sha256(experiment))
        self.assertEqual(policy['metadata']['retrieval_spec_sha256'], expected)

        # Recreate the pre-fingerprint format, retaining its two original hash links.
        del experiment['fit']['retrieval_spec_sha256'], policy['metadata']['retrieval_spec_sha256']
        policy['metadata']['experiment_sha256'] = json_sha256(experiment)
        exp = self.save('old-experiment.json', experiment)
        old = self.save('old-policy.json', policy)
        copied = self.root / 'bound-policy.json'
        digest = hashlib.sha256(old.read_bytes()).hexdigest()
        for wrong_exp, wrong_inputs in ((self.out / 'experiment.json', self.input),
                                        (exp, self.save('rearranged.json', {**self.bundle, 'holdout_review_ids': ['R005']}))):
            with self.subTest(wrong_exp=wrong_exp, wrong_inputs=wrong_inputs), self.assertRaises(ValueError):
                bind_legacy_checkpoint(old, wrong_exp, wrong_inputs, copied)
            self.assertFalse(copied.exists())
        bind_legacy_checkpoint(old, exp, self.input, copied)
        rebound = read_json(copied)
        self.assertEqual(rebound['metadata'].pop('retrieval_spec_sha256'), expected)
        self.assertEqual(rebound['metadata'].pop('legacy_checkpoint_sha256'), digest)
        self.assertEqual(rebound, policy)  # weights, encoder, training config and original experiment hash
        self.assertEqual(hashlib.sha256(old.read_bytes()).hexdigest(), digest)
        with self.assertRaises(FileExistsError):
            bind_legacy_checkpoint(old, exp, self.input, copied)

    def test_legacy_hashes_do_not_waive_formal_label_validation(self):
        self.bundle['labels'] = [{'trajectory_id': 'absent'}]
        self.save('bundle.json', self.bundle)
        inputs = {k: self.bundle[k] for k in ('episodes', 'labels', 'manifest')}
        inputs['holdout'] = self.bundle['holdout_review_ids']
        experiment = {'data_source': 'formal_train_runner', 'inputs_sha256': json_sha256(inputs)}
        exp = self.save('old-experiment.json', experiment)
        old = self.save('old-policy.json', {'metadata': {'data_source': 'formal_train_runner',
                                                         'experiment_sha256': json_sha256(experiment)}})
        copied = self.root / 'bound-policy.json'
        with self.assertRaisesRegex(ValueError, 'orphan'):
            bind_legacy_checkpoint(old, exp, self.input, copied)
        self.assertFalse(copied.exists())


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
