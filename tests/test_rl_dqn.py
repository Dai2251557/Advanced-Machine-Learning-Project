import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name != 'torch':
        raise
    raise unittest.SkipTest('T3 training tests require requirements-rl.txt')

from core.runner import Runner
from core.schema import ACTIONS, ReviewTask, RunnerConfig
from rl.dqn import DQN, ReplayBuffer, TrainConfig, targets, train
from rl.scheduler import DQNScheduler
from rl.synthetic_env import SyntheticEnv


class DQNTests(unittest.TestCase):
    def test_target_chooses_with_online_evaluates_with_target(self):
        online = lambda x: torch.tensor([[100., 4., 3., 0., 0.]]).expand(len(x), -1)
        target = lambda x: torch.tensor([[50., 1., 20., 0., 0.]]).expand(len(x), -1)
        masks = torch.tensor([[False, True, True, False, False], [False] * 5])
        result = targets(online, target, torch.tensor([1., 2.]), torch.zeros(2, 1), masks,
                         torch.tensor([False, True]), .5)
        self.assertEqual(result.tolist(), [1.5, 2.])
        with self.assertRaises(ValueError):
            targets(online, target, torch.ones(2), torch.zeros(2,1), masks, torch.tensor([False, False]), 1)

    def test_buffer_eviction_and_masked_exploration(self):
        replay = ReplayBuffer(2)
        for x in range(3):
            replay.push(x)
        self.assertEqual(set(replay.rows), {1, 2})
        agent = DQN(seed=1)
        env = SyntheticEnv()
        s, info = env.reset(seed=1)
        x = agent.encoder.encode(s, info['action_mask'])
        for eps in (0, .4, 1):
            self.assertEqual(agent.choose(x, info['action_mask'], eps), 0)

    def test_learn_known_choice_export_reload_and_runner(self):
        class HardEnv(SyntheticEnv):
            def reset(self, **kwargs):
                return super().reset(**kwargs, options={'difficulty': 'hard'})
        torch.set_num_threads(1)
        cfg = TrainConfig(steps=900, batch=16, warmup=32, capacity=256, target_interval=40)
        env = HardEnv(config=RunnerConfig(max_steps=2))
        agent, log = train(env, cfg, seed=7)
        self.assertEqual((len(log), agent.updates, agent.target_syncs), (900, 869, 21))
        env.reset(seed=5)
        s, _, _, _, info = env.step('retrieve_more')
        mask = info['action_mask']
        vector = agent.encoder.encode(s, mask)
        self.assertEqual(ACTIONS[agent.choose(vector, mask)], 'write_deep')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'policy.json'
            agent.save(path, {'source':'synthetic', 'seed':7})
            frozen = DQNScheduler.load(path)
            self.assertEqual(frozen.select_action(s, mask), 'write_deep')
            torch.testing.assert_close(agent.online(torch.tensor(vector)), torch.tensor(frozen.q_values(s, mask)),
                                       atol=1e-5, rtol=1e-5)
            runner = Runner({'papers':[{'id':'x','title':'X','abstract':'A synthetic paper.'}]},
                            config=RunnerConfig(max_steps=2))
            result = runner.run_episode(frozen, ReviewTask('fixture','paper',difficulty='hard'), seed=7)
            self.assertTrue(result['success'])
            # 禁用 site-packages 后，冻结推理仍可加载、不引入 torch。
            subprocess.run([sys.executable, '-B', '-S', '-c',
                            'import sys; from rl.scheduler import DQNScheduler; '
                            'DQNScheduler.load(sys.argv[1]); assert "torch" not in sys.modules', str(path)], check=True)
            invalid = json.loads(path.read_text())
            invalid['encoder']['features'].reverse()
            with self.assertRaises(ValueError):
                DQNScheduler(invalid)
