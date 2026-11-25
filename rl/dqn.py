"""CPU Masked Double-DQN；真实调用只在 Runner 中，这里只与仿真环境交互。"""

import copy
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from core.schema import ACTIONS
from rl.state_encoder import FEATURES, StateEncoder


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 20000
    capacity: int = 2048
    target_interval: int = 200  # 单位为实际梯度更新次数
    warmup: int = 128
    batch: int = 32
    hidden: int = 64
    lr: float = .001
    discount: float = 1.0
    epsilon_end: float = .05
    decay_fraction: float = .6

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in (self.steps, self.capacity, self.target_interval,
                                                       self.batch, self.warmup, self.hidden)):
            raise ValueError('Expected positive training sizes')
        if not self.batch <= self.warmup <= self.capacity or self.steps < self.warmup:
            raise ValueError('Expected batch <= warmup <= capacity and steps >= warmup')
        if not (math.isfinite(self.lr) and self.lr > 0 and 0 <= self.discount <= 1 and
                0 <= self.epsilon_end <= 1 and 0 < self.decay_fraction <= 1):
            raise ValueError('Invalid optimizer or exploration setting')


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.rows = []
        self.cursor = 0

    def push(self, row):
        if len(self.rows) < self.capacity:
            self.rows.append(row)
        else:
            self.rows[self.cursor] = row
        self.cursor = (self.cursor + 1) % self.capacity

    def sample(self, count, rng):
        states, actions, rewards, after, masks, done = zip(*rng.sample(self.rows, count))
        return (torch.tensor(states, dtype=torch.float32), torch.tensor(actions),
                torch.tensor(rewards, dtype=torch.float32), torch.tensor(after, dtype=torch.float32),
                torch.tensor(masks, dtype=torch.bool), torch.tensor(done, dtype=torch.bool))


@torch.no_grad()
def targets(online, target, rewards, after, masks, done, discount):
    # 终止步（包括预算耗尽）不做 argmax，也不对全假 Mask bootstrap。
    out = rewards.clone()
    live = ~done
    if live.any():
        if not masks[live].any(dim=1).all():
            raise ValueError('Nonterminal state has no legal action')
        a = online(after[live]).masked_fill(~masks[live], -torch.inf).argmax(dim=1, keepdim=True)
        out[live] += discount * target(after[live]).gather(1, a).squeeze(1)
    return out


class DQN:
    def __init__(self, config=TrainConfig(), seed=0, encoder=None):
        self.config = config
        self.encoder = StateEncoder() if encoder is None else encoder
        self.rng, self.replay_rng = random.Random(seed), random.Random(seed + 1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.online = nn.Sequential(nn.Linear(len(FEATURES), config.hidden), nn.ReLU(),
                                        nn.Linear(config.hidden, config.hidden), nn.ReLU(),
                                        nn.Linear(config.hidden, len(ACTIONS)))
        self.target = copy.deepcopy(self.online).requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.lr)
        self.replay = ReplayBuffer(config.capacity)
        self.updates = self.target_syncs = 0

    @torch.no_grad()
    def choose(self, vector, mask, epsilon=0.0):
        legal = [i for i, ok in enumerate(mask) if ok]
        if not legal:
            raise ValueError('No physically executable action')
        if epsilon > 0 and self.rng.random() < epsilon:
            return self.rng.choice(legal)
        values = self.online(torch.tensor(vector, dtype=torch.float32))
        return int(values.masked_fill(~torch.tensor(mask), -torch.inf).argmax())

    def update(self):
        s, a, r, after, masks, done = self.replay.sample(self.config.batch, self.replay_rng)
        expected = targets(self.online, self.target, r, after, masks, done, self.config.discount)
        loss = F.smooth_l1_loss(self.online(s).gather(1, a[:, None]).squeeze(1), expected)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 1.0)
        self.optimizer.step()
        self.updates += 1
        if self.updates % self.config.target_interval == 0:
            self.target.load_state_dict(self.online.state_dict())
            self.target_syncs += 1
        return loss.item()

    def artifact(self):
        return {'format': 'dense-relu-v1', 'encoder': self.encoder.spec(),
                'layers': [{'weight': layer.weight.detach().tolist(), 'bias': layer.bias.detach().tolist()}
                           for layer in self.online if isinstance(layer, nn.Linear)]}

    def save(self, path, metadata):
        record = {**self.artifact(), 'training': asdict(self.config), 'metadata': metadata}
        with Path(path).open('x', encoding='utf-8') as stream:
            json.dump(record, stream, allow_nan=False, separators=(',', ':'))


def train(env, config=TrainConfig(), seed=0, *, on_step=None):
    """on_step 用独立环境评估；记录原始步级奖励，跨 episode 不 bootstrap。"""
    agent = DQN(config, seed)
    episodes = random.Random(seed)
    def reset():
        return env.reset(seed=episodes.randrange(2 ** 32))
    state, info = reset()
    log = []
    for step in range(1, config.steps + 1):
        mask = info['action_mask']
        vector = agent.encoder.encode(state, mask)
        epsilon = 1 - (1 - config.epsilon_end) * min(1.0, (step - 1) / (config.steps * config.decay_fraction))
        index = agent.choose(vector, mask, epsilon)
        state, reward, terminated, truncated, info = env.step(ACTIONS[index])
        done = terminated or truncated
        mask = info['action_mask']
        agent.replay.push((vector, index, reward, agent.encoder.encode(state, mask), tuple(mask), done))
        loss = agent.update() if len(agent.replay.rows) >= config.warmup else None
        log.append({'step': step, 'action_id': ACTIONS[index], 'reward': reward, 'epsilon': epsilon,
                    'done': done, 'terminal_reason': state['terminal_reason'], 'loss': loss})
        if on_step is not None:
            on_step(step, agent)
        if done:
            state, info = reset()
    return agent, log
