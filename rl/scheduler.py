"""仅用标准库加载冻结的三层 MLP；不加载训练框架，不读取评审信息。"""

import json
import math
from pathlib import Path

from core.schema import ACTIONS
from rl.state_encoder import FEATURES, StateEncoder


class DQNScheduler:
    def __init__(self, artifact):
        self.encoder = StateEncoder(artifact['encoder']['paper_scale'])
        if artifact['format'] != 'dense-relu-v1' or artifact['encoder'] != self.encoder.spec():
            raise ValueError('Incompatible encoder or network format')
        width = len(FEATURES)
        self.layers = []
        for layer in artifact['layers']:
            w, b = layer['weight'], layer['bias']
            if not b or len(w) != len(b) or any(len(row) != width for row in w):
                raise ValueError('Invalid layer shape')
            if not all(math.isfinite(x) for row in [*w, b] for x in row):
                raise ValueError('Nonfinite parameter')
            self.layers.append((tuple(tuple(row) for row in w), tuple(b)))
            width = len(b)
        if len(self.layers) != 3 or width != len(ACTIONS):
            raise ValueError('Expected two hidden layers and five outputs')

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding='utf-8')))

    def reset(self, seed=None):
        pass

    def q_values(self, state, action_mask):
        x = self.encoder.encode(state, action_mask)
        for i, (w, b) in enumerate(self.layers):
            x = [bias + math.sumprod(row, x) for row, bias in zip(w, b)] if hasattr(math, 'sumprod') else [
                bias + sum(v * a for v, a in zip(row, x)) for row, bias in zip(w, b)]
            if i < len(self.layers) - 1:
                x = [max(0.0, v) for v in x]
        return x

    def select_action(self, state, action_mask):
        if not any(action_mask):
            raise ValueError('No physically executable action')
        values = self.q_values(state, action_mask)
        index = max((i for i, ok in enumerate(action_mask) if ok), key=values.__getitem__)
        return ACTIONS[index]
