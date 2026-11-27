"""Versioned V2 interface for DQN integration; quality evaluator is supplied externally.

This is a live, paid-call environment when given EvidenceBackend. It is not a
claim that the provisional benchmark or reward has passed trainability gates.
"""

import math
import random

from agents.retrieval import expand_query, tokens
from core.v2_evidence import EvidenceRunner


ACTION_ORDER = ('retrieve_more', 'rewrite_query', 'read_1', 'read_2', 'read_3', 'write')
FEATURE_NAMES = ('budget_left', 'steps_left', 'papers_fraction', 'opened_fraction',
                 'rewrite_fraction', 'requests_number', 'requests_comparison', 'mentions_abstract',
                 'query_changed', 'reading_feedback_valid', 'supported_reading_fraction', 'last_reading_supported',
                 *[f'candidate_{i}_{name}' for i in range(1, 4)
                                    for name in ('available', 'query_overlap', 'length')])


class V2LiveEnv:
    def __init__(self, corpus, tasks, sections, backend, evaluator, *, token_budget=4500, max_steps=8):
        if not tasks or any(t.get('split') != 'train' for t in tasks):
            raise ValueError('Only public training tasks are accepted.')
        if not callable(evaluator):
            raise ValueError('An independently validated terminal evaluator is required.')
        # Project public fields before any policy observation or model call.
        self.tasks = [{k: t[k] for k in ('review_id', 'survey_instruction', 'split')} for t in tasks]
        self.corpus, self.sections, self.backend = corpus, sections, backend
        self.evaluator = evaluator
        self.token_budget, self.max_steps = token_budget, max_steps
        self.runner = None

    def _candidates(self):
        r = self.runner
        selected = {p['id'] for p in r.selected}
        opened = {s['section_id'] for s in r.opened}
        terms = set(expand_query(r.query + ' ' + (r.gap or '')))
        candidates = []
        for section in r.sections.values():
            if section['paper_id'] in selected and section['section_id'] not in opened:
                metadata = set(tokens(section['heading'] + ' ' + ' '.join(section['search_terms'])))
                overlap = len(terms & metadata) / max(1, len(terms))
                candidates.append((section, overlap))
        return sorted(candidates, key=lambda x: (-x[1], x[0]['section_id']))[:3]

    def _observation(self):
        r = self.runner
        q = r.task['survey_instruction']
        vector = [max(0., 1 - r.tokens_used / r.token_budget),
                  max(0., 1 - len(r.events) / r.max_steps),
                  len(r.selected) / len(r.papers), len(r.opened) / max(1, len(r.sections)),
                  r.rewrite_count / 2, float(any(w in q for w in ('多少', '数量', '幅度', '几次'))),
                  float(any(w in q for w in ('比较', '对比', '分别'))), float('摘要' in q),
                  float(r.query != q)]
        feedback = [e['detail']['reading']['supports_question'] for e in r.events
                    if 'reading' in e.get('detail', {}) and
                    type(e['detail']['reading'].get('supports_question')) is bool]
        vector.extend([float(bool(feedback)), sum(feedback) / max(1, len(feedback)),
                       float(feedback[-1]) if feedback else 0.])
        candidates = self._candidates()
        for i in range(3):
            vector.extend([1., candidates[i][1], min(1., len(candidates[i][0]['text']) / 10000)]
                          if i < len(candidates) else [0., 0., 0.])
        return vector

    def _info(self):
        mask = self.runner.valid_actions()
        candidates = self._candidates()
        return {'interface_version': 'v2-live-1', 'action_order': list(ACTION_ORDER),
                'feature_names': list(FEATURE_NAMES),
                'action_mask': [mask[0], mask[1], *[mask[2] and i < len(candidates) for i in range(3)], mask[3]],
                'candidate_ids': [s['section_id'] for s, _ in candidates],
                'review_id': self.runner.task['review_id']}

    def reset(self, *, seed=None, options=None):
        index = random.Random(seed).randrange(len(self.tasks)) if options is None else options['task_index']
        if type(index) is not int or not 0 <= index < len(self.tasks):
            raise ValueError('Invalid task index.')
        self.runner = EvidenceRunner(self.corpus, self.tasks[index], self.sections, self.backend,
                                     batch_size=2, token_budget=self.token_budget,
                                     max_steps=self.max_steps, read_on_open=True)
        return self._observation(), self._info()

    def step(self, action):
        if self.runner is None:
            raise RuntimeError('Call reset first.')
        if type(action) is int and 0 <= action < len(ACTION_ORDER):
            action = ACTION_ORDER[action]
        if action not in ACTION_ORDER or not self._info()['action_mask'][ACTION_ORDER.index(action)]:
            raise ValueError('Invalid or masked action.')
        before = self.runner.tokens_used
        if action.startswith('read_'):
            target = self._candidates()[int(action[-1]) - 1][0]['section_id']
            transition = self.runner.step('open_section', target)
        else:
            transition = self.runner.step(action)
        cost = self.runner.tokens_used - before
        reward = -.1 * cost / self.token_budget
        reason = self.runner.terminal_reason
        terminated = reason == 'final_written'
        truncated = reason is not None and not terminated
        info = self._info()
        info.update(cost_tokens=cost, terminal_reason=reason, eligible_for_training=reason != 'action_error',
                    transition=transition)
        if terminated:
            quality = self.evaluator(self.runner.result())
            if type(quality) not in (float, int) or not math.isfinite(quality) or not 0 <= quality <= 1:
                raise ValueError('Invalid external quality score; discard this rollout.')
            reward += quality
        elif reason in ('token_budget', 'max_steps'):
            reward -= 1.
        # Transport/schema errors must be excluded, not learnt as action quality.
        return self._observation(), reward, terminated, truncated, info
