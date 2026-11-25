"""Offline numeric simulation of Runner v1. No text agent, API or Judge calls.

An outcome model samples one *joint* action/Verifier outcome. The environment,
not the sampler, advances materials, checks budget eligibility, builds the two
physical masks and computes reward. Snapshots use the ordinary Runner schema.
"""

import copy
import random
from dataclasses import dataclass

from core.runner import total
from core.schema import (ACTIONS, SCORE_KEYS, RunnerConfig, comparable_delta,
                         empty_verifier, verifier_value)
from .reward import RewardConfig, step_reward


@dataclass(frozen=True)
class Outcome:
    """One conditional draw, including aggregates for any retries in that draw.

    action_success describes generation, *before* automatic verification and
    budget settlement. scores contains only valid dimensions (None: no usable
    new check). A successful write can supply its candidate's quality proxy;
    the environment releases it only if the draft actually survives settlement.
    Use simulation_support for sampling provenance, never to mask an action.
    """
    action_success: bool
    input_tokens: int | None
    output_tokens: int | None
    verifier_tokens: int | None = 0
    scores: dict[str, float] | None = None
    estimated_final_quality: float | None = None
    quality_method: str | None = None
    simulation_support: dict | None = None

    def validate(self, action):
        for tokens in (self.input_tokens, self.output_tokens, self.verifier_tokens):
            if tokens is not None and (type(tokens) is not int or tokens < 0):
                raise ValueError("Token counts must be nonnegative integers or None.")
        if type(self.action_success) is not bool:
            raise ValueError("action_success must be boolean.")
        is_write = action.startswith("write_")
        if (is_write or not self.action_success) and (self.scores is not None or self.verifier_tokens != 0):
            raise ValueError("Failed generation and writing cannot run a Verifier.")
        if self.scores is not None:
            scope = {"coverage"} if action == "retrieve_more" else set(SCORE_KEYS)
            if set(self.scores) != scope or any(type(v) not in (int, float) or not 0 <= v <= 1
                                                for v in self.scores.values()):
                raise ValueError("Verifier scores must match scoring scope and be in [0,1].")
        if self.estimated_final_quality is not None and (
                not is_write or not self.action_success or type(self.estimated_final_quality) not in (int, float)
                or not 0 <= self.estimated_final_quality <= 1):
            raise ValueError("Only a generated draft can supply a bounded quality proxy.")


class SyntheticOutcomes:
    """Analytic fixture, not measured performance. Rates are per attempted stage.

    With default budget, >=3 steps and no failures, the best scripted path is
    retrieve, summarize, write. The two efforts are fast/fast for easy,
    deep/fast for medium, deep/deep for hard. Rework doesn't compound quality.
    """
    def __init__(self, action_failure_rate=0.0, verifier_failure_rate=0.0):
        for rate in (action_failure_rate, verifier_failure_rate):
            if type(rate) not in (int, float) or not 0 <= rate <= 1:
                raise ValueError("Failure rates must be in [0,1].")
        self.action_failure_rate = action_failure_rate
        self.verifier_failure_rate = verifier_failure_rate

    def __call__(self, before, action_id, rng):
        level = ("easy", "medium", "hard").index(before["difficulty"])
        coverage = (0.84, 0.77, 0.66)[level]
        deep = action_id.endswith("_deep")
        if action_id == "retrieve_more":
            scores = {"coverage": coverage}
            inputs, outputs = 0, 0  # local retrieval; verification is metered below
        else:
            write = action_id.startswith("write_")
            if write:
                inputs, outputs = (1200, 2400) if deep else (600, 800)
            else:
                inputs, outputs = (900, 1900) if deep else (500, 500)
            if rng.random() < self.action_failure_rate:
                return Outcome(False, inputs, outputs)
            if write:
                quality = verifier_value(before) if before["summary_available"] else None
                if quality is None:
                    quality = (0.48, 0.40, 0.30)[level]
                if deep:
                    quality = min(1.0, quality + (0.01, 0.02, 0.14)[level])
                return Outcome(True, inputs, outputs, estimated_final_quality=quality,
                               quality_method="analytic_fixture")
            score = (0.90, 0.90, 0.87)[level] if deep else (0.90, 0.66, 0.43)[level]
            scores = {k: coverage if k == "coverage" else score for k in SCORE_KEYS}
        if rng.random() < self.verifier_failure_rate:
            scores = None
        return Outcome(True, inputs, outputs, 360, scores)


class SyntheticEnv:
    """Gym-style raw-snapshot environment; all Runner terminal limits end the task.

    Inject `outcome_model(before, action_id, rng) -> Outcome` to replace *only*
    the distribution (e.g. sample a training bucket and its quality proxy).
    The callable receives a private copy of the pre-action snapshot. No sample
    can mark a transition as real. No episode or files are persisted.
    """
    def __init__(self, config=RunnerConfig(), *, outcome_model=None, corpus_size=9,
                 reward_config=RewardConfig(), source="synthetic"):
        if source not in ("synthetic", "calibrated") or (source == "calibrated" and outcome_model is None):
            raise ValueError("Use synthetic, or calibrated with an explicit simulation model.")
        self._check_size(corpus_size)
        self.config = config
        self.outcome_model = SyntheticOutcomes() if outcome_model is None else outcome_model
        self.corpus_size = corpus_size
        self.reward_config = reward_config
        self.source = source
        self._rng = random.Random()
        self._state = None

    @staticmethod
    def _check_size(size):
        if type(size) is not int or size < 1:
            raise ValueError("corpus_size must be a positive integer.")

    def reset(self, *, seed=None, options=None):
        options = {} if options is None else options
        if seed is not None:
            self._rng.seed(seed)
        difficulty = options["difficulty"] if "difficulty" in options else self._rng.choice(("easy", "medium", "hard"))
        if difficulty not in ("easy", "medium", "hard"):
            raise ValueError("Invalid difficulty.")
        review_id = options.get("review_id", f"synthetic-{difficulty}")
        if not isinstance(review_id, str) or not review_id.strip():
            raise ValueError("review_id must be a nonempty string.")
        size = options.get("corpus_size", self.corpus_size)
        self._check_size(size)
        self._size = size
        self._trajectory_id = options.get("trajectory_id") or f"sim-{self._rng.getrandbits(128):032x}"
        self._action_tokens = dict.fromkeys(ACTIONS, 0)
        self._state = {
            "config_version": self.config.config_version, "review_id": review_id, "difficulty": difficulty,
            "retrieved_paper_count": 0, "summary_available": False, "material_revision": 0,
            "verifier": empty_verifier(), "step_count": 0, "max_steps": self.config.max_steps,
            "workflow_tokens_used": 0, "workflow_tokens_budget": self.config.workflow_tokens_budget,
            "action_history": {a: {"call_count": 0, "last_verifier_delta": None,
                                   "consecutive_low_gain_or_failures": 0, "token_share": 0.0} for a in ACTIONS},
            "terminal": False, "terminal_reason": None,
        }
        return self.observe(), {"action_mask": self.valid_actions(), "source": self.source}

    def observe(self):
        if self._state is None:
            raise RuntimeError("Call reset() first.")
        return copy.deepcopy(self._state)

    def _mask(self, state):
        if state["terminal"]:
            return [False] * len(ACTIONS)
        count = state["retrieved_paper_count"]
        return [count < self._size] + [count > 0] * (len(ACTIONS) - 1)

    def valid_actions(self):
        if self._state is None:
            raise RuntimeError("Call reset() first.")
        return self._mask(self._state)

    def step(self, action_id):
        before = self.observe()
        mask = self.valid_actions()
        if action_id not in ACTIONS or not mask[ACTIONS.index(action_id)]:
            raise ValueError(f"Unknown or unavailable action: {action_id}")
        outcome = self.outcome_model(copy.deepcopy(before), action_id, self._rng)
        outcome.validate(action_id)
        after = copy.deepcopy(before)
        after["step_count"] += 1
        hist = after["action_history"][action_id]
        hist["call_count"] += 1
        write = action_id.startswith("write_")
        success = outcome.action_success
        verifier_tokens = outcome.verifier_tokens
        if success and not write:
            after["material_revision"] += 1
            after["summary_available"] = action_id != "retrieve_more"
            if action_id == "retrieve_more":
                after["retrieved_paper_count"] = min(self._size, before["retrieved_paper_count"] +
                                                       self.config.retrieval_batch_size)
            # Like Runner._can_call: if generation used up the budget or its
            # usage is missing, no verification starts and old scores go stale.
            used_before_check = total((before["workflow_tokens_used"], outcome.input_tokens, outcome.output_tokens))
            if used_before_check is None or used_before_check >= self.config.workflow_tokens_budget:
                verifier_tokens = 0
                success = False
            elif outcome.scores is None:
                success = False
            else:
                after["verifier"] = {"scores": {k: outcome.scores.get(k, 0.0) for k in SCORE_KEYS},
                                     "valid_scores": {k: k in outcome.scores for k in SCORE_KEYS},
                                     "checked_material_revision": after["material_revision"], "issues": []}

        step_tokens = total((outcome.input_tokens, outcome.output_tokens, verifier_tokens))
        action_tokens = self._action_tokens.copy()
        action_tokens[action_id] = total((action_tokens[action_id], step_tokens))
        used = after["workflow_tokens_used"] = total(action_tokens.values())
        reason = None
        if used is None or used >= self.config.workflow_tokens_budget:
            reason = "usage_unknown" if used is None else "token_budget"
            if write:
                success = False
        elif write and success:
            reason = "final_written"
        elif after["step_count"] >= self.config.max_steps:
            reason = "max_steps"
        done = reason is not None
        after.update(terminal=done, terminal_reason=reason)
        delta = comparable_delta(before, after) if success and not write else None
        hist["last_verifier_delta"] = delta
        if not success or (delta is not None and delta < self.config.low_gain_threshold):
            hist["consecutive_low_gain_or_failures"] += 1
        else:
            hist["consecutive_low_gain_or_failures"] = 0
        for a, h in after["action_history"].items():
            h["token_share"] = None if used is None else action_tokens[a] / used if used else 0.0

        next_mask = self._mask(after)
        quality = outcome.estimated_final_quality if reason == "final_written" else None
        transition = {
            "config_version": self.config.config_version, "review_id": before["review_id"],
            "trajectory_id": self._trajectory_id, "step_index": after["step_count"],
            "state_before": before, "state_after": after, "state_before_ref": None, "state_after_ref": None,
            "action_id": action_id, "action_mask": mask, "next_mask": next_mask,
            "verifier_before": verifier_value(before), "verifier_after": verifier_value(after), "verifier_delta": delta,
            "input_tokens": outcome.input_tokens, "output_tokens": outcome.output_tokens,
            "verifier_tokens": verifier_tokens, "total_workflow_tokens": step_tokens,
            "success": success, "done": done, "terminal_reason": reason,
            "source": self.source, "execution_kind": "simulated", "calls": [],
            "estimated_final_quality": quality, "quality_method": outcome.quality_method if quality is not None else None,
            "simulation_support": copy.deepcopy(outcome.simulation_support),
        }
        reward = step_reward(transition, self.reward_config)
        self._state, self._action_tokens = copy.deepcopy(after), action_tokens
        info = {"action_mask": next_mask.copy(), "source": self.source,
                "estimated_final_quality": quality, "transition": transition}
        return self.observe(), reward, done, False, info
