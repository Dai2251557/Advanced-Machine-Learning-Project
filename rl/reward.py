"""Simulation reward, charged on step usage, never on cumulative episode usage."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RewardConfig:
    quality_weight: float = 1.0
    cost_weight: float = 1.0
    failure_penalty: float = 1.0
    # A fixed missing-meter penalty, not a token estimate or a measured cost.
    unknown_usage_penalty: float = 1.0

    def __post_init__(self):
        for value in vars(self).values():
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("Reward weights must be finite and nonnegative.")


def step_reward(transition, config=RewardConfig()) -> float:
    """Use only this step's simulated workflow cost and eligible terminal proxy.

    Missing quality on an accepted draft requires explicit cost-only mode;
    a missing meter instead incurs a separate, disclosed penalty. Neither
    missing value is recorded as a measured zero.
    """
    tokens = transition["total_workflow_tokens"]
    budget = transition["state_before"]["workflow_tokens_budget"]
    reward = (-config.unknown_usage_penalty if tokens is None else
              -config.cost_weight * tokens / budget)
    quality = transition.get("estimated_final_quality")
    if transition["done"] and transition["terminal_reason"] == "final_written":
        if quality is None:
            if config.quality_weight:
                raise ValueError("A final quality proxy is required unless quality_weight is zero.")
        elif type(quality) not in (int, float) or not 0 <= quality <= 1:
            raise ValueError("Final quality proxy must be finite and in [0,1].")
        else:
            reward += config.quality_weight * quality
    elif quality is not None:
        raise ValueError("Only an accepted final draft may have a quality proxy.")
    elif transition["done"]:
        reward -= config.failure_penalty
    return float(reward)
