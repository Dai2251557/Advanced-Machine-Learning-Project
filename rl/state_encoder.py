"""真实 Runner 与仿真环境共用的动作前状态编码。"""

from dataclasses import dataclass

from core.schema import ACTIONS, CONFIG_VERSION, SCORE_KEYS, WorkspaceSnapshot


DIFFICULTIES = ("easy", "medium", "hard")
HISTORY_FIELDS = ("call_fraction", "last_delta", "delta_valid", "low_gain_fraction", "token_share")
FEATURES = (
    *(f"difficulty.{d}" for d in DIFFICULTIES),
    "papers_scaled", "can_retrieve_more", "summary_available", "step_fraction",
    "tokens_known", "token_used_fraction", "token_remaining_fraction",
    "verifier_fresh", "verifier_stale",
    *(f"score.{k}" for k in SCORE_KEYS), *(f"score_valid.{k}" for k in SCORE_KEYS),
    *(f"{a}.{field}" for a in ACTIONS for field in HISTORY_FIELDS),
)


@dataclass(frozen=True)
class StateEncoder:
    # 默认尺度是当前配置的 8 步 × 每次 3 篇；不据此推断语料是否耗尽。
    paper_scale: int = 24

    def __post_init__(self):
        if type(self.paper_scale) is not int or self.paper_scale < 1:
            raise ValueError("paper_scale must be a positive integer")

    def spec(self):
        return {"config_version": CONFIG_VERSION, "action_order": list(ACTIONS),
                "features": list(FEATURES), "paper_scale": self.paper_scale}

    def encode(self, state: WorkspaceSnapshot, action_mask: list[bool]) -> tuple[float, ...]:
        if state["config_version"] != CONFIG_VERSION:
            raise ValueError("Unsupported config_version")
        if len(action_mask) != len(ACTIONS) or any(type(v) is not bool for v in action_mask):
            raise ValueError("Expected five boolean action mask entries")
        difficulty = state["difficulty"]
        if difficulty not in DIFFICULTIES:
            raise ValueError("Unknown difficulty")
        steps, budget = state["max_steps"], state["workflow_tokens_budget"]
        used = state["workflow_tokens_used"]
        verifier = state["verifier"]
        checked = verifier["checked_material_revision"]
        fresh = checked == state["material_revision"]
        valid = [fresh and verifier["valid_scores"][k] for k in SCORE_KEYS]
        values = [*(difficulty == d for d in DIFFICULTIES),
                  state["retrieved_paper_count"] / self.paper_scale, action_mask[0],
                  state["summary_available"], state["step_count"] / steps,
                  used is not None, used / budget if used is not None else 0.0,
                  max(0.0, 1 - used / budget) if used is not None else 0.0,
                  fresh, checked >= 0 and not fresh,
                  *(verifier["scores"][k] if ok else 0.0 for k, ok in zip(SCORE_KEYS, valid)), *valid]
        for action in ACTIONS:
            h = state["action_history"][action]
            delta = h["last_verifier_delta"]
            values.extend((h["call_count"] / steps, delta if delta is not None else 0.0,
                           delta is not None, h["consecutive_low_gain_or_failures"] / steps,
                           h["token_share"] if used is not None else 0.0))
        return tuple(map(float, values))
