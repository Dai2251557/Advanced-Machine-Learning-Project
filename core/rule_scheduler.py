"""Fixed state-aware baseline; no LLM routing or Judge access."""

from .schema import ACTIONS, RunnerConfig, SCORE_KEYS, verifier_value


class RuleScheduler:
    def __init__(self, config=None):
        self.config = config or RunnerConfig()

    def reset(self, seed=None):
        pass

    def select_action(self, state, action_mask):
        if len(action_mask) != len(ACTIONS) or not any(action_mask):
            raise ValueError("No physically executable action.")
        allowed = {a for a, valid in zip(ACTIONS, action_mask) if valid}
        summaries = sum(state["action_history"][a]["call_count"] for a in ACTIONS[1:3])
        if not state["retrieved_paper_count"]:
            preferred = "retrieve_more"
        elif state["step_count"] >= state["max_steps"] - 1:
            preferred = "write_fast"
        elif summaries >= self.config.max_summary_calls:
            preferred = "write_fast"
        elif not state["summary_available"]:
            preferred = "summarize_fast"
        else:
            score = verifier_value(state)
            v = state["verifier"]
            if (score is not None and all(v["valid_scores"][k] for k in SCORE_KEYS)
                    and score >= self.config.quality_threshold):
                preferred = "write_fast"
            elif (score is not None and v["scores"]["coverage"] < self.config.quality_threshold
                  and "retrieve_more" in allowed):
                preferred = "retrieve_more"
            else:
                preferred = "summarize_deep"
        if preferred in allowed:
            return preferred
        return next(a for a in ACTIONS if a in allowed)
