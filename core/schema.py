"""Public wire format: snapshots/transitions are JSON-serializable dictionaries.

No Judge results or gold labels belong in any of these structures.
"""

from dataclasses import asdict, dataclass
from typing import Protocol, TypedDict
import math

CONFIG_VERSION = 1
ACTIONS = ("retrieve_more", "summarize_fast", "summarize_deep", "write_fast", "write_deep")
SCORE_KEYS = ("coverage", "factuality", "citation_support", "consistency")


@dataclass(frozen=True)
class ReviewTask:
    review_id: str
    survey_instruction: str
    split: str = "pilot"
    difficulty: str = "medium"
    corpus_scope: str = "local_arxiv_ai"
    config_version: int = CONFIG_VERSION

    def __post_init__(self):
        if self.config_version != CONFIG_VERSION:
            raise ValueError("Unsupported config_version.")
        if self.split not in ("train", "validation", "test", "pilot"):
            raise ValueError("Invalid split.")
        if self.difficulty not in ("easy", "medium", "hard"):
            raise ValueError("Invalid difficulty.")
        if any(not isinstance(v, str) or not v.strip() for v in
               (self.review_id, self.survey_instruction, self.corpus_scope)):
            raise ValueError("Task text fields must be nonempty.")

    @classmethod
    def from_dict(cls, value):
        # Whitelist fields: never forward evaluation_points/gold/Judge labels.
        return cls(**{k: value[k] for k in cls.__dataclass_fields__ if k in value})

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class RunnerConfig:
    config_version: int = CONFIG_VERSION
    max_steps: int = 8
    workflow_tokens_budget: int = 48000
    retrieval_batch_size: int = 3
    max_output_tokens: int = 8192
    output_repair_attempts: int = 1
    cache_enabled: bool = False
    low_gain_threshold: float = 0.02
    quality_threshold: float = 0.75
    max_summary_calls: int = 3

    def __post_init__(self):
        if self.config_version != CONFIG_VERSION:
            raise ValueError("Unsupported config_version.")
        for key in ("max_steps", "workflow_tokens_budget", "retrieval_batch_size",
                    "max_output_tokens", "max_summary_calls"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer.")
        if type(self.output_repair_attempts) is not int or not 0 <= self.output_repair_attempts <= 2:
            raise ValueError("output_repair_attempts must be 0..2.")
        if type(self.cache_enabled) is not bool:
            raise ValueError("cache_enabled must be boolean.")
        for key in ("low_gain_threshold", "quality_threshold"):
            if not math.isfinite(getattr(self, key)) or not 0 <= getattr(self, key) <= 1:
                raise ValueError(f"{key} must be in [0,1].")


class VerifierResult(TypedDict):
    scores: dict[str, float]
    valid_scores: dict[str, bool]
    checked_material_revision: int
    issues: list[str]


class ActionHistory(TypedDict):
    call_count: int
    last_verifier_delta: float | None
    consecutive_low_gain_or_failures: int
    token_share: float | None


class WorkspaceSnapshot(TypedDict):
    config_version: int
    review_id: str
    difficulty: str
    retrieved_paper_count: int
    summary_available: bool
    material_revision: int
    verifier: VerifierResult
    step_count: int
    max_steps: int
    workflow_tokens_used: int | None
    workflow_tokens_budget: int
    action_history: dict[str, ActionHistory]
    terminal: bool
    terminal_reason: str | None


class StepTransition(TypedDict):
    config_version: int
    review_id: str
    trajectory_id: str
    step_index: int
    state_before: WorkspaceSnapshot
    state_after: WorkspaceSnapshot
    state_before_ref: str | None
    state_after_ref: str | None
    action_id: str
    action_mask: list[bool]
    verifier_before: float | None
    verifier_after: float | None
    verifier_delta: float | None
    input_tokens: int | None
    output_tokens: int | None
    verifier_tokens: int | None
    total_workflow_tokens: int | None
    success: bool
    done: bool
    terminal_reason: str | None
    calls: list[dict]


class EpisodeResult(TypedDict):
    config_version: int
    review_id: str
    trajectory_id: str
    success: bool
    terminal_reason: str | None
    final_text: str | None
    final_text_ref: str | None
    total_workflow_tokens: int | None
    step_count: int
    action_history: dict[str, ActionHistory]
    transitions: list[StepTransition]


class BaseScheduler(Protocol):
    def reset(self, seed: int | None = None): ...
    def select_action(self, state: WorkspaceSnapshot, action_mask: list[bool]) -> str: ...


def empty_verifier() -> VerifierResult:
    return {"scores": dict.fromkeys(SCORE_KEYS, 0.0),
            "valid_scores": dict.fromkeys(SCORE_KEYS, False),
            "checked_material_revision": -1, "issues": []}


def verifier_value(snapshot):
    """Equal-weight mean of valid dimensions, only for current materials."""
    v = snapshot["verifier"]
    if v["checked_material_revision"] != snapshot["material_revision"]:
        return None
    values = [v["scores"][k] for k in SCORE_KEYS if v["valid_scores"][k]]
    return sum(values) / len(values) if values else None


def comparable_delta(before, after):
    left, right = verifier_value(before), verifier_value(after)
    if (left is None or right is None or
            before["verifier"]["valid_scores"] != after["verifier"]["valid_scores"]):
        return None
    return right - left
