"""Policy-independent execution; no Judge or training labels are consumed here."""

import copy
import json
import re
import uuid
from dataclasses import asdict
from pathlib import Path

from agents.executor import AgentExecutor, MockBackend
from .schema import (ACTIONS, ReviewTask, RunnerConfig, comparable_delta,
                     empty_verifier, verifier_value)


def total(values):
    values = list(values)
    return None if any(v is None for v in values) else sum(values)


class Runner:
    def __init__(self, corpus, backend=None, config=None, output_dir=None):
        self.config = config or RunnerConfig()
        # Only source text is made visible to execution agents.
        self.papers = [{k: p[k] for k in ("id", "title", "abstract")} for p in corpus["papers"]]
        if (not self.papers or len({p["id"] for p in self.papers}) != len(self.papers)
                or any(not isinstance(v, str) or not v.strip() for p in self.papers for v in p.values())):
            raise ValueError("Corpus requires unique IDs and nonempty source texts.")
        self.dataset_kind = corpus.get("dataset_kind", "unknown")
        self.data_purpose = corpus.get("purpose", "unspecified")
        self.executor = AgentExecutor(backend if backend is not None else MockBackend(), self.config)
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.task = None

    def reset(self, review_task, seed=None):
        task = review_task if isinstance(review_task, ReviewTask) else ReviewTask.from_dict(review_task)
        if self.data_purpose == "pilot_only" and task.split != "pilot":
            raise ValueError("pilot_only corpus is for functional testing only; use split=pilot, not train/validation/test.")
        self.task = task
        self.seed = seed
        self.trajectory_id = "episode-" + uuid.uuid4().hex
        self.episode_dir = None
        if self.output_dir is not None:
            self.episode_dir = self.output_dir / self.trajectory_id
            self.episode_dir.mkdir(parents=True, exist_ok=False)
        self.executor.reset()
        self.selected, self.items, self.transitions = [], [], []
        self.summary_revision = -1
        self.final_text = None
        self.action_tokens = dict.fromkeys(ACTIONS, 0)
        self.state = {
            "config_version": self.config.config_version, "review_id": task.review_id,
            "difficulty": task.difficulty, "retrieved_paper_count": 0,
            "summary_available": False, "material_revision": 0, "verifier": empty_verifier(),
            "step_count": 0, "max_steps": self.config.max_steps,
            "workflow_tokens_used": 0, "workflow_tokens_budget": self.config.workflow_tokens_budget,
            "action_history": {a: {"call_count": 0, "last_verifier_delta": None,
                                   "consecutive_low_gain_or_failures": 0, "token_share": 0.0}
                               for a in ACTIONS},
            "terminal": False, "terminal_reason": None,
        }
        self._persist()
        return self.observe()

    def observe(self):
        self._require_reset()
        return copy.deepcopy(self.state)

    def _require_reset(self):
        if self.task is None:
            raise RuntimeError("Call reset(review_task) first.")

    def action_unavailable_reasons(self):
        self._require_reset()
        if self.state["terminal"]:
            return {a: "episode_terminal" for a in ACTIONS}
        return {a: ("corpus_exhausted" if a == "retrieve_more" and len(self.selected) == len(self.papers)
                    else "no_source_papers" if a != "retrieve_more" and not self.selected else None)
                for a in ACTIONS}

    def valid_actions(self):
        reasons = self.action_unavailable_reasons()
        return [reasons[a] is None for a in ACTIONS]

    def _can_call(self):
        used = total(total((e["prompt_tokens"], e["completion_tokens"])) for e in self.executor.events)
        return used is not None and used < self.config.workflow_tokens_budget

    def _payload(self):
        return {"query": self.task.survey_instruction, "papers": copy.deepcopy(self.selected),
                "items": copy.deepcopy(self.items), "feedback": copy.deepcopy(self.state["verifier"]["issues"])}

    def _retrieve(self):
        # Deterministic lexical ranking; no gold IDs, network calls or model cost.
        terms = set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", self.task.survey_instruction.lower()))
        existing = {p["id"] for p in self.selected}
        candidates = [p for p in self.papers if p["id"] not in existing]
        def rank(p):
            words = set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", (p["title"] + " " + p["abstract"]).lower()))
            return (-len(words & terms), p["id"])
        self.selected.extend(sorted(candidates, key=rank)[:self.config.retrieval_batch_size])
        self.state["material_revision"] += 1
        self.state["retrieved_paper_count"] = len(self.selected)
        # Existing summaries become stale when the source set changes.
        self.items = []
        self.state["summary_available"] = False

    def _terminate(self, reason):
        self.state.update(terminal=True, terminal_reason=reason)

    def step(self, action_id):
        self._require_reset()
        if action_id not in ACTIONS:
            raise ValueError("Unknown action_id; expected one of ACTIONS.")
        mask = self.valid_actions()
        if not mask[ACTIONS.index(action_id)]:
            raise ValueError(f"Action {action_id} unavailable: {self.action_unavailable_reasons()[action_id]}")
        before = self.observe()
        event_start = len(self.executor.events)
        self.state["step_count"] += 1
        history = self.state["action_history"][action_id]
        history["call_count"] += 1
        success, error = False, None
        is_write = action_id.startswith("write_")
        try:
            effort = "high" if action_id.endswith("_deep") else "low"
            if action_id == "retrieve_more":
                self._retrieve()
            elif action_id.startswith("summarize_"):
                items = self.executor.summarize(effort, self._payload(), self._can_call)
                self.items = items
                self.state["material_revision"] += 1
                self.summary_revision = self.state["material_revision"]
                self.state["summary_available"] = True
            else:
                draft = self.executor.write(effort, self._payload(), self._can_call)
                lines = ["# 文献综述", "", "范围：仅依据提供的论文摘要。", ""]
                if self.executor.backend.name == "mock":
                    lines += ["> MOCK 合成演示，不是实际模型输出或实验结果。", ""]
                if self.data_purpose == "pilot_only":
                    lines += ["> PILOT 功能测试样例，不属于正式训练、标定或独立测试结果。", ""]
                for section in draft["sections"]:
                    lines += ["## " + section["heading"], "", section["text"] + " " +
                              " ".join("[" + p + "]" for p in section["paper_ids"]), ""]
                lines += ["## 参考文献", ""] + ["- [" + p["id"] + "] " + p["title"] for p in self.selected]
                self.final_text = "\n".join(lines) + "\n"
                self._terminate("final_written")
            if not is_write:
                self.state["verifier"] = self.executor.verify(
                    self._payload(), self.state["material_revision"], self._can_call)
            success = True
        except (ValueError, RuntimeError) as exc:
            error = str(exc)
        calls = copy.deepcopy(self.executor.events[event_start:])
        workflow = [e for e in calls if e["role"] != "verify"]
        verifiers = [e for e in calls if e["role"] == "verify"]
        input_tokens = total(e["prompt_tokens"] for e in workflow)
        output_tokens = total(e["completion_tokens"] for e in workflow)
        verifier_tokens = total(total((e["prompt_tokens"], e["completion_tokens"])) for e in verifiers)
        step_tokens = total((input_tokens, output_tokens, verifier_tokens))
        self.action_tokens[action_id] = total((self.action_tokens[action_id], step_tokens))
        self.state["workflow_tokens_used"] = total(self.action_tokens.values())
        used = self.state["workflow_tokens_used"]
        if not self.state["terminal"]:
            if used is None:
                self._terminate("usage_unknown")
            elif used >= self.config.workflow_tokens_budget:
                self._terminate("token_budget")
            elif self.state["step_count"] >= self.config.max_steps:
                self._terminate("max_steps")
        after = self.observe()
        delta = comparable_delta(before, after) if success and not is_write else None
        history["last_verifier_delta"] = delta
        if not success or (delta is not None and delta < self.config.low_gain_threshold):
            history["consecutive_low_gain_or_failures"] += 1
        else:
            history["consecutive_low_gain_or_failures"] = 0
        for action in ACTIONS:
            self.state["action_history"][action]["token_share"] = (
                None if used is None else self.action_tokens[action] / used if used else 0.0)
        after = self.observe()
        transition = {
            "config_version": self.config.config_version, "review_id": self.task.review_id,
            "trajectory_id": self.trajectory_id, "step_index": self.state["step_count"],
            "state_before": before, "state_after": after,
            "state_before_ref": None, "state_after_ref": None,
            "action_id": action_id, "action_mask": mask,
            "verifier_before": verifier_value(before), "verifier_after": verifier_value(after),
            "verifier_delta": delta, "input_tokens": input_tokens, "output_tokens": output_tokens,
            "verifier_tokens": verifier_tokens, "total_workflow_tokens": step_tokens,
            "success": success, "error": error, "done": self.state["terminal"],
            "terminal_reason": self.state["terminal_reason"], "calls": calls,
            "execution_kind": "mock" if self.executor.backend.name == "mock" else "real",
        }
        if self.episode_dir:
            prefix = f"step-{self.state['step_count']:03d}"
            for name, value in (("before", before), ("after", after)):
                path = self.episode_dir / (prefix + "-" + name + ".json")
                self._json(path, value)
                transition[f"state_{name}_ref"] = str(path.resolve())
            self._json(self.episode_dir / (prefix + ".json"), transition)
        self.transitions.append(transition)
        self._persist()
        return copy.deepcopy(transition)

    def run_episode(self, policy, review_task, seed=None):
        self.reset(review_task, seed)
        policy.reset(seed=seed)
        while not self.state["terminal"]:
            action = policy.select_action(self.observe(), self.valid_actions())
            self.step(action)
        return self.result()

    def result(self):
        self._require_reset()
        return copy.deepcopy({
            "config_version": self.config.config_version, "review_id": self.task.review_id,
            "trajectory_id": self.trajectory_id, "split": self.task.split, "seed": self.seed,
            "execution_kind": "mock" if self.executor.backend.name == "mock" else "real",
            "dataset_kind": self.dataset_kind, "data_purpose": self.data_purpose,
            "config": asdict(self.config),
            "success": self.state["terminal_reason"] == "final_written",
            "terminal_reason": self.state["terminal_reason"], "final_text": self.final_text,
            "final_text_ref": str((self.episode_dir / "report.md").resolve())
                if self.episode_dir and self.final_text else None,
            "total_workflow_tokens": self.state["workflow_tokens_used"],
            "step_count": self.state["step_count"], "action_history": self.state["action_history"],
            "source_papers": self.selected, "transitions": self.transitions,
        })

    @staticmethod
    def _json(path, value):
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    def _persist(self):
        if self.episode_dir:
            self._json(self.episode_dir / "episode.json", self.result())
            self._json(self.episode_dir / "workspace.json", {
                "state": self.state, "papers": self.selected, "items": self.items,
                "summary_revision": self.summary_revision,
            })
            if self.final_text:
                (self.episode_dir / "report.md").write_text(self.final_text, encoding="utf-8")
