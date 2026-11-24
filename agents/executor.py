"""Shared executors, bounded JSON repair and per-episode request cache."""

import copy
import hashlib
import json
import math
import time
from dataclasses import replace

from core.schema import SCORE_KEYS
from litreview.clients import Completion, DeepSeekBackend, DemoBackend
from litreview.pipeline import nonempty_text, require, validate_ids

SUMMARY_PROMPT = (
    'Summarize every supplied paper in Chinese, addressing the survey instruction. '
    'Use previous summaries and verifier feedback to revise when available. '
    'Return JSON {"items":[{"paper_id":"id","summary":"text","evidence":"exact excerpt"}]}. '
    'Cover each paper exactly once. Copy a nonempty evidence excerpt character-for-character '
    'from its abstract. Do not infer details absent from abstracts.'
)
WRITE_PROMPT = (
    'Write a Chinese literature review with comparisons, limitations and an explicit abstract-only scope. '
    'Use the supplied papers and summaries only. Cover all supplied papers with citations. '
    'Return JSON {"sections":[{"heading":"text","text":"text","paper_ids":["id"]}]}.'
)
VERIFY_PROMPT = (
    'Evaluate supplied materials against the survey instruction and original abstracts. '
    'Return JSON {"scores":{"coverage":0.0,"factuality":0.0,"citation_support":0.0,"consistency":0.0},'
    '"valid_scores":{"coverage":true,"factuality":false,"citation_support":false,"consistency":false},'
    '"issues":["specific actionable issue"]}. Never select a next_action. '
    'Each valid score is in [0,1], higher is better: coverage measures how much of the requested '
    'topic is supported by retrieved materials; factuality measures summary claims supported by abstracts; '
    'citation_support measures correct summary attribution and evidence support; consistency measures '
    'internal and cross-paper summary consistency without conflating different settings. '
    'Anchors: 0=no support/major errors, 0.5=partial coverage or substantial omissions/errors, '
    '1=complete support within supplied abstracts, no identified errors. '
    'With no summaries, evaluate coverage only; all other scores must be 0 and invalid. '
    'With summaries, evaluate all four dimensions. Do not treat missing source details as established facts. '
    'Scores are workflow diagnostics, not independent final quality judgments.'
)


class MockBackend(DemoBackend):
    """Synthetic responses and zero billed tokens: for interface tests only."""
    name = "mock"
    model = "synthetic-mock-not-for-experiments"

    def complete(self, role, instruction, payload):
        if role == "verify":
            has_summary = bool(payload["items"])
            result = Completion(json.dumps({
                "scores": {k: 0.8 if k == "coverage" or has_summary else 0.0 for k in SCORE_KEYS},
                "valid_scores": {k: k == "coverage" or has_summary for k in SCORE_KEYS},
                "issues": [],
            }))
        elif role == "write" and not payload["items"]:
            result = super().complete(role, instruction, {"items": [
                {"paper_id": p["id"], "summary": p["abstract"]} for p in payload["papers"]]})
        else:
            result = super().complete(role, instruction, payload)
        result.prompt_tokens = result.completion_tokens = 0
        result.model = self.model
        return result


class AgentExecutor:
    def __init__(self, backend, config):
        if isinstance(backend, DeepSeekBackend):
            if backend.base_url.rstrip("/") not in ("https://api.deepseek.com", "https://api.deepseek.com/v1"):
                raise ValueError("The shared Runner requires the official DeepSeek endpoint.")
            if backend.model != "deepseek-flash":
                raise ValueError("The shared Runner requires deepseek-flash.")
        self.backend, self.config = backend, config
        self.reset()

    def reset(self):
        self.events = []
        self.cache = {}

    def invoke(self, role, effort, instruction, payload, validator, can_call=lambda: True):
        client = (replace(self.backend, reasoning_effort=effort,
                          max_tokens=self.config.max_output_tokens)
                  if isinstance(self.backend, DeepSeekBackend) else self.backend)
        request_payload = copy.deepcopy(payload)
        for attempt in range(self.config.output_repair_attempts + 1):
            key = hashlib.sha256(json.dumps([self.config.config_version, client.model, role,
                                           effort, instruction, request_payload], sort_keys=True,
                                          ensure_ascii=False).encode()).hexdigest()
            cached = self.cache.get(key) if self.config.cache_enabled else None
            if cached is None and not can_call():
                raise RuntimeError("Workflow budget exhausted before next API call.")
            event = {"role": role, "model_requested": client.model, "model_actual": None,
                     "reasoning_effort": effort, "attempt": attempt + 1,
                     "prompt": instruction, "payload": copy.deepcopy(request_payload),
                     "cache_key": key, "cache_hit": cached is not None,
                     "prompt_tokens": None, "completion_tokens": None, "status": "error"}
            self.events.append(event)
            started = time.monotonic()
            try:
                completion = copy.deepcopy(cached) if cached else client.complete(role, instruction, request_payload)
                event.update(model_actual=completion.model,
                             prompt_tokens=0 if cached else completion.prompt_tokens,
                             completion_tokens=0 if cached else completion.completion_tokens,
                             finish_reason=completion.finish_reason,
                             response_text=completion.content)
                require(completion.finish_reason == "stop", "Output truncated or interrupted.")
                value = json.loads(completion.content)
                require(isinstance(value, dict), "Expected JSON object.")
                validator(value)
                event.update(status="ok", output=copy.deepcopy(value))
                if self.config.cache_enabled:
                    self.cache[key] = copy.deepcopy(completion)
                return value
            except (ValueError, TypeError, KeyError) as exc:
                # Validation feedback is generated locally, never transport exception text.
                event["error"] = "Invalid structured output: " + type(exc).__name__
                if attempt >= self.config.output_repair_attempts:
                    raise ValueError(event["error"]) from None
                request_payload = {**payload, "previous_output": event.get("response_text"),
                                   "validation_feedback": "Repair the JSON schema, exact evidence and citation coverage."}
            except Exception:
                event["error"] = "Transport or backend failure; " + type(self.backend).__name__
                raise RuntimeError(event["error"]) from None
            finally:
                event["elapsed_seconds"] = round(time.monotonic() - started, 6)

    def summarize(self, effort, payload, can_call):
        selected = {p["id"]: p for p in payload["papers"]}

        def validate(value):
            items = value.get("items")
            require(isinstance(items, list) and all(isinstance(i, dict) for i in items), "Invalid items.")
            validate_ids([i.get("paper_id") for i in items], selected)
            require(len(items) == len(selected), "Missing paper summary.")
            for item in items:
                require(nonempty_text(item.get("summary")), "Missing summary.")
                evidence = item.get("evidence")
                require(nonempty_text(evidence) and evidence in selected[item["paper_id"]]["abstract"],
                        "Evidence must be an exact excerpt.")
        return self.invoke("summarize", effort, SUMMARY_PROMPT, payload, validate, can_call)["items"]

    def write(self, effort, payload, can_call):
        selected = {p["id"] for p in payload["papers"]}

        def validate(value):
            sections = value.get("sections")
            require(isinstance(sections, list) and bool(sections), "Missing sections.")
            cited = set()
            for section in sections:
                require(isinstance(section, dict) and nonempty_text(section.get("heading"))
                        and nonempty_text(section.get("text")), "Invalid section.")
                validate_ids(section.get("paper_ids"), selected)
                cited.update(section["paper_ids"])
            require(cited == selected, "Missing paper citations.")
        return self.invoke("write", effort, WRITE_PROMPT, payload, validate, can_call)

    def verify(self, payload, revision, can_call):
        def validate(value):
            require("next_action" not in value, "Verifier cannot select actions.")
            scores, valid = value.get("scores"), value.get("valid_scores")
            require(isinstance(scores, dict) and set(scores) == set(SCORE_KEYS), "Invalid score keys.")
            require(isinstance(valid, dict) and set(valid) == set(SCORE_KEYS), "Invalid validity keys.")
            for key in SCORE_KEYS:
                require(type(scores[key]) in (int, float) and math.isfinite(scores[key])
                        and 0 <= scores[key] <= 1, "Invalid score.")
                require(type(valid[key]) is bool, "Invalid validity.")
                require(valid[key] == (key == "coverage" or bool(payload["items"])), "Invalid scoring scope.")
                require(valid[key] or scores[key] == 0, "Unevaluated score must be zero.")
            require(isinstance(value.get("issues"), list)
                    and all(nonempty_text(i) for i in value["issues"]), "Invalid issues.")
        result = self.invoke("verify", "low", VERIFY_PROMPT, payload, validate, can_call)
        return {"scores": result["scores"], "valid_scores": result["valid_scores"],
                "issues": result["issues"], "checked_material_revision": revision}
