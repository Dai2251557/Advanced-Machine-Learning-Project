"""Four role agents with fixed dispatch, provenance checks and per-call tracing."""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from .clients import Backend


def nonempty_text(value):
    return isinstance(value, str) and bool(value.strip())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_corpus(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    require(isinstance(data, dict), "Corpus must be a JSON object.")
    require(data.get("dataset_kind") in ("illustrative", "arxiv_abstracts"),
            "dataset_kind must be illustrative or arxiv_abstracts.")
    papers = data.get("papers")
    require(isinstance(papers, list) and bool(papers), "papers must be a nonempty list.")
    seen = set()
    for paper in papers:
        require(isinstance(paper, dict), "Every paper must be an object.")
        for key in ("id", "title", "abstract"):
            require(nonempty_text(paper.get(key)), f"Paper field {key} must be nonempty text.")
        require(paper["id"] not in seen, "Duplicate paper ID.")
        require(all(char.isalnum() or char in "-._/" for char in paper["id"]),
                "Paper ID must contain only letters, digits, -, ., _ or /.")
        seen.add(paper["id"])
    return data


def validate_ids(value, allowed, limit=None):
    require(isinstance(value, list) and bool(value), "Expected nonempty paper ID list.")
    require(all(isinstance(pid, str) for pid in value), "Paper IDs must be strings.")
    require(len(set(value)) == len(value), "Duplicate paper ID in agent output.")
    require(set(value) <= set(allowed), "Agent returned an unknown paper ID.")
    require(limit is None or len(value) <= limit, "Agent exceeded top_k.")


def load_task(path: Path, task_id: str, corpus: dict) -> dict:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    require(isinstance(data, dict) and isinstance(data.get("tasks"), list)
            and bool(data["tasks"]), "Task file must contain a nonempty tasks list.")
    seen = {}
    for task in data["tasks"]:
        require(isinstance(task, dict) and nonempty_text(task.get("id")), "Invalid task ID.")
        require(task["id"] not in seen, "Duplicate task ID.")
        require(nonempty_text(task.get("query")), "Task query must be nonempty text.")
        require(type(task.get("top_k")) is int and task["top_k"] > 0,
                "Task top_k must be a positive integer.")
        if "expected_paper_ids" in task:
            validate_ids(task["expected_paper_ids"], [paper["id"] for paper in corpus["papers"]])
        seen[task["id"]] = task
    require(task_id in seen, "Task ID was not found in the task file.")
    return seen[task_id]


@dataclass
class FixedRulePipeline:
    backend: Backend
    events: list = field(default_factory=list, init=False)

    def invoke(self, role, instruction, payload, validator):
        event = {
            "role": role, "agent_id": role + ":" + self.backend.model,
            "attempt": 1 + sum(previous["role"] == role for previous in self.events),
            "input_chars": len(json.dumps(payload, ensure_ascii=False)),
            "prompt_tokens": None, "completion_tokens": None, "status": "error",
        }
        started = time.monotonic()
        self.events.append(event)
        try:
            completion = self.backend.complete(role, instruction, payload)
            event.update(prompt_tokens=completion.prompt_tokens,
                         completion_tokens=completion.completion_tokens)
            require(completion.finish_reason == "stop", "Agent output was truncated or interrupted.")
            require(nonempty_text(completion.content), "Agent returned empty content.")
            value = json.loads(completion.content)
            require(isinstance(value, dict), "Agent output must be a JSON object.")
            event["output"] = value
            validator(value)
            event["status"] = "ok"
            return value
        finally:
            event["elapsed_seconds"] = round(time.monotonic() - started, 6)

    def run(self, corpus, query, top_k=3):
        self.events.clear()
        require(nonempty_text(query), "query must be nonempty text.")
        require(type(top_k) is int and top_k > 0, "top_k must be a positive integer.")
        papers = {p["id"]: p for p in corpus["papers"]}
        retrieval = self.invoke(
            "retrieve", 'Select relevant papers for the query. Return JSON {"paper_ids":["id"]}. '
            'Return at most top_k IDs, or an empty list when nothing is relevant.',
            {"query": query, "papers": list(papers.values()), "top_k": top_k},
            lambda obj: validate_ids(obj.get("paper_ids"), papers, top_k),
        )
        selected = {pid: papers[pid] for pid in retrieval["paper_ids"]}

        def validate_summaries(obj):
            items = obj.get("items")
            require(isinstance(items, list) and bool(items), "Missing summary items.")
            require(all(isinstance(item, dict) for item in items), "Invalid summary item.")
            validate_ids([item.get("paper_id") for item in items], selected)
            require(len(items) == len(selected), "Summaries must cover all selected papers.")
            for item in items:
                require(nonempty_text(item.get("summary")), "Missing summary text.")
                evidence = item.get("evidence")
                require(nonempty_text(evidence) and evidence in selected[item["paper_id"]]["abstract"],
                        f'Summary evidence for {item["paper_id"]} must be an exact excerpt '
                        'from its abstract, including capitalization and punctuation.')

        summary_instruction = (
            'Summarize each paper in Chinese without inferring missing details. '
            'Return JSON {"items":[{"paper_id":"id","summary":"text",'
            '"evidence":"exact nonempty excerpt from the abstract"}]}. Cover each supplied paper once. '
            'Copy evidence character-for-character, preserving capitalization and punctuation; '
            'do not lowercase sentence beginnings, paraphrase, or insert ellipses.'
        )
        summary_payload = {"query": query, "papers": list(selected.values())}
        try:
            summaries = self.invoke("summarize", summary_instruction, summary_payload, validate_summaries)
        except ValueError as exc:
            # One bounded output repair, preserving both attempts and their billing in the trace.
            summaries = self.invoke(
                "summarize", summary_instruction + ' Repair the invalid previous output using '
                'the validation feedback. All evidence must still be copied from the original abstracts.',
                {**summary_payload, "previous_output": self.events[-1].get("output"),
                 "validation_feedback": str(exc)}, validate_summaries,
            )

        def validate_sections(obj):
            sections = obj.get("sections")
            require(isinstance(sections, list) and bool(sections), "Missing report sections.")
            cited = set()
            for section in sections:
                require(isinstance(section, dict), "Invalid report section.")
                require(nonempty_text(section.get("heading")) and nonempty_text(section.get("text")),
                        "Report sections need a heading and text.")
                validate_ids(section.get("paper_ids"), selected)
                cited.update(section["paper_ids"])
            require(cited == set(selected), "Report must cover all selected papers.")

        draft = self.invoke(
            "write", 'Write a Chinese literature review with comparisons and limitations, based only '
            'on supplied summaries. Cite every section, cover every paper, and avoid unsupported claims. '
            'Return JSON {"sections":[{"heading":"text","text":"text","paper_ids":["id"]}]}.',
            {"query": query, "items": summaries["items"]}, validate_sections,
        )

        def validate_review(obj):
            require(type(obj.get("approved")) is bool, "Verifier approved must be a boolean.")
            require(isinstance(obj.get("issues"), list)
                    and all(nonempty_text(issue) for issue in obj["issues"]), "Invalid verifier issues.")
            require(obj["approved"] and not obj["issues"], "Verifier rejected the draft; review trace.json.")

        verification = self.invoke(
            "verify", 'Check the summaries and final draft against the original abstracts. Reject '
            'unsupported claims, misleading citations, contradictions and invented details. '
            'Return JSON {"approved":true,"issues":[]} or {"approved":false,"issues":["reason"]}.',
            {"query": query, "papers": list(selected.values()),
             "summaries": summaries["items"], "draft": draft},
            # Keep reviewer feedback even when the run is rejected.
            lambda obj: self.check_and_record_review(obj, validate_review),
        )
        lines = ["# 文献综述", "", query, "", "范围：仅依据所提供的论文摘要。", ""]
        if self.backend.name == "demo" or corpus["dataset_kind"] == "illustrative":
            lines += ["> 演示输出：含离线流程或虚构示例数据，不能作为正式实验结果。", ""]
        for section in draft["sections"]:
            citations = " ".join("[" + pid + "]" for pid in section["paper_ids"])
            lines += ["## " + section["heading"], "", section["text"] + " " + citations, ""]
        lines += ["## 参考文献", ""]
        lines += [f'- [{pid}] {paper["title"]}' for pid, paper in selected.items()]
        return {
            "report": "\n".join(lines) + "\n", "paper_ids": list(selected),
            "summaries": summaries["items"], "draft": draft, "verification": verification,
        }

    def check_and_record_review(self, obj, validator):
        self.events[-1]["verification"] = obj
        validator(obj)

    def trace(self):
        # Missing billing information stays unknown, rather than becoming a false zero.
        totals = {}
        for key in ("prompt_tokens", "completion_tokens"):
            values = [event[key] for event in self.events]
            totals[key] = sum(values) if values and all(v is not None for v in values) else None
        return {"scheduler": "fixed_rule", "backend": self.backend.name,
                "model": self.backend.model, "events": self.events, "usage": totals,
                "quality_score": None}
