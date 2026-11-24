"""Explicit pilot action smoke test; not a training or evaluation policy."""

import argparse
import json
from pathlib import Path

from litreview.clients import DeepSeekBackend
from litreview.pipeline import load_corpus
from .runner import Runner
from .schema import ReviewTask, RunnerConfig


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("mock", "deepseek"), default="mock")
    parser.add_argument("--output", type=Path, default=root / "outputs/pilot/action-acceptance")
    args = parser.parse_args(argv)
    task = ReviewTask.from_dict(json.loads((root / "examples/runner_task.json").read_text(encoding="utf-8")))
    config = RunnerConfig(**json.loads((root / "config.json").read_text(encoding="utf-8")))
    backend = DeepSeekBackend.from_env() if args.backend == "deepseek" else None
    runner = Runner(load_corpus(root / "data/samples/agent_skill/agent_skill_papers.json"),
                    backend, config, args.output)
    runner.reset(task, seed=0)
    for action in ("retrieve_more", "summarize_fast", "summarize_deep", "write_deep"):
        step = runner.step(action)
        print(json.dumps({"trajectory_id": runner.trajectory_id, "action_id": action,
                          "success": step["success"], "tokens": step["total_workflow_tokens"],
                          "terminal_reason": step["terminal_reason"]}), flush=True)
        if not step["success"]:
            return 1
        if step["done"]:
            break
    return 0 if runner.result()["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
