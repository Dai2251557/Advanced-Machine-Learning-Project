"""Run the shared Runner: python -m core --help."""

import argparse
import json
from pathlib import Path

from litreview.clients import DeepSeekBackend
from litreview.pipeline import load_corpus
from .runner import Runner
from .rule_scheduler import RuleScheduler
from .schema import ReviewTask, RunnerConfig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("mock", "deepseek"), default="mock")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--task", type=Path, required=True, help="One public ReviewTask JSON object")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "config.json")
    parser.add_argument("--output", type=Path, default=Path("outputs/runner"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    config = RunnerConfig(**json.loads(args.config.read_text(encoding="utf-8-sig")))
    task = ReviewTask.from_dict(json.loads(args.task.read_text(encoding="utf-8-sig")))
    backend = DeepSeekBackend.from_env() if args.backend == "deepseek" else None
    runner = Runner(load_corpus(args.corpus), backend, config, args.output)
    result = runner.run_episode(RuleScheduler(config), task, args.seed)
    print(json.dumps({k: result[k] for k in ("trajectory_id", "success", "terminal_reason",
                                           "total_workflow_tokens", "final_text_ref")}, ensure_ascii=False))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
