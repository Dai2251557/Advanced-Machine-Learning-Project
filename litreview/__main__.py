import argparse
import json
import sys
from pathlib import Path

from .clients import DeepSeekBackend, DemoBackend
from .pipeline import FixedRulePipeline, load_corpus, load_task


def main(argv=None):
    parser = argparse.ArgumentParser(description="Four-agent fixed-rule literature review baseline")
    parser.add_argument("--backend", choices=("demo", "deepseek"), default="demo")
    parser.add_argument("--corpus", type=Path, required=True)
    task_input = parser.add_mutually_exclusive_group(required=True)
    task_input.add_argument("--query")
    task_input.add_argument("--task-file", type=Path)
    parser.add_argument("--task-id")
    parser.add_argument("--top-k", type=int, help="Override the task top_k; defaults to 3 with --query")
    parser.add_argument("--output", type=Path, required=True,
                        help="A new directory, to keep previous run artifacts intact")
    args = parser.parse_args(argv)
    if bool(args.task_file) != bool(args.task_id):
        parser.error("--task-file and --task-id must be supplied together")
    pipeline = None
    output_created = False
    exit_code = 1
    try:
        corpus = load_corpus(args.corpus)
        task = load_task(args.task_file, args.task_id, corpus) if args.task_file else None
        query = task["query"] if task else args.query
        top_k = args.top_k if args.top_k is not None else (task["top_k"] if task else 3)
        backend = DemoBackend() if args.backend == "demo" else DeepSeekBackend.from_env()
        pipeline = FixedRulePipeline(backend)
        args.output.mkdir(parents=True, exist_ok=False)
        output_created = True
        result = pipeline.run(corpus, query, top_k)
        (args.output / "report.md").write_text(result["report"], encoding="utf-8")
        (args.output / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved report and trace to {args.output.resolve()}")
        exit_code = 0
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"Run failed: {exc}", file=sys.stderr)
    finally:
        if pipeline is not None and output_created:
            trace = pipeline.trace()
            trace.update(query=query, dataset_kind=corpus["dataset_kind"], top_k=top_k, task_id=args.task_id,
                         data_purpose=corpus.get("purpose", "unspecified"),
                         status="success" if exit_code == 0 else "error")
            try:
                (args.output / "trace.json").write_text(
                    json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
            except OSError:
                print("Could not save trace.json.", file=sys.stderr)
                exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
