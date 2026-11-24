from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .config import discover_projects, load_config
from .delegate import delegate, format_transcript, list_logs
from .eval import (
    DEFAULT_EVALSET,
    DEFAULT_KS,
    build_report,
    format_comparison,
    format_judge_disagreements,
    format_report,
    judge_results,
    judged_metrics,
    load_evalset,
    preflight,
    run_cases,
)
from .index_state import IndexState
from .indexer import index_project
from .multiquery import multi_query_search
from .ollama_client import OllamaClient
from .rerank import RETRIEVAL_K, rerank_results
from .store import Store

VECTOR_DIM = 768


def cmd_index(args: argparse.Namespace) -> None:
    cfg = load_config()
    ollama = OllamaClient(cfg.ollama_url, cfg.embed_model)
    if not ollama.ping():
        print(f"error: cannot reach Ollama at {cfg.ollama_url}", file=sys.stderr)
        sys.exit(1)

    store = Store(cfg.db_path, VECTOR_DIM)
    state = IndexState(cfg.db_path / "index_state.json")

    projects = discover_projects(cfg)
    if args.project:
        if args.project not in projects:
            print(
                f"error: project '{args.project}' not found under configured roots",
                file=sys.stderr,
            )
            sys.exit(1)
        projects = {args.project: projects[args.project]}

    if not projects:
        print("no projects found; check minsky.toml [index] roots", file=sys.stderr)
        sys.exit(1)

    total_start = time.time()
    for name, root in projects.items():
        start = time.time()
        stats = index_project(name, root, cfg, store, state, ollama)
        state.save()
        elapsed = time.time() - start
        print(
            f"{name}: +{stats['files_updated']} files, "
            f"-{stats['files_removed']} removed, "
            f"{stats['commits_indexed']} commits ({elapsed:.1f}s)"
        )
    store.optimize()
    print(f"done in {time.time() - total_start:.1f}s")


def cmd_compact(_args: argparse.Namespace) -> None:
    cfg = load_config()
    store = Store(cfg.db_path, VECTOR_DIM)
    store.optimize()
    print("compacted")


def cmd_search(args: argparse.Namespace) -> None:
    cfg = load_config()
    ollama = OllamaClient(cfg.ollama_url, cfg.embed_model)
    store = Store(cfg.db_path, VECTOR_DIM)

    candidate_k = max(args.top_k, RETRIEVAL_K) if args.rerank else args.top_k
    if args.multi_query:
        results = multi_query_search(
            ollama, cfg.worker_model, store, args.query,
            top_k=candidate_k, project=args.project, kind=args.kind,
        )
    else:
        vector = ollama.embed(args.query)
        results = store.search(vector, top_k=candidate_k, project=args.project, kind=args.kind)
    if args.rerank:
        results = rerank_results(ollama, cfg.worker_model, args.query, results, top_k=args.top_k)
    if not results:
        print("no results")
        return
    for r in results:
        print(f"[{r['project']}] {r['path']}:{r['start_line']} ({r['kind']})")
        snippet = r["content"].strip().replace("\n", " ")
        print(f"  {snippet[:200]}")
        print()


def cmd_projects(_args: argparse.Namespace) -> None:
    cfg = load_config()
    projects = discover_projects(cfg)
    for name, root in sorted(projects.items()):
        print(f"{name}\t{root}")


def cmd_delegate(args: argparse.Namespace) -> None:
    cfg = load_config()
    ollama = OllamaClient(cfg.ollama_url, cfg.worker_model)
    if not ollama.ping():
        print(f"error: cannot reach Ollama at {cfg.ollama_url}", file=sys.stderr)
        sys.exit(1)

    prompt = args.prompt_file.read_text() if args.prompt_file else args.prompt
    if not prompt:
        print("error: provide a prompt or --prompt-file", file=sys.stderr)
        sys.exit(1)
    context = args.context.read_text() if args.context else None
    model = args.model or cfg.worker_model

    response = delegate(
        ollama,
        model,
        prompt,
        context=context,
        db_path=cfg.db_path,
        show=not args.quiet,
    )

    if args.out:
        args.out.write_text(response)
        print(f"wrote {len(response)} chars to {args.out}")
    else:
        print(response)


def cmd_log(args: argparse.Namespace) -> None:
    cfg = load_config()
    logs = list_logs(cfg.db_path)
    if not logs:
        print("no delegate logs yet")
        return

    if args.last:
        logs = logs[-args.last :]

    for path in logs:
        record = json.loads(path.read_text())
        if args.list:
            preview = record["prompt"].strip().replace("\n", " ")[:80]
            print(f"{path.stem}\t{record['timestamp']}\t{record['model']}\t{preview}")
        else:
            print(format_transcript(record))
            print("=" * 60)


def cmd_eval(args: argparse.Namespace) -> None:
    cfg = load_config()
    ollama = OllamaClient(cfg.ollama_url, cfg.embed_model)
    if not ollama.ping():
        print(f"error: cannot reach Ollama at {cfg.ollama_url}", file=sys.stderr)
        sys.exit(1)

    try:
        cases = load_evalset(args.evalset)
    except (OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.case:
        wanted = set(args.case)
        unknown = sorted(wanted - {c.name for c in cases})
        if unknown:
            print(f"error: no such case(s): {', '.join(unknown)}", file=sys.stderr)
            sys.exit(1)
        cases = [c for c in cases if c.name in wanted]

    store = Store(cfg.db_path, VECTOR_DIM)
    ks = tuple(k for k in DEFAULT_KS if k <= args.top_k) or (args.top_k,)

    for warning in preflight(store, cases, cfg.db_path / "index_state.json"):
        print(f"warning: {warning}", file=sys.stderr)

    results = run_cases(
        cases, ollama, store, args.top_k,
        worker_model=cfg.worker_model, multi_query=args.multi_query,
    )
    report = build_report(results, ks, cfg.embed_model, args.top_k, args.evalset)

    judgments = None
    if args.judge:
        judgments = judge_results(ollama, cfg.worker_model, results)
        report["metrics"].update(judged_metrics(judgments))

    baseline = None
    if args.compare:
        try:
            baseline = json.loads(args.compare.read_text())
        except (OSError, json.JSONDecodeError) as e:
            print(f"error: cannot read baseline {args.compare}: {e}", file=sys.stderr)
            sys.exit(1)

    if args.json:
        if baseline:
            report["baseline"] = {"timestamp": baseline.get("timestamp")}
        print(json.dumps(report, indent=2))
    else:
        print(format_report(results, ks, report, verbose=args.verbose))
        if judgments:
            jm = judged_metrics(judgments)
            if "judged_sufficient@1" in jm:
                line = f"judged_sufficient@1={jm['judged_sufficient@1']:.0%}"
                if "judge_errors" in jm:
                    line += f"  judge_errors={jm['judge_errors']}"
                print(line)
            print(format_judge_disagreements(results, judgments))
        if baseline:
            print(format_comparison(report, baseline))

    if args.save:
        args.save.write_text(json.dumps(report, indent=2))
        print(f"saved baseline to {args.save}", file=sys.stderr)

    if args.fail_under is not None:
        primary = report["metrics"].get(f"hit@{max(ks)}", 0.0)
        if primary < args.fail_under:
            print(
                f"error: hit@{max(ks)}={primary:.0%} is below --fail-under {args.fail_under:.0%}",
                file=sys.stderr,
            )
            sys.exit(1)


def cmd_mcp(_args: argparse.Namespace) -> None:
    from .server import run

    run()


def main() -> None:
    parser = argparse.ArgumentParser(prog="minsky")
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="index configured projects")
    p_index.add_argument("--project", help="index only this project")
    p_index.set_defaults(func=cmd_index)

    p_search = sub.add_parser("search", help="search the index from the command line")
    p_search.add_argument("query")
    p_search.add_argument("--project")
    p_search.add_argument("--kind", choices=["code", "doc", "note", "git"])
    p_search.add_argument("--top-k", type=int, default=8, dest="top_k")
    p_search.add_argument(
        "--rerank", action="store_true", help="rerank results with the worker model (slower)"
    )
    p_search.add_argument(
        "--multi-query",
        action="store_true",
        dest="multi_query",
        help="widen retrieval with worker-model query rewrites, fused via RRF (slower)",
    )
    p_search.set_defaults(func=cmd_search)

    p_projects = sub.add_parser("projects", help="list discovered projects")
    p_projects.set_defaults(func=cmd_projects)

    p_delegate = sub.add_parser(
        "delegate", help="send a boilerplate/bulk task to the local worker model"
    )
    p_delegate.add_argument("prompt", nargs="?", help="the task, as a plain-text prompt")
    p_delegate.add_argument(
        "--prompt-file", type=Path, help="read the prompt from a file instead"
    )
    p_delegate.add_argument(
        "--context", type=Path, help="a file whose contents are prepended as context"
    )
    p_delegate.add_argument(
        "--model", help="override the worker model (default: from minsky.toml)"
    )
    p_delegate.add_argument(
        "--out", type=Path, help="write the response to a file instead of stdout"
    )
    p_delegate.add_argument(
        "--quiet", action="store_true", help="don't stream the live prompt/response to stderr"
    )
    p_delegate.set_defaults(func=cmd_delegate)

    p_log = sub.add_parser("log", help="browse past delegate conversations")
    p_log.add_argument("--last", type=int, help="show only the last N conversations")
    p_log.add_argument(
        "--list",
        action="store_true",
        help="one-line-per-conversation summary instead of full transcripts",
    )
    p_log.set_defaults(func=cmd_log)

    p_eval = sub.add_parser("eval", help="measure retrieval quality against evalset.toml")
    p_eval.add_argument(
        "--evalset", type=Path, default=DEFAULT_EVALSET, help="path to the eval set TOML"
    )
    p_eval.add_argument(
        "--top-k", type=int, default=10, dest="top_k", help="retrieval window to score within"
    )
    p_eval.add_argument(
        "--case", action="append", help="run only this case (repeatable)"
    )
    p_eval.add_argument("--save", type=Path, help="write this run to a baseline JSON file")
    p_eval.add_argument("--compare", type=Path, help="diff this run against a saved baseline")
    p_eval.add_argument(
        "--multi-query",
        action="store_true",
        dest="multi_query",
        help="widen retrieval with worker-model query rewrites, fused via RRF (slower)",
    )
    p_eval.add_argument(
        "--judge",
        action="store_true",
        help="have the worker model judge whether the #1 hit's content actually answers the query",
    )
    p_eval.add_argument("--json", action="store_true", help="emit the raw report as JSON")
    p_eval.add_argument(
        "--verbose", action="store_true", help="for misses, show what came back instead"
    )
    p_eval.add_argument(
        "--fail-under",
        type=float,
        dest="fail_under",
        help="exit non-zero if the widest hit@k falls below this ratio (e.g. 0.8)",
    )
    p_eval.set_defaults(func=cmd_eval)

    p_mcp = sub.add_parser("mcp", help="run the MCP server over stdio")
    p_mcp.set_defaults(func=cmd_mcp)

    p_compact = sub.add_parser("compact", help="compact the DB and reclaim disk space")
    p_compact.set_defaults(func=cmd_compact)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
