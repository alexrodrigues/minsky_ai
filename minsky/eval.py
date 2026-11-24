from __future__ import annotations

import math
import time
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .multiquery import multi_query_search
from .ollama_client import OllamaClient
from .store import Store

DEFAULT_EVALSET = Path(__file__).resolve().parent.parent / "evalset.toml"
DEFAULT_KS = (1, 3, 5, 10)
STALE_AFTER_HOURS = 24.0
# Max characters of the top hit's content sent to the judge model per case.
JUDGE_SNIPPET_CHARS = 800


@dataclass
class Case:
    """One retrieval expectation: a query, and the paths that should come back."""

    name: str
    query: str
    expect: list[str]
    project: str | None = None
    kind: str | None = None


@dataclass
class CaseResult:
    case: Case
    rank: int | None  # 1-based rank of the first expected hit; None if it never appeared
    hit_path: str | None
    top_paths: list[str]
    # Content of the #1 result, kept only for judge_case; not serialized into reports.
    top_content: str | None = None


def load_evalset(path: Path) -> list[Case]:
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    entries = raw.get("case", [])
    if not entries:
        raise ValueError(f"{path} defines no [[case]] entries")

    cases: list[Case] = []
    seen: set[str] = set()
    for i, entry in enumerate(entries, 1):
        for required in ("name", "query", "expect"):
            if not entry.get(required):
                raise ValueError(f"case #{i} in {path} is missing '{required}'")
        name = entry["name"]
        if name in seen:
            raise ValueError(f"duplicate case name '{name}' in {path}")
        seen.add(name)
        expect = entry["expect"]
        if isinstance(expect, str):
            expect = [expect]
        cases.append(
            Case(
                name=name,
                query=entry["query"],
                expect=list(expect),
                project=entry.get("project"),
                kind=entry.get("kind"),
            )
        )
    return cases


def _match(path: str, expect: list[str]) -> bool:
    low = path.lower()
    return any(e.lower() in low for e in expect)


def preflight(store: Store, cases: list[Case], state_path: Path) -> list[str]:
    """Catch the two silent failure modes before they look like bad recall.

    A project that was never indexed returns zero hits rather than an error, and
    a stale index returns confident hits against content that has since moved.
    """
    warnings: list[str] = []

    indexed = set(store.list_projects())
    if not indexed:
        warnings.append("index is empty - run `minsky index` before trusting these numbers")
    else:
        wanted = {c.project for c in cases if c.project}
        missing = sorted(wanted - indexed)
        if missing:
            warnings.append(
                "cases target projects absent from the index (they can only miss): "
                + ", ".join(missing)
            )

    if state_path.exists():
        age_hours = (time.time() - state_path.stat().st_mtime) / 3600
        if age_hours >= STALE_AFTER_HOURS:
            warnings.append(
                f"index last updated {age_hours / 24:.1f} days ago - "
                "misses may be staleness, not retrieval"
            )

    return warnings


def run_cases(
    cases: list[Case],
    ollama: OllamaClient,
    store: Store,
    top_k: int,
    worker_model: str | None = None,
    multi_query: bool = False,
) -> list[CaseResult]:
    if multi_query and worker_model is None:
        raise ValueError("multi_query requires worker_model")

    results: list[CaseResult] = []
    for case in cases:
        if multi_query:
            hits = multi_query_search(
                ollama, worker_model, store, case.query,
                top_k=top_k, project=case.project, kind=case.kind,
            )
        else:
            vector = ollama.embed(case.query)
            hits = store.search(vector, top_k=top_k, project=case.project, kind=case.kind)
        paths = [h["path"] for h in hits]

        rank: int | None = None
        hit_path: str | None = None
        for position, path in enumerate(paths, 1):
            if _match(path, case.expect):
                rank, hit_path = position, path
                break

        top_content = hits[0]["content"] if hits else None
        results.append(
            CaseResult(
                case=case, rank=rank, hit_path=hit_path, top_paths=paths, top_content=top_content
            )
        )
    return results


def _relevance_vector(top_paths: list[str], expect: list[str], k: int) -> list[int]:
    return [1 if _match(p, expect) else 0 for p in top_paths[:k]]


def _dcg(relevances: list[int]) -> float:
    # i starts at 0, so the discount is log2(2)=1 for the first position.
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))


def _ndcg(relevances: list[int]) -> float:
    """nDCG against the ideal ordering of this case's own retrieved relevances.

    Unlike hit@k (which only asks "did any acceptable path appear"), this
    rewards relevant hits ranking earlier, and rewards more of a case's
    multiple acceptable paths showing up in the same top-k window rather
    than collapsing that detail down to a single first-match rank.
    """
    if not any(relevances):
        return 0.0
    idcg = _dcg(sorted(relevances, reverse=True))
    return _dcg(relevances) / idcg if idcg > 0 else 0.0


def metrics(results: list[CaseResult], ks: tuple[int, ...]) -> dict[str, float]:
    """hit@k = share of queries whose expected path appeared in the top k.

    Named hit@k rather than recall@k on purpose: a case can list several
    acceptable paths, and finding any one of them counts as success.
    """
    n = len(results)
    if n == 0:
        return {}
    out: dict[str, float] = {}
    for k in ks:
        out[f"hit@{k}"] = sum(1 for r in results if r.rank is not None and r.rank <= k) / n
    out["mrr"] = sum(1 / r.rank for r in results if r.rank is not None) / n
    for k in ks:
        out[f"ndcg@{k}"] = sum(
            _ndcg(_relevance_vector(r.top_paths, r.case.expect, k)) for r in results
        ) / n
    return out


def build_report(
    results: list[CaseResult],
    ks: tuple[int, ...],
    embed_model: str,
    top_k: int,
    evalset: Path,
) -> dict[str, Any]:
    return {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "embed_model": embed_model,
        "evalset": str(evalset),
        "top_k": top_k,
        "cases": len(results),
        "metrics": metrics(results, ks),
        "ranks": {r.case.name: r.rank for r in results},
    }


def judge_case(
    ollama: OllamaClient, model: str, result: CaseResult, snippet_chars: int = JUDGE_SNIPPET_CHARS
) -> bool | None:
    """Ask the worker model whether the #1 retrieved chunk actually looks
    sufficient to answer the query, independent of whether its path matched
    the evalset's expected path(s). This is a check on hit@k itself, not a
    replacement for it: a case can match the expected path on a chunk that
    doesn't really contain the answer (a false positive for hit@k), or miss
    every listed path while a genuinely good answer lives somewhere the
    evalset never named (a false negative).

    Returns None (not "no") on any call or parse error, so a judge failure
    is never silently counted as "insufficient".
    """
    if not result.top_content:
        return False
    prompt = (
        f"Query: {result.case.query}\n\n"
        f"Retrieved content (top search result):\n{result.top_content[:snippet_chars]}\n\n"
        "Would this content actually help answer the query above? Answer with "
        "ONLY the single word YES or NO, nothing else."
    )
    try:
        response = ollama.chat(
            model,
            prompt,
            system="You are a strict relevance judge. You answer only YES or NO.",
        )
        answer = response.strip().upper()
        if answer.startswith("YES"):
            return True
        if answer.startswith("NO"):
            return False
        raise ValueError(f"unparseable judge response: {response!r}")
    except Exception:
        return None


def judge_results(ollama: OllamaClient, model: str, results: list[CaseResult]) -> list[bool | None]:
    return [judge_case(ollama, model, r) for r in results]


def judged_metrics(judgments: list[bool | None]) -> dict[str, float]:
    decided = [j for j in judgments if j is not None]
    if not decided:
        return {}
    out = {"judged_sufficient@1": sum(1 for j in decided if j) / len(decided)}
    errors = sum(1 for j in judgments if j is None)
    if errors:
        out["judge_errors"] = errors
    return out


def format_judge_disagreements(results: list[CaseResult], judgments: list[bool | None]) -> str:
    """Surface cases where hit@1 and the judge disagree on the #1 result
    specifically - the judge only ever looks at that one chunk, so this is
    the only apples-to-apples comparison. A rank-2+ or total miss isn't a
    "disagreement" here, it's just restating an already-known hit@1 miss.
    """
    lines: list[str] = []
    for r, j in zip(results, judgments):
        if j is None:
            continue
        hit_at_1 = r.rank == 1
        if hit_at_1 and not j:
            lines.append(f"  {r.case.name}: path matched at #1 but judge says NOT sufficient")
        elif not hit_at_1 and j:
            where = f"matched later at #{r.rank}" if r.rank else "missed entirely"
            lines.append(f"  {r.case.name}: path {where}, but judge says #1 IS sufficient")
    if not lines:
        return ""
    return "\ndisagreements (path-match vs judge):\n" + "\n".join(lines)


def format_report(
    results: list[CaseResult],
    ks: tuple[int, ...],
    report: dict[str, Any],
    verbose: bool = False,
) -> str:
    lines: list[str] = []
    width = max((len(r.case.name) for r in results), default=4)

    for r in results:
        if r.rank is None:
            lines.append(f"MISS  {r.case.name:<{width}}   --   {r.case.query}")
            if verbose:
                for path in r.top_paths[:5]:
                    lines.append(f"            got: {path}")
        else:
            lines.append(f"ok    {r.case.name:<{width}}  #{r.rank:<3} {r.case.query}")

    lines.append("")
    m = report["metrics"]
    summary = "  ".join(f"hit@{k}={m[f'hit@{k}']:.0%}" for k in ks)
    lines.append(f"{report['cases']} cases   {summary}   mrr={m['mrr']:.3f}")
    ndcg_summary = "  ".join(f"ndcg@{k}={m[f'ndcg@{k}']:.3f}" for k in ks if f"ndcg@{k}" in m)
    if ndcg_summary:
        lines.append(ndcg_summary)
    lines.append(f"embed_model={report['embed_model']}  top_k={report['top_k']}")

    misses = [r.case.name for r in results if r.rank is None]
    if misses:
        lines.append(f"missed: {', '.join(misses)}")
    return "\n".join(lines)


def format_comparison(report: dict[str, Any], baseline: dict[str, Any]) -> str:
    """Diff a run against a saved baseline, so a chunking or model change has a verdict."""
    lines = ["", f"vs baseline {baseline['timestamp']} (embed_model={baseline['embed_model']})"]

    for key, value in report["metrics"].items():
        before = baseline.get("metrics", {}).get(key)
        if before is None:
            lines.append(f"  {key}: {value:.3f} (new)")
            continue
        delta = value - before
        arrow = "+" if delta > 0 else ("-" if delta < 0 else "=")
        lines.append(f"  {key}: {before:.3f} -> {value:.3f}  {arrow}{abs(delta):.3f}")

    old_ranks = baseline.get("ranks", {})
    new_ranks = report["ranks"]

    fixed, broke, moved = [], [], []
    for name, new in new_ranks.items():
        if name not in old_ranks:
            continue
        old = old_ranks[name]
        if old is None and new is not None:
            fixed.append(f"{name} (miss -> #{new})")
        elif old is not None and new is None:
            broke.append(f"{name} (#{old} -> miss)")
        elif old != new and old is not None and new is not None:
            moved.append(f"{name} (#{old} -> #{new})")

    for label, items in (("fixed", fixed), ("REGRESSED", broke), ("moved", moved)):
        if items:
            lines.append(f"  {label}: {', '.join(items)}")

    added = sorted(set(new_ranks) - set(old_ranks))
    dropped = sorted(set(old_ranks) - set(new_ranks))
    if added:
        lines.append(f"  new cases: {', '.join(added)}")
    if dropped:
        lines.append(f"  absent from this run: {', '.join(dropped)}")

    return "\n".join(lines)
