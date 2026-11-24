# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Minsky is a local RAG (retrieval-augmented generation) layer over the user's `~/Developer` projects (code, docs, notes, git history), exposed as an MCP server so Claude can search it directly. Pure Python, single package, no monorepo structure.

Pipeline: `minsky/indexer.py` (AST-aware chunking via tree-sitter for source files, section-aware chunking for TOML, blank-line heuristic for everything else) -> `minsky/store.py` (embedded LanceDB at `~/.minsky/db`, embeddings via local Ollama `nomic-embed-text`) -> `minsky/server.py` (MCP server exposing `search`, `list_projects`, `list_configured_projects`). `search` takes two opt-in params, both off by default since each adds worker-model round trips: `rerank` (`minsky/rerank.py`) fetches a wider candidate pool and has the worker model reorder it by relevance before truncating; `multi_query` (`minsky/multiquery.py`) has the worker model generate alternative phrasings of the query and fuses the vector-search results across all of them via Reciprocal Rank Fusion, so a result only one wording's embedding would have surfaced can still rank highly. The two compose: fused candidates can be reranked before truncating to `top_k`.

`minsky delegate` offloads boilerplate/bulk work to a local Ollama worker model (`minsky/delegate.py`), logging every call to `~/.minsky/delegate_logs/`.

## Commands

There is no test suite or CI, but `ruff` is configured in `pyproject.toml` and
`minsky eval` measures retrieval quality (see **Retrieval evaluation** below). Run via the venv entry point:

```
.venv/bin/minsky projects
.venv/bin/minsky index [--project NAME]
.venv/bin/minsky search "<query>" [--project P] [--kind code|doc|note|git] [--top-k N] [--rerank] [--multi-query]
.venv/bin/minsky delegate "<prompt>" [--prompt-file FILE] [--context FILE] [--model M] [--out FILE] [--quiet]
.venv/bin/minsky log [--list] [--last N]
.venv/bin/minsky mcp        # runs the MCP server over stdio
.venv/bin/minsky compact    # compacts LanceDB, reclaims disk space
.venv/bin/minsky eval [--evalset F] [--top-k N] [--case NAME] [--save F] [--compare F] [--multi-query] [--judge] [--json] [--verbose] [--fail-under R]
```

Requires Ollama running locally (`brew services start ollama`) with `nomic-embed-text` pulled for indexing/search, and the worker model pulled for `delegate`. Commands fail fast with a clear stderr error if Ollama is unreachable, but not if the specific model isn't pulled.

## Config

Runtime config is `minsky.toml` at the repo root (not `.env` - there is none). Key settings: `general.worker_model`, `general.db_path`, `index.roots`, `git.max_commits_per_project`.

**Gotcha**: the code-level default for `worker_model` in `minsky/config.py` is `qwen2.5-coder:7b`, but `minsky.toml` overrides it to `gpt-oss:20b` (current, swapped from `qwen2.5-coder:14b` after a head-to-head test on real-shaped tasks found qwen introduced a silent data-corruption bug that gpt-oss avoided - `qwen2.5-coder:32b` was tried even earlier and dropped for showing no accuracy gain at 2x memory/latency). Don't trust the code default as the actual current model; always check `minsky.toml`.

## Retrieval evaluation

`minsky eval` scores retrieval against `evalset.toml` at the repo root - each
`[[case]]` is a query plus the path(s) a correct answer must surface (`expect`
entries are case-insensitive substring matches on the result path). It reports
**hit@1/3/5/10**, MRR, and **nDCG@1/3/5/10**. Named hit@k, not recall@k, because
a case may list several acceptable paths and finding any one counts. nDCG is
graded rather than binary: it rewards a relevant hit ranking earlier, and
rewards more of a case's acceptable paths showing up in the same top-k window,
a detail hit@k's first-match-only rank collapses away - computed against the
ideal ordering of each case's own retrieved relevances, not a corpus-wide
ground truth.

Two more opt-in passes, each off by default since they add a worker-model call
per case:

- `--multi-query`: runs retrieval through `minsky/multiquery.py`'s
  `multi_query_search` instead of a single embed call, so the measured numbers
  reflect the same RRF-fused retrieval `search(multi_query=True)` does live.
- `--judge`: asks the worker model whether the #1 hit's actual content would
  answer the query, independent of whether its path matched `expect`. This is
  a check on hit@k itself, not a replacement - it catches both a path match
  that isn't really the answer, and a path miss where the content was fine but
  lived somewhere the evalset never named. Disagreements between path-matching
  and the judge print separately from the main report. A judge call that
  errors or returns something unparseable counts as neither a pass nor a fail
  (reported as `judge_errors`), never silently as "insufficient".

The point is before/after, not the absolute number:

```
.venv/bin/minsky eval --save baseline.json      # record where you are
# ... change chunk size, embed model, AST boundaries ...
.venv/bin/minsky index && .venv/bin/minsky eval --compare baseline.json
```

The comparison prints per-case movement and labels `REGRESSED` for anything that
fell out of the window - a chunking change that lifts the average while breaking
three specific lookups is otherwise invisible.

`eval` preflights the two silent failure modes before scoring, so a miss isn't
misread as bad retrieval: it warns when a case targets a project absent from the
index, and when `index_state.json` is more than 24h old.

Use `--fail-under 0.8` to make it exit non-zero (for a hook or cron). Add cases
from queries you actually ran - the numbers only mean something if the queries
are real.

**Fixed finding**: `worker-model-config` ("which local model is used for
delegated work") used to miss, because the blank-line heuristic packed
`minsky.toml`'s whole first half into one 1438-char chunk, so `worker_model`
competed with every other setting in that chunk and lost to prose in
`README.md`. `.toml` files now get section-aware
chunking (`minsky/chunking.py`'s `chunk_toml`): one chunk per `[table]`/
`[[table]]`, each prefixed with a `# [section.path]` header line, falling
back to the blank-line heuristic only if the file has no table structure at
all. Re-running the indexer after this kind of chunking change requires
clearing the affected files' `IndexState` entries first - the content-hash
skip in `indexer.py` doesn't know the chunking *logic* changed, only whether
the file's bytes did.

## Delegate review discipline

When reviewing `minsky delegate` output (or acting on it), always actually run or compile it - never accept it on a skim. This has caught real bugs before (wrong function signatures, version-incompatible syntax, broken imports). Only delegate boilerplate/mechanical work in the first place; judgment-heavy or multi-file-reasoning tasks stay with Claude directly.

## Other gotchas

- AST-aware chunking only covers real source extensions (`.py .js .jsx .ts .tsx .go .rs .java .rb .c .h .cpp .hpp .cs .swift .kt .sh .sql`); `.toml` gets its own section-aware chunking (one chunk per `[table]`); everything else (`.md .json`, etc.) falls back to a blank-line heuristic.
- Git indexing covers commit messages only, no diffs; `.gitignore` is read only from each project's root, not nested.
- No file watcher - index freshness depends entirely on manually/cron-triggered `minsky index`.