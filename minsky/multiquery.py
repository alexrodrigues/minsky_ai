from __future__ import annotations

import json
import re

from .ollama_client import OllamaClient
from .store import Store

# Number of alternative phrasings to generate in addition to the original query.
NUM_VARIANTS = 3
# Reciprocal rank fusion constant, matching the k used in the earlier
# (reverted) vector+FTS hybrid-search experiment.
RRF_K = 60
# Candidate pool fetched per query variant before fusion.
CANDIDATE_K = 20


def generate_query_variants(
    ollama: OllamaClient, model: str, query: str, n: int = NUM_VARIANTS
) -> list[str]:
    """Ask the worker model for alternative phrasings of `query` that might
    surface different but still-relevant results than the original wording.

    Fails safe to an empty list (only the original query gets searched) on
    any call, parse, or validation error - query rewriting is an optional
    widening pass, never the reason a search fails.
    """
    prompt = (
        f"Query: {query}\n\n"
        f"Generate {n} alternative phrasings of this search query, using different "
        "words or angles that might surface different but still relevant results. "
        "Output ONLY a JSON array of strings, nothing else - no explanation, no "
        "markdown fences, no code block."
    )
    try:
        response = ollama.chat(
            model,
            prompt,
            system="You are a precise search-query rewriting assistant. "
            "You output only valid JSON arrays of strings, nothing else.",
        )
        resp_text = response.strip()
        try:
            parsed = json.loads(resp_text)
        except Exception:
            match = re.search(r"\[.*\]", resp_text, re.DOTALL)
            if not match:
                raise
            parsed = json.loads(match.group())

        if not isinstance(parsed, list) or not all(isinstance(x, str) for x in parsed):
            raise ValueError("query-rewrite response is not a list of strings")
        return [q.strip() for q in parsed if q.strip()][:n]
    except Exception:
        return []


def multi_query_search(
    ollama: OllamaClient,
    worker_model: str,
    store: Store,
    query: str,
    top_k: int,
    project: str | None = None,
    kind: str | None = None,
) -> list[dict]:
    """Retrieve-then-fuse: generate alternative phrasings of `query`, run
    vector search for each (plus the original), and fuse the ranked lists via
    Reciprocal Rank Fusion. A result that only one phrasing's wording happens
    to surface can still rank highly, instead of depending on the single
    embedding of however the query was originally worded.
    """
    variants = generate_query_variants(ollama, worker_model, query)
    queries = [query] + variants

    scores: dict[tuple, float] = {}
    rows: dict[tuple, dict] = {}
    for q in queries:
        vector = ollama.embed(q)
        hits = store.search(vector, top_k=CANDIDATE_K, project=project, kind=kind)
        for rank, hit in enumerate(hits, 1):
            key = (hit["path"], hit.get("start_line"))
            scores[key] = scores.get(key, 0.0) + 1 / (RRF_K + rank)
            rows.setdefault(key, hit)

    ordered = sorted(scores, key=lambda key: scores[key], reverse=True)
    return [rows[key] for key in ordered[:top_k]]
