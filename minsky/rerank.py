from __future__ import annotations

import json
import re

from .ollama_client import OllamaClient

# Candidate pool size to fetch from the vector store before reranking.
RETRIEVAL_K = 20
# Max characters of each result's content sent to the LLM per candidate.
SNIPPET_CHARS = 400


def rerank_results(
    ollama: OllamaClient,
    model: str,
    query: str,
    results: list[dict],
    top_k: int,
) -> list[dict]:
    """Retrieve-then-rerank: `results` is a wider candidate pool from vector
    search, already ranked by similarity. This asks the worker model to
    reorder them by actual relevance to `query`, then truncates to `top_k`.

    Fails safe to the original vector-similarity order (truncated to
    top_k) on any call, parse, or validation error - reranking is an
    optional quality pass layered on an already-working search, never the
    reason a search call fails.
    """
    if len(results) <= 1:
        return results[:top_k]

    entries = "\n\n".join(
        f"[{i + 1}] {r.get('path', '')}\n{r.get('content', '')[:SNIPPET_CHARS]}"
        for i, r in enumerate(results)
    )
    prompt = (
        f"Query: {query}\n\n"
        f"Candidate snippets:\n{entries}\n\n"
        f"Return a JSON array of the {top_k} most relevant candidate numbers above, "
        "ordered from most to least relevant to the query. Return fewer than "
        f"{top_k} if fewer are genuinely relevant. Output ONLY the JSON array "
        "(e.g. [3, 1, 2]), nothing else - no explanation, no markdown fences, no code block."
    )

    try:
        response = ollama.chat(
            model,
            prompt,
            system="You are a precise relevance-ranking assistant. "
            "You output only valid JSON arrays, nothing else.",
        )

        resp_text = response.strip()
        try:
            parsed = json.loads(resp_text)
        except Exception:
            match = re.search(r"\[[\d,\s]+\]", resp_text)
            if not match:
                raise
            parsed = json.loads(match.group())

        if not isinstance(parsed, list) or not all(isinstance(x, int) for x in parsed):
            raise ValueError("reranker response is not a list of integers")

        # Lenient by design: the model may legitimately return fewer than
        # top_k indices (not everything is relevant), so this doesn't
        # require a full permutation of all candidates - just that every
        # index it does return is a valid, non-duplicate candidate number.
        # Anything duplicated or out of range is dropped rather than
        # treated as a hard failure.
        seen: set[int] = set()
        ordered: list[int] = []
        for i in parsed:
            if 1 <= i <= len(results) and i not in seen:
                seen.add(i)
                ordered.append(i)
        if not ordered:
            raise ValueError("reranker response contained no valid candidate indices")

        reordered = [results[i - 1] for i in ordered]
        remaining = [r for j, r in enumerate(results) if (j + 1) not in seen]
        return (reordered + remaining)[:top_k]
    except Exception:
        return results[:top_k]
