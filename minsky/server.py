from __future__ import annotations

from mcp.server import MCPServer

from .config import discover_projects, load_config
from .multiquery import multi_query_search
from .ollama_client import OllamaClient
from .rerank import RETRIEVAL_K, rerank_results
from .store import Store

VECTOR_DIM = 768

mcp = MCPServer("minsky")

_cfg = load_config()
_ollama = OllamaClient(_cfg.ollama_url, _cfg.embed_model)
_store = Store(_cfg.db_path, VECTOR_DIM)


@mcp.tool()
def search(
    query: str,
    project: str | None = None,
    kind: str | None = None,
    top_k: int = 8,
    rerank: bool = False,
    multi_query: bool = False,
) -> list[dict]:
    """Search Minsky's index of the user's local projects (code, docs, notes, git history).

    Args:
        query: natural-language search query.
        project: optional project name to restrict the search to.
        kind: optional filter, one of "code", "doc", "note", "git".
        top_k: max number of results to return.
        rerank: if True, fetch a wider candidate pool and have the worker
            model reorder it by relevance before truncating to top_k. Slower
            (an extra local LLM call) but often more accurate for queries
            where plain vector similarity picks a close-but-wrong match.
        multi_query: if True, have the worker model generate alternative
            phrasings of the query and fuse the vector search results across
            all of them (RRF). Slower (worker-model call plus extra embed/
            search calls per variant) but can surface results the original
            wording's embedding alone would have missed. Composes with
            rerank: when both are set, the fused candidates get reranked
            before truncating to top_k.
    """
    candidate_k = max(top_k, RETRIEVAL_K) if rerank else top_k
    if multi_query:
        results = multi_query_search(
            _ollama, _cfg.worker_model, _store, query,
            top_k=candidate_k, project=project, kind=kind,
        )
    else:
        vector = _ollama.embed(query)
        results = _store.search(vector, top_k=candidate_k, project=project, kind=kind)
    if rerank:
        results = rerank_results(_ollama, _cfg.worker_model, query, results, top_k=top_k)
    return results


@mcp.tool()
def list_projects() -> list[str]:
    """List all projects currently indexed by Minsky."""
    return _store.list_projects()


@mcp.tool()
def list_configured_projects() -> dict[str, str]:
    """List projects Minsky is configured to index (may include not-yet-indexed ones)."""
    return {name: str(path) for name, path in discover_projects(_cfg).items()}


def run() -> None:
    mcp.run()


if __name__ == "__main__":
    run()
