# Architecture Constraints

- Embeddings are generated locally via Ollama (`nomic-embed-text` model) - nothing is ever sent to a cloud API for embedding.
- The vector store is LanceDB, embedded (no server process), stored at `~/.minsky/db` by default.
- Chunking strategy is file-type-dependent: real source code extensions (.py, .js, .ts, .go, .rs, .java, .rb, .c, .cpp, .swift, .kt, .sh, .sql, etc.) use tree-sitter AST-aware chunking that splits on function/class/definition boundaries; `.toml` files get section-aware chunking (one chunk per `[table]`); everything else falls back to a blank-line heuristic.
- The MCP server (`minsky/server.py`) exposes exactly three tools: `search(query, project?, kind?, top_k?, rerank?, multi_query?)`, `list_projects()`, `list_configured_projects()`. A setup agent must not add, remove, or rename these tools.
- Two opt-in retrieval quality passes exist, both off by default since each adds a worker-model round trip: `rerank` (worker model reorders a wider candidate pool by relevance) and `multi_query` (worker model generates alternative phrasings of the query, fused via Reciprocal Rank Fusion). A setup agent should leave both off by default unless the user explicitly asks to enable them.
- A separate `delegate` CLI command sends boilerplate/bulk tasks to a local worker model directly, with no involvement from the index/vector store at all - this is a completely independent code path from search.
- Indexing is on-demand (`minsky index`), not a live file watcher - there is no background process.