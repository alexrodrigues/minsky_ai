# Non-Goals

- No cloud embedding or LLM APIs - everything (embeddings, worker-model calls) stays local via Ollama. Why: privacy is a core design goal, not an afterthought.
- No live file-watcher / continuous background indexing - indexing is on-demand only (`minsky index`). Why: keeps the system simple and predictable; staleness is handled by the user re-running index, not hidden behind automatic magic.
- Git indexing covers commit messages only, never diffs. Why: keeps indexing fast; diffs are large and rarely what a semantic search query is actually looking for.
- No multi-tenant server / no network-exposed API - the MCP server runs locally over stdio for a single user's own Claude Code/Desktop session. Why: this is a personal local tool, not a hosted service.
- No telemetry or usage tracking of any kind.
- `.gitignore` is only read from each indexed project's own root, not nested subdirectory `.gitignore` files. Why: a documented, known limitation, not a goal to silently fix as part of a setup task.
- No automatic model selection/benchmarking - the user (or setup agent, once) picks a worker model explicitly; minsky doesn't try multiple models and compare them automatically.

A setup agent configuring this project should stay within these boundaries - if a user explicitly asks for something on this list, that's a feature request for the project itself, not something to improvise during setup.