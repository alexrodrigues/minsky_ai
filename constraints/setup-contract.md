# Setup Contract

1. The agent may **ONLY** write to `minsky.toml` as configuration output. It must never edit any file under `minsky/` (the package source) during setup.

2. Before pulling any Ollama model (`ollama pull ...`), the agent must tell the user the model name and its approximate download size, and get explicit confirmation-models can be many gigabytes.

3. Before running `minsky index` against a directory tree, if that tree looks very large (the agent should say what "large" means in practice-for example, more than 100,000 items or the user's entire home directory), confirm with the user first rather than indexing silently.

4. The agent must never run `git commit`, `git push`, or create/modify any GitHub repository as part of this setup flow-setup is local-only.

5. The agent must never read, display, or embed contents of files that look like secrets (`.env`, files containing "SECRET", "API_KEY", "PASSWORD", private key files) even if they fall inside a directory being indexed. Note to the user that minsky's own `.gitignore`-respecting indexer already skips common cases, but the setup agent should still flag this explicitly rather than assume.

6. The agent should ask the user which directories to index rather than silently defaulting to indexing their entire home directory.

7. If any step fails (Ollama not installed, a model pull fails, `pip install` fails), the agent should stop and report the failure clearly rather than silently continuing with a broken setup.

These rules exist because this setup flow runs with real filesystem and process access on the user's machine - treat them as constraints on the agent doing the setup, not suggestions.