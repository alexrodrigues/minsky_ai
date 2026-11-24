# Config Schema

This document is the authoritative schema for `minsky.toml`.
An AI coding agent configuring this project for a new user must use only the keys listed here and must not invent any new keys or change the expected types.

## Config Sections and Keys

| Section | Key | Type | Default | Description |
|---------|-----|------|---------|-------------|
| `[general]` | `db_path` | `Path` | `~/.minsky/db` | Path to the embedded LanceDB vector store. Supports `~` expansion. |
| | `ollama_url` | `string` | `http://localhost:11434` | Base URL for the Ollama service. |
| | `embed_model` | `string` | `nomic-embed-text` | Model name used for embedding generation. |
| | `worker_model` | `string` | `qwen2.5-coder:7b` (code default) | Model name used for delegated/worker inference - see note below, the shipped `minsky.toml` overrides this. |
| `[index]` | `roots` | `list[Path]` | `[]` | List of directories; each immediate subdirectory is treated as a project. |
| | `extra_projects` | `list[Path]` | `[]` | Individual project directories outside the `roots` directories. |
| | `extensions` | `set[string]` | `{}` | File extensions to index (e.g., `".py", ".md"`). |
| | `ignore_dirs` | `set[string]` | `{}` | Directory names to exclude from indexing. |
| | `note_files` | `set[string]` | `{}` | File names considered notes. |
| | `note_dirs` | `set[string]` | `{}` | Directory names considered notes. |
| | `max_file_bytes` | `int` | `500000` | Maximum file size in bytes to index (default 500 KB). |
| `[git]` | `enabled` | `bool` | `true` | Whether to index Git history. |
| | `max_commits_per_project` | `int` | `500` | Maximum number of commits to include per project. |

> **Note**: The configuration file is named `minsky.toml`.
> **Note**: Path values support `~` expansion to the user's home directory.
> **Note**: `worker_model`'s code-level default (`qwen2.5-coder:7b`) is not the
> actual model a fresh clone uses - the shipped `minsky.toml` sets it to
> `gpt-oss:20b`. Always check `minsky.toml` for the real value, don't trust the
> code default.

## Important Configuration Guidance

- The `index.roots` list is the primary item a setup agent should edit.
  It defines directories whose immediate subdirectories are each treated as one project.
  Use `index.extra_projects` for projects that reside outside these root directories.

- Do **not** add keys that are not listed in this schema.
- Do **not** remove the `[git]` section; it is required for proper functionality.
- Do **not** set `index.max_file_bytes` to an absurdly large value without first consulting the user.

---