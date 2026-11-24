from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pathspec

from .chunking import Chunk, chunk_file
from .config import Config
from .index_state import IndexState
from .ollama_client import OllamaClient
from .store import Store

GIT_RECORD_SEP = "\x1e"
GIT_FIELD_SEP = "\x1f"
EMBED_BATCH_SIZE = 64


def _load_gitignore(project_root: Path) -> pathspec.PathSpec | None:
    gi = project_root / ".gitignore"
    if not gi.exists():
        return None
    lines = gi.read_text(errors="ignore").splitlines()
    return pathspec.PathSpec.from_lines("gitwildmatch", lines)


def _classify(rel_path: Path, cfg: Config) -> str | None:
    if rel_path.name in cfg.index.note_files:
        return "note"
    if any(part in cfg.index.note_dirs for part in rel_path.parts):
        return "note"
    if rel_path.suffix in {".md", ".mdx", ".rst", ".txt"}:
        return "doc"
    if rel_path.suffix in cfg.index.extensions:
        return "code"
    return None


def walk_project_files(project_root: Path, cfg: Config) -> list[tuple[Path, str]]:
    spec = _load_gitignore(project_root)
    results: list[tuple[Path, str]] = []

    def _walk(dir_path: Path):
        for child in sorted(dir_path.iterdir()):
            rel = child.relative_to(project_root)
            if child.is_dir():
                if child.name in cfg.index.ignore_dirs:
                    continue
                if spec and spec.match_file(str(rel) + "/"):
                    continue
                _walk(child)
            elif child.is_file():
                if spec and spec.match_file(str(rel)):
                    continue
                kind = _classify(rel, cfg)
                if kind is None:
                    continue
                try:
                    if child.stat().st_size > cfg.index.max_file_bytes:
                        continue
                except OSError:
                    continue
                results.append((rel, kind))

    if project_root.is_dir():
        _walk(project_root)
    return results


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _embed_all(ollama: OllamaClient, texts: list[str]) -> list[list[float]]:
    """Embed in bounded sub-batches (HTTP payload size), independent of DB transaction batching."""
    vectors: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        vectors.extend(ollama.embed_many(texts[i : i + EMBED_BATCH_SIZE]))
    return vectors


def _collect_changed_files(
    project: str,
    project_root: Path,
    files: list[tuple[Path, str]],
    state: IndexState,
) -> tuple[list[tuple[str, str, str, float, list[Chunk]]], int]:
    """Returns (changed entries, count skipped as unchanged).

    Each entry: (path_str, kind, hash, mtime, chunks).
    """
    changed: list[tuple[str, str, str, float, list[Chunk]]] = []
    skipped = 0
    for rel_path, kind in files:
        full_path = project_root / rel_path
        try:
            text = full_path.read_text(encoding="utf-8", errors="strict")
        except (UnicodeDecodeError, OSError):
            continue

        path_str = str(rel_path)
        content_hash = _sha256(text)
        if state.get(project, path_str) == content_hash:
            skipped += 1
            continue

        try:
            mtime = full_path.stat().st_mtime
        except OSError:
            mtime = 0.0
        changed.append((path_str, kind, content_hash, mtime, chunk_file(text, rel_path.suffix)))
    return changed, skipped


def index_project(
    project: str,
    project_root: Path,
    cfg: Config,
    store: Store,
    state: IndexState,
    ollama: OllamaClient,
) -> dict[str, int]:
    files = walk_project_files(project_root, cfg)
    seen_paths = {str(rel) for rel, _ in files}

    changed, skipped = _collect_changed_files(project, project_root, files, state)

    known_paths = state.paths_for_project(project)
    removed_paths = [p for p in known_paths if not p.startswith("git/") and p not in seen_paths]

    paths_to_replace = [entry[0] for entry in changed] + removed_paths
    if paths_to_replace:
        store.delete_paths(project, paths_to_replace)

    all_texts: list[str] = []
    for _, _, _, _, chunks in changed:
        all_texts.extend(c.text for c in chunks)

    new_rows: list[dict] = []
    if all_texts:
        vectors = _embed_all(ollama, all_texts)
        vi = 0
        for path_str, kind, content_hash, mtime, chunks in changed:
            for i, c in enumerate(chunks):
                new_rows.append(
                    {
                        "id": f"{project}:{path_str}:{i}",
                        "project": project,
                        "path": path_str,
                        "kind": kind,
                        "content": c.text,
                        "mtime": mtime,
                        "start_line": c.start_line,
                        "vector": vectors[vi],
                    }
                )
                vi += 1

    for path_str, _, content_hash, _, _ in changed:
        state.set(project, path_str, content_hash)
    for path_str in removed_paths:
        state.remove(project, path_str)

    git_rows, commits_indexed = _collect_git_rows(project, project_root, cfg, state, ollama)
    new_rows.extend(git_rows)

    if new_rows:
        store.add(new_rows)

    return {
        "files_updated": len(changed),
        "files_removed": len(removed_paths),
        "commits_indexed": commits_indexed,
    }


def _collect_git_rows(
    project: str,
    project_root: Path,
    cfg: Config,
    state: IndexState,
    ollama: OllamaClient,
) -> tuple[list[dict], int]:
    if not cfg.git.enabled or not (project_root / ".git").is_dir():
        return [], 0

    fmt = f"%H{GIT_FIELD_SEP}%ad{GIT_FIELD_SEP}%s{GIT_FIELD_SEP}%b{GIT_RECORD_SEP}"
    try:
        proc = subprocess.run(
            [
                "git", "-C", str(project_root), "log",
                f"-n{cfg.git.max_commits_per_project}",
                "--date=short", f"--pretty=format:{fmt}",
            ],
            capture_output=True, text=True, timeout=30, check=True,
        )
    except (subprocess.CalledProcessError, OSError):
        return [], 0

    records = [r for r in proc.stdout.split(GIT_RECORD_SEP) if r.strip()]
    new_meta: list[tuple[str, str, str]] = []
    for record in records:
        parts = record.strip("\n").split(GIT_FIELD_SEP)
        if len(parts) < 3:
            continue
        commit_hash, date, subject = parts[0], parts[1], parts[2]
        body = parts[3] if len(parts) > 3 else ""
        path_str = f"git/{commit_hash}"
        if state.get(project, path_str) == commit_hash:
            continue
        content = f"[{date}] {subject}\n\n{body}".strip()
        new_meta.append((commit_hash, path_str, content))

    if not new_meta:
        return [], 0

    vectors = _embed_all(ollama, [content for _, _, content in new_meta])
    rows = []
    for (commit_hash, path_str, content), vec in zip(new_meta, vectors):
        rows.append(
            {
                "id": f"{project}:{path_str}",
                "project": project,
                "path": path_str,
                "kind": "git",
                "content": content,
                "mtime": 0.0,
                "start_line": 0,
                "vector": vec,
            }
        )
        state.set(project, path_str, commit_hash)

    return rows, len(rows)
