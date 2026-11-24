from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "minsky.toml"


def _expand(path: str) -> Path:
    return Path(path).expanduser().resolve()


@dataclass
class GitConfig:
    enabled: bool = True
    max_commits_per_project: int = 500


@dataclass
class IndexConfig:
    roots: list[Path] = field(default_factory=list)
    extra_projects: list[Path] = field(default_factory=list)
    extensions: set[str] = field(default_factory=set)
    ignore_dirs: set[str] = field(default_factory=set)
    note_files: set[str] = field(default_factory=set)
    note_dirs: set[str] = field(default_factory=set)
    max_file_bytes: int = 500_000


@dataclass
class Config:
    db_path: Path
    ollama_url: str
    embed_model: str
    worker_model: str
    index: IndexConfig
    git: GitConfig


def load_config(path: Path | None = None) -> Config:
    path = path or DEFAULT_CONFIG_PATH
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    general = raw.get("general", {})
    idx = raw.get("index", {})
    git = raw.get("git", {})

    return Config(
        db_path=_expand(general.get("db_path", "~/.minsky/db")),
        ollama_url=general.get("ollama_url", "http://localhost:11434"),
        embed_model=general.get("embed_model", "nomic-embed-text"),
        worker_model=general.get("worker_model", "qwen2.5-coder:7b"),
        index=IndexConfig(
            roots=[_expand(p) for p in idx.get("roots", [])],
            extra_projects=[_expand(p) for p in idx.get("extra_projects", [])],
            extensions=set(idx.get("extensions", [])),
            ignore_dirs=set(idx.get("ignore_dirs", [])),
            note_files=set(idx.get("note_files", [])),
            note_dirs=set(idx.get("note_dirs", [])),
            max_file_bytes=idx.get("max_file_bytes", 500_000),
        ),
        git=GitConfig(
            enabled=git.get("enabled", True),
            max_commits_per_project=git.get("max_commits_per_project", 500),
        ),
    )


def discover_projects(cfg: Config) -> dict[str, Path]:
    """Map project name -> project root directory."""
    projects: dict[str, Path] = {}
    for root in cfg.index.roots:
        if not root.is_dir():
            continue
        for child in sorted(root.iterdir()):
            if child.is_dir() and not child.name.startswith("."):
                projects[child.name] = child
    for extra in cfg.index.extra_projects:
        if extra.is_dir():
            projects[extra.name] = extra
    return projects

