#!/usr/bin/env python
"""Build the pre-indexed LanceDB for the public chat demo.

Minsky indexes itself here - the demo's corpus is this repo's own source, not
anything personal. Run this locally (Ollama must be running with
nomic-embed-text pulled), then upload the resulting directory to the Modal
volume:

    .venv/bin/python deploy/build_demo_index.py
    modal volume put minsky-demo-db deploy/build/db /
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from minsky.config import load_config  # noqa: E402
from minsky.index_state import IndexState  # noqa: E402
from minsky.indexer import index_project  # noqa: E402
from minsky.ollama_client import OllamaClient  # noqa: E402
from minsky.store import Store  # noqa: E402

VECTOR_DIM = 768
BUILD_DB_PATH = Path(__file__).resolve().parent / "build" / "db"


def main() -> None:
    cfg = load_config()
    cfg.db_path = BUILD_DB_PATH
    cfg.db_path.mkdir(parents=True, exist_ok=True)

    ollama = OllamaClient(cfg.ollama_url, cfg.embed_model)
    if not ollama.ping():
        print("error: cannot reach Ollama locally - start it before running this", file=sys.stderr)
        sys.exit(1)

    store = Store(cfg.db_path, VECTOR_DIM)
    state = IndexState(cfg.db_path / "index_state.json")

    start = time.time()
    stats = index_project("minsky_ai", REPO_ROOT, cfg, store, state, ollama)
    state.save()
    store.optimize()
    print(
        f"minsky_ai: +{stats['files_updated']} files, {stats['commits_indexed']} commits "
        f"({time.time() - start:.1f}s)"
    )
    print(f"built demo index at {cfg.db_path}")


if __name__ == "__main__":
    main()
