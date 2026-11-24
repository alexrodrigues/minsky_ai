from __future__ import annotations

import json
from pathlib import Path


class IndexState:
    """Tracks a content hash per (project, path) so unchanged files are skipped on reindex."""

    def __init__(self, state_path: Path):
        self.state_path = state_path
        if state_path.exists():
            self._data: dict[str, str] = json.loads(state_path.read_text())
        else:
            self._data = {}

    @staticmethod
    def _key(project: str, path: str) -> str:
        return f"{project}\x1f{path}"

    def get(self, project: str, path: str) -> str | None:
        return self._data.get(self._key(project, path))

    def set(self, project: str, path: str, content_hash: str) -> None:
        self._data[self._key(project, path)] = content_hash

    def paths_for_project(self, project: str) -> set[str]:
        prefix = f"{project}\x1f"
        return {k[len(prefix):] for k in self._data if k.startswith(prefix)}

    def remove(self, project: str, path: str) -> None:
        self._data.pop(self._key(project, path), None)

    def save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self._data))
